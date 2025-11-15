"""
Test 对应 Tester：
    Tester：负责执行测试过程的组件或角色。它使用训练好的模型在测试集上进行评估，以衡量模型在未见过的数据上的性
"""
import copy

import torch

from Tools.base_manager import BaseManager
import os
import time
import numpy as np
from Models.BBAVectors.decoder import decode_prediction,non_maximum_suppression
import cv2
import matplotlib.pyplot as plt
import matplotlib
from Utils.DOTA_devkit.ResultMerge_multi_process import mergebypoly
matplotlib.use('TkAgg')


class TestModule(BaseManager):
    def __init__(self, params, **kwargs):
        super().__init__(params)

    def imshow_heatmap(self, pr_dec, images):
        wh = pr_dec['wh']
        hm = pr_dec['hm']
        cls_theta = pr_dec['cls_theta']
        wh_w = wh[0, 0, :, :].data.cpu().numpy()
        wh_h = wh[0, 1, :, :].data.cpu().numpy()
        hm = hm[0, 0, :, :].data.cpu().numpy()
        cls_theta = cls_theta[0, 0, :, :].data.cpu().numpy()
        images = np.transpose((images.squeeze(0).data.cpu().numpy() + 0.5) * 255, (1, 2, 0)).astype(np.uint8)
        wh_w = cv2.resize(wh_w, (images.shape[1], images.shape[0]))
        wh_h = cv2.resize(wh_h, (images.shape[1], images.shape[0]))
        hm = cv2.resize(hm, (images.shape[1], images.shape[0]))
        fig = plt.figure(1)
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.set_xlabel('width')
        ax1.imshow(wh_w)
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.set_xlabel('height')
        ax2.imshow(wh_h)
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.set_xlabel('center hm')
        ax3.imshow(hm)
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_xlabel('input image')
        ax5.imshow(cls_theta)
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_xlabel('input image')
        ax6.imshow(images)
        plt.savefig('/media/sonel/0632509B033959B0/代码以及实验结果/可视化结果/temp_show/heatmap.png')


    def load_model(self):
        """ 不对优化器进行加载
        保存的checkpoint里面是7个字典的信息 {
            'optimizer_state_dict'：dict:2{}，
            'epoch':0，
            'scaler_state_dict':dict:0{}，
            'best':0,
            'model_state_dict':(orderedDIct)
        }
        """
        
        checkpoint = None  
        if self.params["trainer"]["resume_checkpoint"] is not None:  
            
            checkpoint_path = self.params["trainer"]["resume_checkpoint"]
            
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage )  

            self.start_epoch = checkpoint["epoch"]  
            
            
            self.models.load_state_dict(checkpoint["{}_state_dict".format('model')],strict=False)  

        
        if not checkpoint:
            pass
            
            

        
        if self.start_epoch != 1:  
            print("LOADED EPOCH: {}".format(self.start_epoch), flush=False)  
        else:  
            print("New train!")
        pass

    def write_result(self, mergedata = False):
        """这个是自己写的,对测试数据写出"""
        result_path = os.path.join(self.paths['output_folder'], 'result_' + self.dataset_dict["valid"].name)
        if not os.path.exists(result_path):  
            os.mkdir(result_path)
        self.models = self.models.to(self.device)  
        self.models.eval()
        self.decoder.write_results(d_models=self.models,  
                                   d_decoder=self.decoder,
                                   d_dataset=self.dataset_dict["test"],
                                   d_result_path=result_path,
                                   d_device=self.device)
        if mergedata:
            if self.dataset_dict['test'].name == 'dota':  
                mergepath = os.path.join(self.paths['output_folder'],
                                         os.path.normpath(os.path.basename(
                                             self.paths['output_folder'])) + "_merge03_" + str(self.start_epoch))
                if not os.path.exists(mergepath):
                    os.makedirs(mergepath)

                mergebypoly(result_path, mergepath)  



    def modeltest(self, draw_line):
        self.models = self.models.to(self.device)
        self.models.eval()

        result_path = os.path.join(self.paths['output_folder'], 'result_' + self.dataset_dict["valid"].name)
        if not os.path.exists(result_path):  
            os.mkdir(result_path)

        self.models.eval()

        data_loader = torch.utils.data.DataLoader(self.dataset_dict['valid'],  
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)

        total_time = []
        for cnt, data_dict in enumerate(data_loader):
            image = data_dict['image'][0].to(self.device)
            
            
            
            begin_time = time.time()
            with torch.no_grad():
                pr_decs = self.models(image)

            self.imshow_heatmap(pr_decs, image)

            
            decoded_pts = []
            decoded_scores = []
            predictions = self.decoder.ctdet_decode(pr_decs)
            
            pts0, scores0 = decode_prediction(predictions, self.dataset_dict['valid'], 608,608, cnt, 4)
            
            
            
            
            decoded_pts.append(pts0)
            decoded_scores.append(scores0)
            
            results = {cat: [] for cat in self.dataset_dict['valid'].category}
            for cat in self.dataset_dict['valid'].category:
                if cat == 'background':
                    continue
                pts_cat = []
                scores_cat = []
                for pts0, scores0 in zip(decoded_pts, decoded_scores):
                    pts_cat.extend(pts0[cat])
                    scores_cat.extend(scores0[cat])
                pts_cat = np.asarray(pts_cat, np.float32)
                scores_cat = np.asarray(scores_cat, np.float32)
                if pts_cat.shape[0]:
                    nms_results = non_maximum_suppression(pts_cat, scores_cat)
                    results[cat].extend(nms_results)

            end_time = time.time()
            total_time.append(end_time - begin_time)

            
            if draw_line:
                ori_image = self.dataset_dict['valid'].load_image(cnt)
                ori_image_gt = copy.deepcopy(ori_image)
                
                ori_image_0 = copy.deepcopy(ori_image)
                ori_image_ = self.dataset_dict['valid'].load_image(cnt)
                ori_image_gt_ = copy.deepcopy(ori_image_)

                height, width, _ = ori_image.shape  
                
                
                
                
                for cat_i,cat in enumerate(self.dataset_dict['valid'].category):
                    if cat == 'background':
                        continue
                    result = results[cat]
                    for pred in result:
                        score = pred[-1]
                        tl = np.asarray([pred[0], pred[1]], np.float32)
                        tr = np.asarray([pred[2], pred[3]], np.float32)
                        br = np.asarray([pred[4], pred[5]], np.float32)
                        bl = np.asarray([pred[6], pred[7]], np.float32)

                        tt = (np.asarray(tl, np.float32) + np.asarray(tr, np.float32)) / 2
                        rr = (np.asarray(tr, np.float32) + np.asarray(br, np.float32)) / 2
                        bb = (np.asarray(bl, np.float32) + np.asarray(br, np.float32)) / 2
                        ll = (np.asarray(tl, np.float32) + np.asarray(bl, np.float32)) / 2

                        box = np.asarray([tl, tr, br, bl], np.float32)  
                        cen_pts = np.mean(box, axis=0)  
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1,
                                                     self.dataset_dict['valid'].color_pans[cat_i],2,cv2.LINE_8)
                        ori_image_ = cv2.drawContours(ori_image_, [np.int0(box)], -1,
                                                     self.dataset_dict['valid'].color_pans[cat_i],2,cv2.LINE_8)
                        
                        
                        
                        
                        cv2.putText(ori_image, '{:.2f} {}'.format(score, cat), (int(box[1][0]), int(box[1][1])),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1,1)
                
                
                gt_anno = self.dataset_dict['valid'].load_annotation(cnt)
                
                for pt_i,pts_4 in enumerate(gt_anno['pts']):  
                    bl = pts_4[0, :]
                    tl = pts_4[1, :]
                    tr = pts_4[2, :]
                    br = pts_4[3, :]
                    cen_pts = np.mean(pts_4, axis=0)
                    box = np.asarray([bl, tl, tr, br], np.float32)
                    box = np.int0(box)
                    
                    
                    cat_temp_value = self.dataset_dict['valid'].category[gt_anno['cat'][pt_i]]
                    color_temp_value = self.dataset_dict['valid'].color_pans[gt_anno['cat'][pt_i]]
                    cv2.drawContours(ori_image_gt, [box], 0, color_temp_value, 2)
                    cv2.drawContours(ori_image_gt_, [box], 0, color_temp_value, 2)
                    cv2.putText(ori_image_gt,
                                cat_temp_value, (int(tr[0]), int(tr[1])),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5,(255, 255, 255), 1, 1)

                
                
                
                
                

                if False:
                    
                    
                    path = "/media/sonel/0632509B033959B0/sh-temp-bba/"
                    pathindex = cnt//300+1
                    group_folder = path+str(pathindex)+"/"
                    if not os.path.exists(group_folder):
                        os.makedirs(group_folder)
                    
                    if len(decoded_pts[0]['ship']) > 8 and len(decoded_pts[0]['harbor']) < 3:  
                        cv2.imwrite(group_folder+str(cnt)+"_origin.png",ori_image_0)  
                        cv2.imwrite(group_folder+str(cnt)+"_predl.png", ori_image)  
                        cv2.imwrite(group_folder+str(cnt)+"_pred.png", ori_image_)  

                    
                    
                    
                    
                    
                    
                    print(cnt, "/", len(self.dataset_dict['valid']))
                k = cv2.waitKey(0) & 0xFF
                if k == ord('q'):
                    cv2.destroyAllWindows()
                    exit()
            

        total_time = total_time[1:]
        print('avg time is {}'.format(np.mean(total_time)))
        print('FPS is {}'.format(1. / np.mean(total_time)))

