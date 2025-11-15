import torch.nn.functional as F
import torch
from tqdm import tqdm
import os
import torch
import numpy as np
from Utils.DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast, py_cpu_nms_poly


def decode_prediction(predictions, dsets, input_w,input_h, index, down_ratio):
    predictions = predictions[0, :, :]  
    
    ori_image = dsets.load_image(index)  
    h, w, c = ori_image.shape

    pts0 = {cat: [] for cat in dsets.category}  
    scores0 = {cat: [] for cat in dsets.category}
    for pred in predictions:  
        cen_pt = np.asarray([pred[0], pred[1]], np.float32)  
        tt = np.asarray([pred[2], pred[3]], np.float32)
        rr = np.asarray([pred[4], pred[5]], np.float32)
        bb = np.asarray([pred[6], pred[7]], np.float32)
        ll = np.asarray([pred[8], pred[9]], np.float32)
        tl = tt + ll - cen_pt
        bl = bb + ll - cen_pt
        tr = tt + rr - cen_pt
        br = bb + rr - cen_pt
        score = pred[10]  
        clse = pred[11]  
        pts = np.asarray([tr, br, bl, tl], np.float32)  
        pts[:, 0] = pts[:, 0] * down_ratio / input_w * w  
        pts[:, 1] = pts[:, 1] * down_ratio / input_h * h
        pts0[dsets.category[int(clse)]].append(pts)  
        scores0[dsets.category[int(clse)]].append(score)  
    return pts0, scores0


def non_maximum_suppression(pts, scores):
    nms_item = np.concatenate([pts[:, 0:1, 0],
                               pts[:, 0:1, 1],
                               pts[:, 1:2, 0],
                               pts[:, 1:2, 1],
                               pts[:, 2:3, 0],
                               pts[:, 2:3, 1],
                               pts[:, 3:4, 0],
                               pts[:, 3:4, 1],
                               scores[:, np.newaxis]], axis=1)
    nms_item = np.asarray(nms_item, np.float64)
    keep_index = py_cpu_nms_poly_fast(dets=nms_item, thresh=0.1)  
    return nms_item[keep_index]

class DecDecoder(object):
    def __init__(self, K, conf_thresh, num_classes):
        self.K = K  
        self.conf_thresh = conf_thresh  
        self.num_classes = num_classes  

    def _topk(self, scores):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_clses = (topk_ind // self.K).int()
        topk_inds = self._gather_feat( topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)  
        keep = (hmax == heat).float()
        return heat * keep  

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def ctdet_decode(self, pr_decs):
        heat = pr_decs['hm']  
        wh = pr_decs['wh']
        reg = pr_decs['reg']
        cls_theta = pr_decs['cls_theta']

        batch, c, height, width = heat.size()  
        heat = self._nms(heat)  

        scores, inds, clses, ys, xs = self._topk(heat)  
        reg = self._tranpose_and_gather_feat(reg, inds)  
        reg = reg.view(batch, self.K, 2)  
        xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]  
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]  
        clses = clses.view(batch, self.K, 1).float()  
        scores = scores.view(batch, self.K, 1)  
        wh = self._tranpose_and_gather_feat(wh, inds)  
        wh = wh.view(batch, self.K, 10)  
        
        cls_theta = self._tranpose_and_gather_feat(cls_theta, inds)  
        cls_theta = cls_theta.view(batch, self.K, 1)  
        mask = (cls_theta>0.8).float().view(batch, self.K, 1)  
        
        tt_x = (xs+wh[..., 0:1])*mask + (xs)*(1.-mask)
        tt_y = (ys+wh[..., 1:2])*mask + (ys-wh[..., 9:10]/2)*(1.-mask)
        rr_x = (xs+wh[..., 2:3])*mask + (xs+wh[..., 8:9]/2)*(1.-mask)
        rr_y = (ys+wh[..., 3:4])*mask + (ys)*(1.-mask)
        bb_x = (xs+wh[..., 4:5])*mask + (xs)*(1.-mask)
        bb_y = (ys+wh[..., 5:6])*mask + (ys+wh[..., 9:10]/2)*(1.-mask)
        ll_x = (xs+wh[..., 6:7])*mask + (xs-wh[..., 8:9]/2)*(1.-mask)
        ll_y = (ys+wh[..., 7:8])*mask + (ys)*(1.-mask)
        
        detections = torch.cat([xs,                      
                                ys,                      
                                tt_x,
                                tt_y,
                                rr_x,
                                rr_y,
                                bb_x,
                                bb_y,
                                ll_x,
                                ll_y,
                                scores,
                                clses],
                               dim=2)  

        index = (scores > self.conf_thresh).squeeze(0).squeeze(1)  
        detections = detections[:,index,:]
        return detections.data.cpu().numpy()  

    
    def write_results(self, d_models, d_decoder, d_dataset, d_result_path, d_device):
        """传入模型,数据集,输出文件夹路径,设备"""
        
        results = {cat: {img_id['id_name']: [] for img_id in d_dataset.imgann_path_ids} for cat in d_dataset.category}  
        for index in range(len(d_dataset)):  
            data_dict = d_dataset.__getitem__(index)  
            image = data_dict['image'].to(d_device)
            img_id = data_dict['img_id']
            image_w = data_dict['image_w']  
            image_h = data_dict['image_h']

            with torch.no_grad():
                pr_decs = d_models(image)  

            decoded_pts = []
            decoded_scores = []
            if str(d_device)!="cpu":torch.cuda.synchronize(d_device)  
            predictions = d_decoder.ctdet_decode(pr_decs)  
            pts0, scores0 = decode_prediction(predictions, d_dataset, d_dataset.input_w,d_dataset.input_h, index, d_models.down_ratio)  
            decoded_pts.append(pts0)
            decoded_scores.append(scores0)

            
            for cat in d_dataset.category:
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
                    results[cat][img_id].extend(nms_results)  
            
            

        for cat in d_dataset.category:  
            if cat == 'background':
                continue
            with open(os.path.join(d_result_path, 'Task1_{}.txt'.format(cat)), 'w') as f:
                for img_id in results[cat]:
                    for pt in results[cat][img_id]:
                        f.write('{} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                            img_id, pt[8], pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]))
