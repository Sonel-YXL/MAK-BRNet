
"""
Eval 对应 Evaluator：
    Evaluator：负责执行评估过程的组件或角色。它通常用于在训练过程中的验证阶段，使用验证集来评估模型的性能，并提供反馈以调整模型参数或进行早停。
"""

import torch
from Tools.base_manager import BaseManager
import os
from Datasets.base_evaluation_task import evaluation_OOD

class EvalModule(BaseManager):
    def __init__(self, params, **kwargs):
        super().__init__(params)

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
            
            
            self.models.load_state_dict(checkpoint["{}_state_dict".format('model')])

        
        if not checkpoint:
            pass
            
            

        
        if self.start_epoch != 1:  
            print("LOADED EPOCH: {}".format(self.start_epoch), flush=False)  
        else:  
            print("New train!")
        pass

    def evaluation(self, args):  
        self.models = self.models.to(self.device)
        self.models.eval()

        result_path = os.path.join(self.paths['output_folder'], 'result_' + self.dataset_dict["valid"].name)
        if not os.path.exists(result_path):  
            os.mkdir(result_path)

        self.models.eval()
        self.decoder.write_results(d_models=self.models,
                                   d_decoder=self.decoder,
                                   d_dataset=self.dataset_dict["valid"],
                                   d_result_path=result_path,
                                   d_device=self.device)

        if self.dataset_dict["valid"].name == 'dota':  
            merge_path = os.path.join(self.paths['output_folder'], 'merge_' + self.dataset_dict["valid"].name)
            if not os.path.exists(merge_path):
                os.mkdir(merge_path)
            self.dataset_dict["valid"].merge_crop_image_results(result_path, merge_path)
            
            file_arr = [id_inf['id_name'] for id_inf in self.dataset_dict['valid'].imgann_path_ids]
            
            
            
            

            map, ap = evaluation_OOD(os.path.join(os.path.join(self.paths['output_folder'],"result_dota", "Task1_{:s}.txt")),
                                     os.path.join(self.dataset_dict['valid'].imgann_path[0]['ann_file'], "{:s}.txt"),
                                     file_arr, 
                                     self.dataset_dict['valid'].category)
            print( "map",map,"ap", ap)
            return None
        else:  
            ap = self.dataset_dict["valid"].dec_evaluation(result_path)
            return ap



'''
def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


class Pytorch_model:
    def __init__(self, model_path, post_p_thre=0.7, gpu_id=None):
        
        
        
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False
        self.model = build_model(config['arch'])
        self.post_process = get_post_processing(config['post_processing'])
        self.post_process.box_thresh = post_p_thre
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

    def predict(self, img_path: str, is_output_polygon=False, short_size: int = 1024):
        
        
        
        
        
        
        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = resize_image(img, short_size)
        
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        batch = {'shape': [(h, w)]}
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.model(tensor)
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
            t = time.time() - start
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t


def save_depoly(model, input, save_path):
    traced_script_model = torch.jit.trace(model, input)
    traced_script_model.save(save_path)


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--model_path', default=r'model_best.pth', type=str)
    parser.add_argument('--input_folder', default='./test/input', type=str, help='img path for predict')
    parser.add_argument('--output_folder', default='./test/output', type=str, help='img path for output')
    parser.add_argument('--thre', default=0.3,type=float, help='the thresh of post_processing')
    parser.add_argument('--polygon', action='store_true', help='output polygon or box')
    parser.add_argument('--show', action='store_true', help='show result')
    parser.add_argument('--save_resut', action='store_true', help='save box and score to txt file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import pathlib
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from Utils.util import show_img, draw_bbox, save_result, get_file_list

    args = init_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    
    model = Pytorch_model(args.model_path, post_p_thre=args.thre, gpu_id=0)
    img_folder = pathlib.Path(args.input_folder)
    for img_path in tqdm(get_file_list(args.input_folder, p_postfix=['.jpg'])):
        preds, boxes_list, score_list, t = model.predict(img_path, is_output_polygon=args.polygon)
        img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list)
        if args.show:
            show_img(preds)
            show_img(img, title=os.path.basename(img_path))
            plt.show()
        
        os.makedirs(args.output_folder, exist_ok=True)
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(args.output_folder, img_path.stem + '_result.jpg')
        pred_path = os.path.join(args.output_folder, img_path.stem + '_pred.jpg')
        cv2.imwrite(output_path, img[:, :, ::-1])
        cv2.imwrite(pred_path, preds * 255)
        save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, args.polygon)
'''
