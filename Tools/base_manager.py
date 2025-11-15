"""trainer,evaluate和test都能够继承的一个类"""
import datetime
import anyconfig
import os
import random
import shutil
from tqdm import tqdm,trange  
import cv2
from addict import Dict  
import copy
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.cuda.amp import autocast  
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Conv2d, Linear, InstanceNorm2d
from torch.nn.init import zeros_, ones_, kaiming_uniform_  
import torch
from torch.utils.data import Dataset, DataLoader
from time import time, strftime, localtime
from torch.cuda.amp import GradScaler
from torch.optim import Adam  
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import os
import anyconfig  

from Datasets import build_dataset
from Datasets.base_dataset import *
from Models import build_model
from Models import build_decoder
from Models import build_loss
from thop import profile




def init_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)


class BaseManager:
    def __init__(self, args):
        """ 初始化参数，硬件信息，文件夹创建，数据集加载，模型加载
        """
        self.writer = None  
        self.paths = None  

        self.params = Dict(copy.deepcopy(args))  
        self.device = torch.device("cuda:0" if self.params.trainer.device.lower() == "gpu" else "cpu")  
        init_seed(self.params.trainer.seed)  
        self.init_hardware_config()  
        self.init_paths()

        self.phases = self.params.data.phases  
        self.dataset_dict = {}
        if "train" in self.phases:
            self.dataset_dict['train'] = build_dataset("train", self.params["data"])
            self.train_loader = self.load_dataloaders(self.dataset_dict['train'])
        if "valid" in self.phases:
            self.dataset_dict['valid'] = build_dataset("valid", self.params["data"])
        if "test" in self.phases:
            pass  
            self.dataset_dict['test'] = build_dataset("test", self.params["data"])

        print(self.dataset_dict['train'])  

        self.start_epoch = 1
        self.total_epoch = self.params["trainer"]["epochs"]  
        self.current_epoch = -1  

        if True:  
            self.models = build_model(self.params["model"])
            dummy_input = torch.randn(1, 3, 608, 608)
            
            flops, params = profile(self.models, (dummy_input,), verbose=False)
            print('flops: ', flops, 'params: ', params)
            print('1e6 flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
            print('flops: %.2f G, params: %.2f M' % (flops / (1024.0 * 1024 * 1024), params / (1024.0 * 1024)))

        
        self.models = build_model(self.params["model"])  

        self.criterion = build_loss(self.params["model"]["module_loss"])
        self.optimizer = eval(self.params["trainer"]["optimizer"]["type"])(self.models.parameters(), **self.params["trainer"]["optimizer"]["args"])
        self.scheduler = eval(self.params["trainer"]["lr_scheduler"]["type"])(self.optimizer,  **self.params["trainer"]["lr_scheduler"]["args"])
        
        
        

        self.load_model()  

        
        self.begin_time = None  

        self.best = None  
        self.save_memory = self.params['trainer']['save_memory']  


        self.max_mem_usage_by_epoch = list()  

        
        self.decoder = build_decoder(self.params["model"]["decoder"])  

    def init_hardware_config(self):
        """输出服务器的参数,设置随机数种子
        """
        if str(self.device) != "cpu":  
            print('Pytorch版本:{}    CUDA版本:{}    显卡数量:{}'.format(torch.__version__, torch.version.cuda,
                                                                        torch.cuda.device_count()))
            GPU_inf = torch.cuda.get_device_properties(self.device)
            print("GPU型号：{}  ,显存：{}M  ,线程数：{},  CUDA算力：{}".format(GPU_inf.name,
                                                                           int(GPU_inf.total_memory / (1024 * 1024)),
                                                                           GPU_inf.multi_processor_count,
                                                                           torch.cuda.get_device_capability()))
        else:
            print("强制使用CPU进行 !\n")

    def init_paths(self):
        """创建输出的目录等,并将路径保存在self.path下。
        :return:
        """
        current_time = datetime.datetime.now().strftime("%m%d%H")  
        project_output_path = os.path.join(self.params.rootdir, "Outputs")
        output_path = os.path.join(project_output_path, self.params["trainer"]["output_dir"].replace("time",current_time))
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.makedirs(output_path, exist_ok=True)  
        checkpoints_path = os.path.join(output_path, "checkpoints")  
        os.makedirs(checkpoints_path, exist_ok=True)
        log_path = os.path.join(output_path, "log")
        os.makedirs(log_path, exist_ok=True)  

        
        self.paths = {
            "log": log_path,
            "checkpoints": checkpoints_path,
            "output_folder": output_path
        }

        self.writer = SummaryWriter(self.paths["log"])
        anyconfig.dump(self.params, os.path.join(self.paths['output_folder'], 'config.yaml')) 

    def load_dataloaders(self,dataset_f):
        """进行数据加载器"""
        
        
        
        
        
        
        fn = eval(self.params.data.train.loader.pop('collate_fn'))
        return DataLoader(dataset_f, collate_fn=fn , **self.params.data.train.loader)

    def save_model(self, name, epoch, keep_weights=False):
        """  last_15,best_15,model_15
        功能：保存模型权重文件，OOD：这里自己感觉可以根据情况进行保存模型参数。权重优化器学习率训练间隔最优指标（last、best）
        :param epoch: int类型，此次训练的epoch数量
        :param name: str类型，如：'bset'，"last"
        :param keep_weights: bool类型 是否保存以前的历史文件，默认是不保存的
        :return:
            这里参考了pytorch.dbnet的一些写法。可以对以下文件进行保存
                epoch：当前训练的轮次数量
                state_dict：权重模型权重。
                optimizer：优化器状态
                metrics：评价指标信息。这里考虑下，应该可以保存最优的状态信息。
        """
        path = os.path.join(self.paths["checkpoints"], "{}_{}.pt".format(name, epoch))  

        
        for filename in os.listdir(self.paths["checkpoints"]):  
            
            if (not keep_weights) and (name in filename):  
                
                del_file_name_temp = os.path.join(self.paths["checkpoints"], filename)
                os.remove(del_file_name_temp)

        content = {
            'epoch': epoch,  
            'optimizer_state_dict': self.optimizer.state_dict(),  
            
            'best': self.best,  
            "model_state_dict": self.models.state_dict(),
            
        }

        
        torch.save(content, path)

    def load_model(self):
        """
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

            
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)  

        
        if not checkpoint:
            pass
            
            

        
        if self.start_epoch != 1:  
            print("LOADED EPOCH: {}".format(self.start_epoch), flush=False)  
        else:  
            print("New train!")
        pass

    def run_epoch(self, phase):  
        """训练一个epoch的方法,返回总体loss(在想要不要返回每个部分的loss)"""

        if phase == 'train':
            self.models.train()
        else:
            self.models.eval()

        
        
        t = tqdm(self.train_loader, leave=True,
                 desc="{} {}/{}".format(strftime("%m/%d %H:%M:%S", localtime()),
                                        self.current_epoch, self.total_epoch), unit=' B')
        running_loss = 0.
        vis_value_dict = {}
        for current_batch, batch_data in enumerate(t):  
            for name in batch_data:  
                batch_data[name] = batch_data[name].to(device=self.device, non_blocking=True)
            if phase == 'train':
                self.optimizer.zero_grad()
                
                with torch.enable_grad():
                    pr_decs = self.models(batch_data['input'])
                    loss, display_values = self.criterion(pr_decs, batch_data)
                    loss.backward()
                    self.optimizer.step()
                
                
            else:
                with torch.no_grad():
                    pr_decs = self.models(batch_data['input'])
                    loss = self.criterion(pr_decs, batch_data)
            running_loss += loss.item()

            t.set_postfix(values="loss:"+str(round(running_loss/(current_batch+1),4)))  

            for key in display_values.keys():
                
                
                self.writer.add_scalar('TRAIN/LOSS/{}'.format(key), display_values[key], (self.current_epoch-1)*len(self.train_loader)+current_batch)

            for temp_key, temp_value in display_values.items():  
                temp_value = round(temp_value, 4)
                vis_value_dict.setdefault(temp_key, []).append(temp_value)
            
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss, vis_value_dict


        pass
    def train_net(self):
        
        focus_metric_name = self.params["metric"]["focus_metric"]  
        metrics_name = self.params["metric"]["train_metrics"]  
        display_values = None  

        self.models.to(self.device)  
        criterion = self.criterion

        print("="*50+"训练阶段"+"="*50)
        self.begin_time = time()  
        train_loss = []  
        ap_list = []  
        save_best = None  
        save_loss = None  
        for self.current_epoch in range(self.start_epoch, self.total_epoch+1):
            epoch_loss, vis_epoch = self.run_epoch("train")  
            self.scheduler.step(self.current_epoch)  

            train_loss.append(epoch_loss)
            np.savetxt(os.path.join(self.paths["log"], 'train_loss.txt'), train_loss, fmt='%.6f')  

            
            if 'valid' in self.phases and self.current_epoch % self.params.trainer.eval_on_valid == 0:
                
                mAP, aps = self.dec_eval()  
                print("mAP",mAP)
                print("aps",aps)
                ap_list.append(mAP)
                np.savetxt(os.path.join(self.paths['log'], 'ap_list.txt'), ap_list, fmt='%.6f')
                
                metric_save = {"mAP": mAP, "loss": epoch_loss}
                mkey = next(iter(self.params.trainer.best_model_metric))
                if (save_best is None
                        or (metric_save[mkey] > save_best and self.params.trainer.best_model_metric[mkey] == "high")
                        or (metric_save[mkey] < save_best and self.params.trainer.best_model_metric[mkey] == "low")):
                    save_best = metric_save[mkey]
                    self.best = save_best
                    self.save_model(epoch=self.current_epoch, name="best_"+mkey, keep_weights=False)

            
            self.save_model(epoch=self.current_epoch, name="last")

            if (save_loss is None) or epoch_loss < save_loss:  
                save_loss = epoch_loss
                self.save_model(epoch=self.current_epoch, name="best_Loss")

            if self.save_memory:  
                self.update_memory_consumption()  
            

            interval_save_weights = self.params["trainer"]["interval_save_weights"]  
            if interval_save_weights and self.current_epoch % interval_save_weights == 0:
                self.save_model(epoch=self.current_epoch, name="weigths", keep_weights=True)  
            self.writer.flush()  

    
    
    

    
    def dec_eval(self):
        """
        对传入的数据集dataset_d进行评估,返回总体map值
        """
        result_path = os.path.join(self.paths['output_folder'], 'result_' + self.dataset_dict["valid"].name)
        if not os.path.exists(result_path):  
            os.mkdir(result_path)
        self.models.eval()
        self.decoder.write_results(d_models=self.models,
                                   d_decoder=self.decoder,
                                   d_dataset=self.dataset_dict["valid"],
                                   
                                   d_result_path=result_path,
                                   d_device=self.device)

        mAP, aps = self.dataset_dict["valid"].dec_evaluation(result_path)  
        return mAP, aps  
        

    
    @staticmethod
    def set_model_learnable(model, learnable=True):
        """
        功能：使用迁移学习的时候，设置部分模型梯度不可导。
        :param model: 部分模型，如:encoder
        :param learnable: 是否可以反向传播
        """
        for p in list(model.parameters()):
            p.requires_grad = learnable  

    
    @staticmethod
    def weights_init(m):
        """
        功能:从头开始模型训练的权重初始化,这个被.apply(fn)调用  在init中load_model调用
        :param m: 模型的每一层进行遍历
        :return:
        """
        
        if isinstance(m, Conv2d) or isinstance(m, Linear):  
            if m.weight is not None:  
                kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                zeros_(m.bias)  
        elif isinstance(m, InstanceNorm2d):  
            
            
            if m.weight is not None:  
                
                ones_(m.weight)  
            if m.bias is not None:  
                zeros_(m.bias)

    
    def update_memory_consumption(self):
        """更新显存消耗,txt文件进行更新
        :return: 保存的memory.txt文件
        """
        
        self.max_mem_usage_by_epoch.append(torch.cuda.max_memory_allocated())
        
        torch.cuda.reset_peak_memory_stats()
        with open(os.path.join(self.paths["log"], "memory.txt"), 'a', encoding='utf-8') as f:  
            current = round(self.max_mem_usage_by_epoch[-1] / 1e9, 2)  
            max = round(np.max(self.max_mem_usage_by_epoch) / 1e9, 2)  
            min = round(np.min(self.max_mem_usage_by_epoch) / 1e9, 2)  
            median = round(np.median(self.max_mem_usage_by_epoch) / 1e9, 2)  
            mean = round(np.mean(self.max_mem_usage_by_epoch) / 1e9, 2)  
            f.write("E{} - Current: {} Go - Max: {} Go - Min: {} Go - Mean: {} Go - Median: {} Go\n".format(
                self.current_epoch, current, max, min, mean, median))

    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    

    @staticmethod
    def merge_batch_data(pred_total, gt_total, pred_batch_data_dict, gt_batch_data_dict):
        """在推理的batch数据中对相关的数据进行合并
        :param pred_total: 总的预测数据
        :param gt_total: 总的gt数据
        :param pred_batch_data_dict: 需要合并的预测数据
        :param gt_batch_data_dict: 需要合并的gt数据
        :return: 返回总的pred和gt list
        """
        res_pred = pred_total
        res_gt = gt_total
        
        for i_ in range(len(gt_batch_data_dict['filename'])):
            gt_temp_dict = dict()
            for batch_data_key, batch_data_value in gt_batch_data_dict.items():
                if batch_data_key in ['filename', 'target_nums', 'labels', 'polygons', 'ignore_tags', 'scores']:
                    gt_temp_dict[batch_data_key] = batch_data_value[i_]
                    if batch_data_key in ['labels', 'polygons']:  
                        gt_temp_dict[batch_data_key] = batch_data_value[i_][:gt_batch_data_dict['target_nums'][i_]]
                        gt_temp_dict[batch_data_key] = gt_temp_dict[batch_data_key].detach().cpu().numpy()
            res_gt.append(gt_temp_dict)
        
        res_pred = res_pred + pred_batch_data_dict
        return res_pred, res_gt

    
    def train_batch(self, batch_data):
        raise NotImplementedError

    
    def evaluate_batch(self, batch_data):
        raise NotImplementedError


