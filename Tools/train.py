

"""
Train 对应 Trainer：
    Trainer：负责执行训练过程的组件或角色。它管理模型的训练循环，包括前向传播、计算损失、反向传播和参数更新。
"""

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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Conv2d, Linear, InstanceNorm2d
from torch.nn.init import zeros_, ones_, kaiming_uniform_
from time import time, strftime, localtime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import os
import anyconfig

from Tools.base_manager import BaseManager


class TrainModule(BaseManager):
    def __init__(self, params, **kwargs):
        super().__init__(params)



'''











            ''''''
            
            
            
            
            
            
            
            
            
            if False: 
                
                

                
                post_process_dict = self.post_processing.get_target_instance(pred_temp)  
                

                
                box_temp = []
                for curr_polygons in post_process_dict['polygons']:
                        box = polygonToRotRectangle_batch(curr_polygons)  
                        box_temp.append(box)
                if len(box_temp) > 0:  
                    box_temp = np.array(box_temp)
                    post_process_dict['bboxes'] = torch.tensor(box_temp)  
                else:
                    box_temp = np.array([], dtype=np.float32).reshape(0, 5)
                    post_process_dict['bboxes'] = torch.tensor(box_temp)

                
                self.train_matrics.pro
'''

