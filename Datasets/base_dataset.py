

import os.path as osp
import sys
import warnings
from collections import OrderedDict
from abc import ABC, abstractmethod  
import os
import glob
import json
import re
import cv2
from PIL import Image
import copy
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_sharpness, equalize
from torchvision.transforms import RandomPerspective, RandomCrop, ColorJitter
from torchvision import transforms


from Datasets.Transform_OOD import build_transform
from Datasets.Label_OOD import build_label


def collater(data):
    out_data_dict = {}
    for name in data[0]:  
        out_data_dict[name] = []
    for sample in data:  
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:  
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict


class BaseDataset(Dataset):
    """
    加载标签,数据增强,
    """

    def __init__(self, phase, params):
        """每次创建一个数据集,进行函数初始化，数据的加载  创建phase状态的数据集
        :phase str,"train","valid"或者"test"
        """

        
        self.params = params

        self.name = params.name
        self.phase = phase
        self.input_h = params.input_img_hw[0]
        self.input_w = params.input_img_hw[1]
        self.category = params.dataset_class
        self.color_pans = params.color
        self.num_classes = len(self.category)
        self.cat_ids = {cat: i for i, cat in enumerate(self.category)}
        self.max_objs = 500
        
        self.imgann_path = params[phase].dataset.data_path  
        self.skipEmptyLabels = params[phase].dataset.skipEmptyLabels  
        if self.phase in ["train", "valid"]:  
            self.imgann_path_ids = self.load_img_ids(self.imgann_path)  
        if self.phase == "test":
            
            self.imgann_path_ids = self.load_img_only(self.imgann_path)
            pass  

        self.augmentation = build_transform(self.params[self.phase].dataset.augmentation)
        self.net_label = build_label(self.params[self.phase].dataset.net_label)

        
        
        

        pass

    def load_img_only(self, imgann_path_list):
        """仅在创建测试集时用到,返回所有图像的列表"""
        ori_data_infos = []
        for imgann_path in imgann_path_list:  
            
            
            imageExtension = glob.glob(imgann_path['img_file'] + '/*')[0]
            imageExtension = imageExtension[imageExtension.rfind('.')+1:]
            img_files = glob.glob(imgann_path['img_file'] + '/*.'+imageExtension)  
            t_dataset = tqdm(img_files, leave=True,desc="数据集加载  -> ")
            for ann_file in t_dataset:
                id_name = ann_file[:ann_file.rfind('.')].split('/')[-1]
                img_name = os.path.join(imgann_path['img_file'], id_name + '.' + imageExtension)  
                data_info = {
                    'id_name': id_name,
                    "imgpath": img_name,
                }
                ori_data_infos.append(data_info)  
        return ori_data_infos


    def load_img_ids(self, imgann_path_list):
        ori_data_infos = []
        for imgann_path in imgann_path_list:  
            ann_Extension = glob.glob(imgann_path['ann_file'] + '/*')[0]
            ann_Extension=ann_Extension[ann_Extension.rfind('.')+1:]
            imageExtension = glob.glob(imgann_path['img_file'] + '/*')[0]
            imageExtension=imageExtension[imageExtension.rfind('.')+1:]
            ann_files = glob.glob(imgann_path['ann_file'] + '/*.'+ann_Extension)  
            t_dataset = tqdm(ann_files, leave=True,desc="数据集加载  -> ")
            for ann_file in t_dataset:
                if self.skipEmptyLabels and os.path.getsize(ann_file) == 0:  
                    continue
                id_name = ann_file[:ann_file.rfind('.')].split('/')[-1]
                img_name = os.path.join(imgann_path['img_file'], id_name + '.' + imageExtension)  
                data_info = {
                    'id_name': id_name,
                    "imgpath": img_name,
                    "annpath": ann_file
                }
                ori_data_infos.append(data_info)  
        return ori_data_infos

    def load_image(self, index):
        
        
        
        
        temp_id = self.imgann_path_ids[index]  
        imgFile = temp_id['imgpath']
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)  
        return img

    def load_annotation(self, index):
        image = self.load_image(index)
        h, w, c = image.shape
        valid_pts = []
        valid_cat = []
        valid_dif = []
        with open(self.imgann_path_ids[index]['annpath'], 'r') as f:
            for i, line in enumerate(f.readlines()):
                
                obj = re.split(r"[,\s]+", line.strip())  
                if len(obj)>8:  
                    x1 = min(max(float(obj[0]), 0), w - 1)
                    y1 = min(max(float(obj[1]), 0), h - 1)
                    x2 = min(max(float(obj[2]), 0), w - 1)
                    y2 = min(max(float(obj[3]), 0), h - 1)
                    x3 = min(max(float(obj[4]), 0), w - 1)
                    y3 = min(max(float(obj[5]), 0), h - 1)
                    x4 = min(max(float(obj[6]), 0), w - 1)
                    y4 = min(max(float(obj[7]), 0), h - 1)
                    
                    xmin = max(min(x1, x2, x3, x4), 0)
                    xmax = max(x1, x2, x3, x4)
                    ymin = max(min(y1, y2, y3, y4), 0)
                    ymax = max(y1, y2, y3, y4)
                    if ((xmax - xmin) > 10) and ((ymax - ymin) > 10):  
                        valid_pts.append([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
                        valid_cat.append(self.cat_ids[obj[8]])
                        valid_dif.append(int(obj[9]))
        f.close()
        annotation = {}
        annotation['pts'] = np.asarray(valid_pts, np.float32)
        annotation['cat'] = np.asarray(valid_cat, np.int32)
        annotation['dif'] = np.asarray(valid_dif, np.int32)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        return annotation

    def __len__(self):
        return len(self.imgann_path_ids)

    def apply_data_augmentation(self, data: dict) -> dict:
        """
        OOD：只对单一的数据（字典类型）进行数据增强操作。输出也是字典形式
        :param data: data.sample数据
        :return: 数据增强后的图片和标注。
        """
        res_data = dict()  

        if len(self.augmentation) == 0:  
            return data

        
        for aug_ in self.augmentation:
            res_data = aug_(data)
        return res_data

    @staticmethod
    def apply_list(transform_list, t_image, t_label):
        """转换列表,图像,标签{cat,pts,dif}
        返回转换图像,转换标签
        """
        if len(transform_list) == 0:  
            return t_image, t_label

        for aug_ in transform_list:  
            t_image, t_label = aug_(t_image, t_label)
        return t_image, t_label

    @staticmethod
    def apply_label(transform_list, t_image,t_label):
        """转换列表,图像,标签{cat,pts,dif}
        返回转换图像,转换标签
        """
        if len(transform_list) == 0:  
            return t_label

        for aug_ in transform_list:  
            t_label = aug_(t_image,t_label)
        return t_label

    def processing_test(self, image, input_h, input_w):
        image = cv2.resize(image, (input_w, input_h))  
        out_image = image.astype(np.float32) / 255.  
        out_image = out_image - 0.5  
        out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)  
        out_image = torch.from_numpy(out_image)
        return out_image

    def __getitem__(self, idx):
        image = self.load_image(idx)  
        image_h, image_w, c = image.shape

        
        
        if self.phase == 'train':  
            annotation = self.load_annotation(idx)  
            image, annotation = self.apply_list(self.augmentation, image, annotation)  
            
            
            
            sample = self.apply_label(self.net_label, image, annotation)
            
            return sample  

        elif self.phase == 'test' or self.phase == 'valid':
            img_id = self.imgann_path_ids[idx]['id_name']  
            image = self.processing_test(image, self.input_h, self.input_w)  
            return {'image': image,  
                    'img_id': img_id,
                    'image_w': image_w,  
                    'image_h': image_h}  

        

    def dec_evaluation(self, result_path):
        """这个需要进行重写,比如dota数据集是可以不调用这个,hrsc数据集则可以进行评测的"""
        return None

    @staticmethod
    def get_transforms(transforms_config):
        tr_list = []
        for item in transforms_config:
            if 'args' not in item:
                args = {}
            else:
                args = item['args']
            cls = getattr(transforms, item['type'])(**args)
            tr_list.append(cls)
        tr_list = transforms.Compose(tr_list)
        return tr_list

    def __repr__(self):
        """规范化类输出，可以通过manager进行调用。
        """
        
        
        

        shape_img = self.__getitem__(0)['input'].shape[1:]

        strthis = ("数据集加载: 数量: {}\t 数据集样例:图像大小: {}\n".format(len(self.imgann_path_ids), shape_img)
                   + "数据集总类别: {}\n".format(self.category))
        return "<%s.%s> \n%s \n" % (self.__class__.__module__, self.__class__.__name__, strthis)

