
'''
@File       :   DBNet.py
@Modify Time:   2024/4/14 下午12:53
@Author     :   Sonel
@Version    :   1.0
@Contact    :   sonel@qq.com
@Description:
'''

import copy
import math
import warnings
import numpy as np
import cv2
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from Models.FPNBBA.module_backbone import build_backbone
from Models.FPNBBA.module_neck import build_neck

from addict import Dict


class resfpn(nn.Module):
    def __init__(self, backbone_name, backbone_args, fpn_name, fpn_args):
        """
        OOD:这里进行重新构建了,根据一个配置文件进行构建模型.
        :param model_config:
        """
        super().__init__()

        self.backbone = build_backbone(backbone_name, **backbone_args)
        self.neck = build_neck(fpn_name,  in_channels=self.backbone.out_channels, **fpn_args)
        self.gradients = None

    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        extra_mid = self.backbone(inputs)

        extra_res = self.neck(extra_mid)
        return extra_res

    def forward(self, inputs):
        x = self.extract_feat(inputs)


        return x


    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.extract_feat(x)

class LSKmodule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.convl = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_m = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.convl(attn1)

        attn1 = self.conv0_s(attn1)
        attn2 = self.conv1_s(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0,:,:].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv_m(attn)
        return x * attn


class FPNLSKBBA(nn.Module):
    def __init__(self, backbone_name, backbone_args, fpn_name, fpn_args,heads, pretrained, down_ratio, final_kernel, head_conv):
        super(FPNLSKBBA, self).__init__()
        self.down_ratio = down_ratio
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))

        self.base_network = resfpn(backbone_name, backbone_args, fpn_name, fpn_args)
        self.lsk_part = LSKmodule(256)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),

                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),

                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1,
                                             padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)



    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)






        c2_combine = x
        c2_combine = self.lsk_part(c2_combine)












        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict



if __name__ == '__main__':
    model_cfg = {
        "name": "DBCNet",
        "model_block": {
            "backbone": {
                "type": "resnet18",
                "pretrained": True,
                "in_channels": 3
            },
            "neck": {
                "type": "FPN",
                "inner_channels": 256,
            },
            "head": {
                "type": "DBCHead",
                "num_classes": 9,
                "out_channels": 2,
                "k": 50,
            },
        },
        "module_loss": {
            "type": "DBModuleLoss",
            "args": {
                "num_classes": 9
            },

        }
    }

    arch_model = eval(model_cfg['name'])(model_cfg['model_block'])
    print(arch_model)

    device = torch.device('cuda:0')
    x = torch.zeros(2, 3, 608, 608).to(device)
    arch_model=arch_model.to(device)
    import time

    y = arch_model(x)
    print("y.shape: ", len(y))


