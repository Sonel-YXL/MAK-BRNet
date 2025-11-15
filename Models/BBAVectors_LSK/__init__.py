import torch

from Models.BBAVectors_LSK.ctrbox_net import CTRBOX_LSK
from Models.BBAVectors_LSK.decoder import DecDecoder
from Models.BBAVectors_LSK.loss import LossAll_nan

if __name__ == '__main__':

    num_classes = {'dota': 15, 'hrsc': 1}
    config_net = {
        'heads': {'hm': num_classes["hrsc"],
                  'wh': 10,
                  'reg': 2,
                  'cls_theta': 1
                  },
        'pretrained': True,
        'down_ratio': 4,
        'final_kernel': 1,
        'head_conv': 256
    }


    model = CTRBOX(**config_net)

    decoder_model=DecDecoder(K=500,conf_thresh=0.18,num_classes=15)
    x=torch.randn((2,3,608,608))
    print(x.size())
    y = model(x)
    print(y.keys())