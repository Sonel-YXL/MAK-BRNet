import torch
import math
import einops
import torch.nn as nn

class RetinaHead(nn.Module):
    def __init__(self):
        super().__init__()
        channels=[64,128,256,512]
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(3):
            self.cls_convs.append(
                nn.Conv2d(
                    channels[i],
                    channels[i+1],
                    3,
                    stride=1,
                    padding=1))
            self.reg_convs.append(
                nn.Conv2d(
                    channels[i],
                    channels[i+1],
                    3,
                    stride=1,
                    padding=1))

    def forward(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        return cls_feat,reg_feat

if __name__ == '__main__':
    w, h = 608, 608
    model1 = RetinaHead()



    x = [torch.randn((2, 64, 304, 304)),
         torch.randn((2, 128, 152, 152)),
         torch.randn((2, 256, 76, 76)),
         torch.randn((2, 512, 38, 38))]
    x = torch.rand(1, 64, 32, 32)
    y = model1(x)
    print(y.keys())