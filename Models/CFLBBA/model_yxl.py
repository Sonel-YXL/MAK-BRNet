


import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Resize
from convnext import ConvNeXt, convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
from convnext_isotropic import ConvNeXtIsotropic, convnext_isotropic_small, convnext_isotropic_base, \
    convnext_isotropic_large
from convnext import convnext_yxl
from Models.FPNBBA.module_neck import FPN

class resfpn(nn.Module):
    def __init__(self):
        """
        OOD:这里进行重新构建了,根据一个配置文件进行构建模型.
        :param model_config:
        """
        super().__init__()

        self.backbone = convnext_tiny(pretrained=True)
        self.neck = FPN(in_channels=[96, 192, 384, 768],inner_channels=256) 
        self.gradients = None  

    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        extra_mid = self.backbone(inputs)
        
        extra_res = self.neck(extra_mid)
        return extra_res  

    def forward(self, inputs):
        x = self.extract_feat(inputs)  
        
        
        return x

if __name__ == "__main__":
    
    input_tensor = torch.rand(1, 3, 608, 608)
    print(f"输入张量大小: {input_tensor.shape}")

    
    model = resfpn()
    model.eval()  
    
    with torch.no_grad():
        output = model(input_tensor)
    print(f"模型输出大小: ", len(output))

    exit()
    
    print("\n--- ConvNeXt 模型演示 ---")
    convnext_models = {
        "resfpn": resfpn,
        
        
        
        
        
        
    }

    for name, model_fn in convnext_models.items():
        print(f"\n加载模型: {name}")
        
        model = model_fn(pretrained=True)  
        model.eval()  

        
        
        
        
        

        
        with torch.no_grad():
            output = model(input_tensor)

        print(f"模型输出大小: ",len(output))