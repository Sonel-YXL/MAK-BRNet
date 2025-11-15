from torchvision.models import resnet50


import torch
import torchvision
from thop import profile
import torchvision.models as models

print('==> Building model..')

model = models.resnet50()

dummy_input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, (dummy_input,),verbose=False)
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / (1024.0*1024), params /  (1024.0*1024)))