import torch
from mbv2_supernet import MobileNetV2
import dmmo as dm
model = MobileNetV2(10)
inputs = torch.randn(1,3,32,32)
print(model(inputs))
