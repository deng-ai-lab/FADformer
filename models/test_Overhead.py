import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from FADformer import FADformer, FADformer_mini

net1 = FADformer().to('cuda:0')

input = torch.randn(1, 3, 256, 256).to('cuda:0')
flops1, params1 = profile(net1, inputs=(input,))

# test method 01
print('FADformer_overhead')
print('FLOPs = ' + str(flops1/1000**3) + 'G')
print('Params = ' + str(params1/1000**2) + 'M')

# test method 02
flops = FlopCountAnalysis(net1, input)
print("FLOPs: ", flops.total()/1000**3)
print(parameter_count_table(net1))

