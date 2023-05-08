import torch
import numpy as np
from torch import nn
#import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity

class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        out = self.linear(input)
        threshold = out.sum(axis=1).mean().item()
        hi_idx = (mask > threshold).nonzero(as_tuple=True)
        return out, hi_idx

model = MyModule(100, 10).cuda()
input = torch.rand(128, 100).cuda()
mask = torch.rand((100, 100, 100), dtype=torch.float).cuda()

# Warm-up
model(input, mask)

with profile(with_stack=True, profile_memory=True, use_cuda=True) as prof:
    with record_function("model_inference"):
        out, idx = model(input, mask)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
#print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
