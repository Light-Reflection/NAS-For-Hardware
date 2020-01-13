
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from  flops_counter import get_model_complexity_info as get_flops
# TODO: FIX the import issue
import torch
import time
import numpy as np

TEST_TIMES = 5

def get_flops_params(model, input_shape, stat):
    flops, params = get_flops(model, input_shape, print_per_layer_stat=stat)
    return flops, params

def set_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True

def warm_up(op, inputs, warms_times=1000):
    "warmup hardware to reduce error"
    with torch.no_grad():
        for _ in range(warms_times):
            op(inputs)

def get_op_info(op, input_shape, batch_size, platform, seed=0):
    set_torch(seed=0)
    flops, params = get_flops_params(op, input_shape, True)
    print('flops:{}, params:{}'.format(flops, params))
    inputs = list(input_shape)
    inputs.insert(0, batch_size)
    inputs = torch.randn(inputs)
    if platform == 'GPU':
        print("="*20+" GPU Testing "+'='*20)
        op.cuda()
        inputs = inputs.cuda()
    if platform == 'CPU':
        print("="*20+" CPU Testing "+'='*20)

    op.eval()

    warm_up(op, inputs)
    all_t = []
    for _ in range(TEST_TIMES):
        all_t.append(get_latency(op, inputs)/batch_size)
    t_mean, t_var = cal_mean_var(all_t)
    return t_mean, t_var

def cal_mean_var(alist):
    # remove the min and max return the remaining eles in array
    sorted_list = sorted(alist)
    cal_arrary = np.array(sorted_list[1:-1])
    mean = np.mean(cal_arrary)
    var = np.var(cal_arrary)
    return mean, var

def get_latency(op, inputs, nums_data=1000):
    # return the cost time of processing one bacth data
    with torch.no_grad():
        torch.cuda.synchronize()
        tic = time.time()
        for _ in range(nums_data):
            op(inputs)
        torch.cuda.synchronize()
        toc = time.time()
    return (toc-tic)/nums_data

if __name__ == '__main__':
    from operations import OPS
    def get_model_parameters_number(model):
        params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params_num
    ops = OPS['MB6_3x3'](3, 32, 1, False)
    print(get_model_parameters_number(ops))
    print(get_op_info(ops, input_shape=(3,28,28), batch_size=256, platform='CPU'))


"""
INFO. Input=28, bs=256
      OP                  |  Params(Bytes) |   Macs      |CPU Running Time | GPU Running Time  
MB6_3x3(3, 32, 1, False)  |   792          |  674.24 K   | 0.00016         |                  

"""

