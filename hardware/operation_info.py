
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
from  flops_counter import get_model_complexity_info as get_flops
# TODO: FIX the import issue
import torch
import time
import numpy as np
from operations import OPS
import os
import logging
TEST_TIMES = 5
PRIM =  ['MB3_5x5', 'MB6_5x5', 'MB3_3x3', 'MB6_3x3']
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
    inputs = list(input_shape)
    inputs.insert(0, batch_size)
    inputs = torch.randn(inputs)
    if platform == 'GPU':
        op.cuda()
        inputs = inputs.cuda()

    op.eval()

    warm_up(op, inputs)
    all_t = []
    for _ in range(TEST_TIMES):
        all_t.append(get_latency(op, inputs)/batch_size)
    t_mean, t_var = cal_mean_var(all_t)
    return flops, params, t_mean, t_var

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

def get_ops(in_channel, out_channel):
    ops = {} 
    # in_channels = [3, 32, 64, 128, 256, 512]
    # out_channels =  [32, 64, 128, 256, 512, 1024]
    in_channels = [3,32,16,24,24,32,32,32,64,64,64,96,96,96,160,160,160,320,1280]
    for op in PRIM:
        op_1 = OPS[op](in_channel, out_channel, 1, False)
        ops[op+'({}, {}, 1, False)'.format(in_channel, out_channel)] = op_1
        op_2 = OPS[op](in_channel, out_channel, 2, False)
        ops[op+'({}, {}, 2, False)'.format(in_channel, out_channel)] = op_2
    return ops

def set_logger(path, type, level=logging.DEBUG):
    """ Set training logger

    Args:
        path: the log save path
        type: the name to save log, the log will save as $type.log
        level: the logging level of FileHandler and StreamHandler

    Returns: logger.

    """
    logger = logging.getLogger(type)
    logger.setLevel(level)  # set level
    fh = logging.FileHandler(os.path.join(path, type +'.log'))
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
    

if __name__ == '__main__':
    import logging
    logger = set_logger(path='./', type='ops_cpu')
    # len

    # in_channels = [32,16,24,24,32,32,32,64,64,64,96,96,96,160,160,160,320,1280]

    bs = 128
    platform = 'CPU'
    logger.info("batch size: "+ str(bs))
    logger.info("platform: "+str(platform))
    input_shape = [(3,32, 32),(32,32,32),(16,32,32),(24,16,16),(32,8,8),(64,4,4),(96,4,4),(160,2,2),(320,2,2)]  # channel first
    for i,shape in enumerate(input_shape):
        logger.info("input shape: " + str(shape))
        in_channel = shape[0]
        if i == len(input_shape)-1:
            out_channel = 1280
        else:
            
            out_channel = input_shape[i+1][0]
        ops = get_ops(in_channel, out_channel)
        for name, op in ops.items():
            flops, params, tm, tv = get_op_info(op, input_shape=shape, batch_size=bs, platform=platform)
            logger.info("Op: {}, params: {}, flops: {}, mean of time: {}, var of time: {}".format(name, params, flops, tm, tv))
    # bs = 1
    # platform = 'CPU'
    # conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    # flops, params, tm, tv = get_op_info(conv1, input_shape=(3,32,32), batch_size=bs, platform=platform)
    # print(flops, params)

   # from operations import OPS
   # def get_model_parameters_number(model):
   #     params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
   #     return params_num
   # ops = OPS['MB6_3x3'](3, 32, 1, False)
   # print(get_model_parameters_number(ops))
   # print(get_op_info(ops, input_shape=(3,28,28), batch_size=256, platform='CPU'))



"""
GPU INFO. Input=28, bs=1256
      OP                  |  Params(Bytes) |   Macs      |CPU Running Time | GPU Running Time  
MB6_3x3(3, 32, 1, False)  |   792          |  674.24 K   | 0.00016         |                  

"""


"""
INFO. Input=28, bs=256
      OP                  |  Params(Bytes) |   Macs      |CPU Running Time | GPU Running Time  
MB6_3x3(3, 32, 1, False)  |   792          |  674.24 K   | 0.00016         |                  

"""

