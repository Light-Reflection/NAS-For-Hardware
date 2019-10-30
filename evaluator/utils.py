import os
import sys
import numpy as np
import logging
import shutil
import random
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import collections
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn as nn
import time
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg, self.sum, self.cnt = 0, 0, 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt

def distribute_set_up(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12335'
    dist.init_process_group('nccl', rank = rank, world_size = world_size)
    torch.manual_seed(42+rank)

def distribute_cleanup():
    dist.destroy_process_group()

def save_model(model, path):
    try:
        torch.save(model.module.state_dict(), path)
    except AttributeError: 
        print('** '+ 'Current Training without DDP or DP '+' **'+' saving model.state_dict() in ', path)
        torch.save(model.state_dict(), path)

def cifar10_data_transform(cfg):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg['data_mean'], cfg['data_std']),
        ])

    if cfg['cutout']:
        train_transform.transforms.append(Cutout(cfg['cutout_length']))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg['data_mean'], cfg['data_std']),
        ])
    return train_transform, valid_transform



def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('Save path is exists')

def setup_logger(path, name):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger(name).addHandler(fh)

def cifar10_data_transform():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]),
        ])

    # if cfg['cutout']:
        # train_transform.transforms.append(Cutout(cfg['cutout_length']))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]),
        ])
    return train_transform, valid_transform

def set_cudnn(mode):
    cudnn.enabled = True
    if mode == 'deterministic':
        cudnn.deterministic = True
    elif mode == 'benchmark':
        cudnn.benchmark = True
    elif mode == 'disable':
        cudnn.enabled = False
    else:
        raise Exception('Error cudnn mode: %s' % mode)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def reproducibility(cudnn_mode, seed):
    set_seed(seed)
    set_cudnn(cudnn_mode)

def load_model(model, load_path, strict=True):
    load_dict = torch.load(load_path)
    assert isinstance(load_dict,(collections.OrderedDict))==True, 'Only support resume model form dict'
    model_dict = model.state_dict()
    # print('***', model_dict.keys())
    # print('---', load_dict.keys())
    unload_in_model = {k:v for k,v in model_dict.items() if k not in load_dict}
    load_in_model = {k:v for k,v in load_dict.items() if k in model_dict}
    # print(unload_in_model.keys())
    if strict:
      assert len(unload_in_model.keys()) == 0, unload_in_model.keys()
      model.load_state_dict(load_dict)
    else:
      model.logger.info('Unload state_dict: %s', str(unload_in_model.keys()))
      model.logger.info('Load state_dict: %s', str(load_in_model.keys()))
      model_dict.update(load_in_model)
      model.load_state_dict(model_dict)

    return model


class BN_Correction(object):
    """docstring for BN_Correction"""
    def __init__(self, model, train_queue, num_bacthes=4, logger=None):
        super(BN_Correction, self).__init__()
        self.model = model
        self.logger = logger
        num_bn = 0

        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.momentum = 1
                num_bn += 1
        if self.logger:
          self.logger.info('Number of BN momentumn reset: {}'.format(num_bn))
        raw_data = []
        count = 0
        for data,_ in train_queue:
            raw_data.append(data)
            count += 1
            if count >= num_bacthes:
                break
        self.data = torch.cat(raw_data, dim=0)
        # assert h == w == img_dim, (h, w, img_dim)

        self.data = self.data.cuda()

        # self.save is deleted in this script 

    def __call__(self):
        print("Rest BN")
        # net_idx = str(self.model.net_id)
        # if self.logger:
        #   self.logger.info("Reseting BN running stats in Net_idx: {}".format(net_idx))
        torch.cuda.synchronize()
        tic = time.time()
        self.model.train()
        with torch.no_grad():
          _ = self.model(self.data)
        torch.cuda.synchronize()
        toc = time.time()
        if self.logger:
          self.logger.info("Reset BN running stats cost time:{:.1f} s".format(toc-tic))

