import os
import glob
import torch.distributed as dist
import torch
import logging
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
import torchvision
from dmmo.evaluator.utils import *
# from dmmo.SuperNet import supernet
import torch.distributed as dist
import time
import datetime

# parser.add_argument("--local_rank", default=0, type=int)
class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, model, train_queue, valid_queue, epoch, optimizer, scheduler, criterion, logger, writer, rank=0, world_size=1):
        super(Trainer, self).__init__()
        self._model = model
        self._optimizer = optimizer(model.parameters(), lr= 0.04) # set optim **kw later
        self._scheduler = scheduler(self._optimizer, [100,250]) # set scheduler **kw later
        self._criterion = criterion
        self._logger = logger 
        self._writer = writer
        self._epoch = epoch
        self._rank = rank
        self._wsize = world_size
        # self._current_epoch = 0
        # self._epoch_size = len(self._train_data)//batch_size
        self._using_ddp = world_size > 1
        self._train_queue = train_queue
        self._valid_queue = valid_queue
        self._loss = AvgrageMeter()
        self._prec1 = AvgrageMeter()
        self._prec5 = AvgrageMeter()

    def reset_stats(self):
        self._loss.reset()
        self._prec1.reset()
        self._prec5.reset()

    def update_stats(self, loss, prec1, prec5, bs):
        self._loss.update(loss, bs)
        self._prec1.update(prec1, bs)
        self._prec5.update(prec5, bs)

    def write_stats(self, step, loss, prec1, prec5, phase):
        self._writer.add_scalar('{}_loss'.format(phase), loss, step)
        self._writer.add_scalar('{}_prec1'.format(phase), prec1, step)
        self._writer.add_scalar('{}_prec5'.format(phase), prec5, step)

    def run(self, start_epoch=0, save_path='.'):
        best_prec = 0
        save_frequency = 50
        for current_epoch in range(start_epoch, self._epoch):
            self._current_epoch = current_epoch
            tic = time.time()
            self._scheduler.step()
            self.train_epoch()
            torch.cuda.synchronize()
            if self._rank == 0:
                self._logger.info('Epoch %03d | Time: %s | train_acc %f | learning_rate %s', current_epoch, 
                    str(datetime.timedelta(seconds=round(time.time()-tic))), self._prec1.avg, str(self._scheduler.get_lr()))
            if current_epoch % save_frequency == 0 and self._rank == 0 :
                save_model(self._model, os.path.join(save_path, 'Epoch{}.pt'.format(current_epoch)))

            tic = time.time()
            self.validate()
            torch.cuda.synchronize()

            # get best precision
            if best_prec < self._prec1.avg:
                best_prec = self._prec1.avg
                save_model(self._model, os.path.join(save_path, 'best_weights.pt'))
            if self._rank == 0:
                self._logger.info('Epoch %03d |  Time: %s | valid_acc %f | best_acc %f', current_epoch, 
                    str(datetime.timedelta(seconds=round(time.time()-tic))), self._prec1.avg, best_prec)
        if self._rank == 0:
            save_model(self._model, os.path.join(save_path, 'final_weights.pt'))

    def train_epoch(self, report_freq = 200):
        # not support auxiliary loss
        self._model.train()
        self.reset_stats()
        for step, (inputs, targets) in enumerate(self._train_queue):
            inputs, targets = inputs.cuda(), targets.cuda()
            self._optimizer.zero_grad()

            logits = self._model(inputs)

            loss = self._criterion(logits, targets)
            loss.backward()
            self._optimizer.step()

            prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
            # iters = (self._current_epoch-1) * epoch_size + step 
            if self._using_ddp:
                # Using DDP
                reduced_loss = reduce_tensor(loss.data, self._wsize)
                prec1 = reduce_tensor(prec1, self._wsize)
                prec5 = reduce_tensor(prec5, self._wsize)
            else:
                reduced_loss = loss.data

            self.update_stats(reduced_loss.item(), prec1.item(), prec5.item(), inputs.size(0))
            if step % report_freq == 0 and self._rank == 0: # set rank for logger
                self._logger.info('train: %03d | loss: %e | prec1:%f | prec5: %f', step, self._loss.avg, self._prec1.avg, self._prec5.avg)

    def validate(self):
        self.inference(mode='train')

    def predict(self, resolution_encoding=None, channel_encoding=None, op_encoding=None, ksize_encoding=None):
        print("Into Predict Module......")
        self.inference('search', resolution_encoding, channel_encoding, op_encoding, ksize_encoding)
        print(resolution_encoding, channel_encoding, op_encoding, ksize_encoding)
        print(self._prec1.avg)
        return self._prec1.avg # Get accuarcy

    def inference(self, mode='train', resolution_encoding=None, channel_encoding=None, op_encoding=None, ksize_encoding=None):
        self._model.eval()
        self.reset_stats()
        tic = time.time()
        for step, (inputs, targets) in enumerate(self._valid_queue):
            with torch.no_grad():
                inputs, targets = inputs.cuda(), targets.cuda()
                if mode == 'train':
                    logits = self._model(inputs)
                elif mode == 'search':
                    logits = self._model.predict(inputs, resolution_encoding, channel_encoding, op_encoding, ksize_encoding)
            loss =  self._criterion(logits, targets)
            prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
            if self._using_ddp:
                # Using DDP
                reduced_loss = reduce_tensor(loss.data, self._wsize)
                prec1 = reduce_tensor(prec1, self._wsize)
                prec5 = reduce_tensor(prec5, self._wsize)
            else:
                reduced_loss = loss.data

            self.update_stats(reduced_loss.item(), prec1.item(), prec5.item(), inputs.size(0))
            if self._rank == 0 and mode == 'train':
                self.write_stats(self._current_epoch, self._loss.avg, self._prec1.avg, self._prec5.avg, 'valid')

# using trainer

# debug test 




def main(rank, world_size):
    if rank == 0:
        print("into main .......")
    reproducibility(cudnn_mode='deterministic', seed=0)
    train_queue, valid_queue = load_data(data_root='~/data/CIFAR', batch_size=128, num_workers=4)
    epoch = 10
    optimizer = torch.optim.SGD
    scheduler = torch.optim.lr_scheduler.MultiStepLR
    criterion = torch.nn.CrossEntropyLoss()
    logger, writer = set_logger_writer('./test')

    # output info 
    assert cudnn.benchmark != cudnn.deterministic or cudnn.enabled == False
    if rank == 0:
        logger.info('|| torch.backends.cudnn.enabled = %s' % cudnn.enabled)
        logger.info('|| torch.backends.cudnn.benchmark = %s'% cudnn.benchmark)
        logger.info('|| torch.backends.cudnn.deterministic = %s' % cudnn.deterministic)
        logger.info('|| torch.cuda.initial_seed = %d' % torch.cuda.initial_seed())

# <<<<<<< Updated upstream
#     # model = MobileNet().cuda()
#     from SuperNet import SuperNet
#     from model import MobileNet
#     net_cfg = {}
#     net_cfg['layer'] = 10
#     torch.cuda.set_device(rank)
#     model = MobileNet().cuda()
# =======
    model = MobileNet().cuda()
    #from SuperNet import SuperNet
    #net_cfg = {}
    #net_cfg['layer'] = 10
    #model = SuperNet(net_cfg).cuda()
# >>>>>>> Stashed changes
    if world_size > 1:
        distribute_set_up(rank, world_size)
        n = torch.cuda.device_count()//world_size # default run all GPUs
        device_ids = list(range(rank * n, (rank + 1)*n)) # split GPUs 
        # import your model in this command
        # model = model.to(device_ids[0]) or set_device(rank)
        model = DDP(model, device_ids = device_ids)
    trainer = Trainer(model, train_queue, valid_queue, epoch, optimizer, scheduler, criterion, logger, writer, rank, world_size) # init trainer
    trainer.run() # run trainer


def run_main(fn, world_size):
    mp.spawn(fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def test(rank, world_size,model):
    print("into main .......")
    reproducibility(cudnn_mode='deterministic', seed=0)
    train_queue, valid_queue = load_data(data_root='/home/dm/data/CIFAR', batch_size=32, num_workers=4)
    epoch = 10
    optimizer = torch.optim.SGD
    scheduler = torch.optim.lr_scheduler.MultiStepLR
    criterion = torch.nn.CrossEntropyLoss()
    logger, writer = set_logger_writer('./test')

    # output info 
    assert cudnn.benchmark != cudnn.deterministic or cudnn.enabled == False
    logger.info('|| torch.backends.cudnn.enabled = %s' % cudnn.enabled)
    logger.info('|| torch.backends.cudnn.benchmark = %s'% cudnn.benchmark)
    logger.info('|| torch.backends.cudnn.deterministic = %s' % cudnn.deterministic)
    logger.info('|| torch.cuda.initial_seed = %d' % torch.cuda.initial_seed())

    if world_size > 1:
        distribute_set_up(rank, world_size)
        n = torch.cuda.device_count()//world_size # default run all GPUs
        device_ids = list(range(rank * n, (rank + 1)*n))
        # import your model in this command
        # model = model.to(device_ids[0])
        ddp_model = DDP(model, device_ids = device_ids)
    trainer = Trainer(model, train_queue, valid_queue, epoch, optimizer, scheduler, criterion, logger, writer, rank, world_size) # init trainer
    trainer.run() # run trainer


# if __name__ == '__main__':
#     model = supernet(num_of_ops = 10,layers = 20,num_of_classes = 2000)
    # main(0,1)
    # must need __main__ func
   
    # test(0,1,model = model)


