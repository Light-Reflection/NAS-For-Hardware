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
import dmmo 
import torch.distributed as dist
import time
import datetime

# parser.add_argument("--local_rank", default=0, type=int)

def load_data(data_root, batch_size, num_workers):
    train_transform, valid_transform = cifar10_data_transform()
    train_data =  torchvision.datasets.CIFAR10(data_root, train=True, transform=train_transform)
    valid_data = torchvision.datasets.CIFAR10(data_root, train=False, transform=valid_transform)
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return train_queue, valid_queue
def set_logger_writer(save_path):
    create_exp_dir(save_path, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
    setup_logger(save_path, 'trainer')
    logger = logging.getLogger('trainer')
    writer =  SummaryWriter(save_path, 'summary')
    logger.info('Create logger and wirter in the path: %s ' % save_path)
    return logger, writer

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, model, train_queue, valid_queue, epoch, optimizer, \
        scheduler, criterion, logger, writer, rank=0, world_size=1, stats='S'):
        super(Trainer, self).__init__()
        self._model = model
        self._optimizer = optimizer(model.parameters(), \
        lr= 0.1, momentum=0.9, weight_decay=0.0005) # set optim **kw later
        self._scheduler = scheduler(self._optimizer, [60,120,200,250]) # set scheduler **kw later
        self._criterion = criterion
        self._logger = logger 
        self._writer = writer
        self._epoch = epoch
        self._rank = rank
        self._stats = stats
        self._wsize = world_size
        self._using_ddp = world_size > 1
        self._train_queue = train_queue
        self._valid_queue = valid_queue
        self._loss = AvgrageMeter()
        self._prec1 = AvgrageMeter()
        self._prec5 = AvgrageMeter()
        self._bn_setter_init = True
        self._bn_setter_num_batch = 1


    def reset_stats(self):
        self._loss.reset()
        self._prec1.reset()
        self._prec5.reset()

    def update_stats(self, loss, prec1, prec5, bs):
        self._loss.update(loss, bs)
        self._prec1.update(prec1, bs)
        self._prec5.update(prec5, bs)

    def write_stats(self, epoch, loss, prec1, prec5, phase):
        self._writer.add_scalar('{}_loss'.format(phase), loss, epoch)
        self._writer.add_scalar('{}_prec1'.format(phase), prec1, epoch)
        self._writer.add_scalar('{}_prec5'.format(phase), prec5, epoch)

    def run(self, start_epoch=0, save_path='./logs/'):
        check_path(save_path) # create path if not exists 
        best_prec = 0
        save_frequency = 50
        for current_epoch in range(start_epoch, self._epoch):
            self._current_epoch = current_epoch
            tic = time.time()
            self._scheduler.step()
            self.train_epoch()
            torch.cuda.synchronize()
            if self._rank == 0:
                self._logger.info('Epoch %03d | Time: %s | train_acc %f | learning_rate %s', \
                    current_epoch, str(datetime.timedelta(seconds=round(time.time()-tic))), self._prec1.avg, str(self._scheduler.get_lr()))
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
        if self._rank == 0:
            self.write_stats(self._current_epoch, reduced_loss, prec1, prec5, self._stats+'_train')

    def validate(self):
        if self._stats == 'S':
            self._logger.info("Val SuperNet")
            self.inference(mode='trainS')
        elif  'D' in self._stats:
            self._logger.info("Val DispatchNet")
            self.inference(mode='trainD')
        else:
            raise ValueError('Check Trainer stats')
        
    def predict(self, resolution_encoding=None, channel_encoding=None, op_encoding=None, ksize_encoding=None):
        if self._bn_setter_init:
            # only initalize once
            self._bn_setter =  BN_Correction(self._model, self._train_queue, self._bn_setter_num_batch)  
            self._bn_setter_init = False
        self._logger.info("Into Predict Module......")
        self._logger.info("======== Rest bn =========")
        self._bn_setter()
        self.inference('search', resolution_encoding, channel_encoding, op_encoding, ksize_encoding)
        self._logger.info('Net encoding:')
        self._logger.info('resolution_encoding:%s ; channel_encoding:%s; op_encoding:%s; ksize_encoding:%s',\
            resolution_encoding, channel_encoding, op_encoding, ksize_encoding)
        self._logger.info('net Prec1 avg: %s', self._prec1.avg)
        return self._prec1.avg # Get accuarcy

    def inference(self, mode, resolution_encoding=None, channel_encoding=None, op_encoding=None, ksize_encoding=None):
        if mode == 'trainS':
            self._model.train() # Train SuperNet 
        elif mode == 'search' or mode == 'trainD':
            self._model.eval() # Search SUperNet and Train Dispatch
        else:
            raise NotImplementedError
        self.reset_stats()
        tic = time.time()
        for step, (inputs, targets) in enumerate(self._valid_queue):
            with torch.no_grad():
                inputs, targets = inputs.cuda(), targets.cuda()
                if 'train' in mode:
                    logits = self._model(inputs) # Train Net
                elif mode == 'search':
                    logits = self._model.predict(inputs, \
                        resolution_encoding, channel_encoding, op_encoding, ksize_encoding)
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
            if self._rank == 0 and 'train' in mode:
                self.write_stats(self._current_epoch, self._loss.avg, self._prec1.avg, self._prec5.avg, self._stats+'_valid')

def train_smodel(rank, world_size):
    if rank == 0:
        print("into main .......")
    reproducibility(cudnn_mode='deterministic', seed=0)
    train_queue, valid_queue = load_data(data_root='/media/dm/d/data/CIFAR', batch_size=128, num_workers=4)
    epoch = 300
    optimizer = torch.optim.SGD
    scheduler = torch.optim.lr_scheduler.MultiStepLR
    criterion = torch.nn.CrossEntropyLoss()
    logger, writer = set_logger_writer('./test')
    stats = 'D'

    # output info 
    assert cudnn.benchmark != cudnn.deterministic or cudnn.enabled == False
    if rank == 0:
        logger.info('|| torch.backends.cudnn.enabled = %s' % cudnn.enabled)
        logger.info('|| torch.backends.cudnn.benchmark = %s'% cudnn.benchmark)
        logger.info('|| torch.backends.cudnn.deterministic = %s' % cudnn.deterministic)
        logger.info('|| torch.cuda.initial_seed = %d' % torch.cuda.initial_seed())

    ren = [2, 5, 4, 1, 2]
    # cen = [1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2]
    cen = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    oen = [1, 3, 3, 1, 3, 1, 2, 2, 0, 3, 1, 3, 2, 3] 
    model = dmmo.supernet(channels = [(32,2), (16,1), [(24,2),(40,2),(80,1),(112,1),(192,2)], (320,1), (1280,1)],
                        layers = 19,num_of_classes = 10,name = 'testing', division=2,
                        search_direction = [True,True,True,False],
                        constrain = 1000000000,
                       ).cuda()
    model = model.dispatch(ren, cen, oen)
    print(model)
    # model = torchvision.models.MobileNetV2(10).cuda()
    # if world_size > 1:
    #     distribute_set_up(rank, world_size)
    #     n = torch.cuda.device_count()//world_size # default run all GPUs
    #     device_ids = list(range(rank * n, (rank + 1)*n)) # split GPUs 
    #     # import your model in this command
    #     # model = model.to(device_ids[0]) or set_device(rank)
    #     model = DDP(model, device_ids = device_ids)
    # trainer = Trainer(model, train_queue, valid_queue, epoch, optimizer, scheduler, criterion, logger, writer, rank, world_size, stats=stats) # init trainer
    # trainer.run() # run trainer


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
#     main(0,1)
#     must need __main__ func
   
#     test(0,1,model = model)


