import torch 
import torch.nn as nn
from dmmo.runner.train_model import *
from dmmo.evaluator.Search import Searcher
import time

class AutoModel(object):
    def __init__(self,
                 name='auto_model',
                 maxtime=60* 60 * 3,
                 seed=None,
                 ):
        self.name = name
        self.tuner = None
        self.maxtime = maxtime
        self.seed = seed



    def fit(self,
            epoch = 300,
            optimizer = torch.optim.SGD,
            scheduler = torch.optim.lr_scheduler.MultiStepLR,
            criterion = torch.nn.CrossEntropyLoss(),
            data_root = None,
            save_path = './report',
            rank = 0,
            world_size = 1,
            model = None,
            train_data_loader = None, 
            valid_data_loader = None,
            max_samples = 100,
            target_acc = 100,
            top_k = 5,
            batch_size=128,
            **kwargs,
           ):
        """
           A print function which is the same as the original func

        :param int epoch: The epochs you want to iterate
        :param str optimizer: The optimizer you want to choose
        :param str criterion: The loss function you want to choose
        :param int world_size: The numbers of GPU you want to use
        :param int data_root: The root of data, there are two ways to input data, you can write your own data loader

        :return: no return a nn.Module modle: tok_k submodels of supernet
        """
        reproducibility(cudnn_mode='deterministic', seed=0)
        save_path = os.path.join('./{}'.format(self.name), time.strftime("%Y%m%d-%H%M%S"), 'logs')
        logger, writer = set_logger_writer(save_path)
        #logger.info(model)

        # output info 
        assert cudnn.benchmark != cudnn.deterministic or cudnn.enabled == False
        logger.info('|| torch.backends.cudnn.enabled = %s' % cudnn.enabled)
        logger.info('|| torch.backends.cudnn.benchmark = %s'% cudnn.benchmark)
        logger.info('|| torch.backends.cudnn.deterministic = %s' % cudnn.deterministic)
        logger.info('|| torch.cuda.initial_seed = %d' % torch.cuda.initial_seed())
        train_queue = train_data_loader
        valid_queue = valid_data_loader
    
        if world_size > 1:
            if rank == 0:
                logger.info(' ## You are using DDP ##')
            distribute_set_up(rank, world_size)
            n = torch.cuda.device_count()//world_size # default run all GPUs
            device_ids = list(range(rank * n, (rank + 1)*n))    
            # import your model in this command
            # model = model.to(device_ids[0])
            model = DDP(model, device_ids = device_ids)
        if data_root:
            train_queue, valid_queue = load_data(data_root, batch_size=batch_size, num_workers=4)
            self._trainer = Trainer(model, train_queue, valid_queue, epoch, optimizer, scheduler, criterion, logger, writer, rank, world_size) # init trainer
            self._trainer.run()
        else:
            train_queue = train_data_loader
            valid_queue = valid_data_loader

            self._trainer = Trainer(model, train_queue, valid_queue, epoch, optimizer, scheduler, criterion, logger, writer, rank, world_size) # init trainer
            self._trainer.run(save_path=os.path.join(save_path, 'SuperNet-checkpoints'))

        self.search(model,top_k,target_acc,max_samples)
        ite = 0
        for i in self.topk_encoding:
            diction = self.convert_encoding_list_to_dict(i[0])
            resolution_encoding = diction['resolution_encoding']
            channel_encoding = diction['channel_encoding']
            op_encoding = diction['op_encoding']
            model.dispatch(resolution_encoding,channel_encoding, op_encoding)
        for j in range(len(self.subnets)):
            train_queue, valid_queue = load_data(data_root, batch_size=128, num_workers=4)
            self._trainer = Trainer(self.subnets[j], train_queue, valid_queue, epoch, optimizer, scheduler, criterion, logger, writer, rank, world_size, stats='D'+str(j))
            # Train DispatchNet 
            self.submodelname = 'submodel-number-{}-checkpoints'.format(j)
            self._trainer.run(save_path = os.path.join(save_path, self.submodelname))
            print('*'*20,'Subnet number',ite,'training finished!','*'*20)
            ite +=1

            

    def get_net_acc(self, net_dict):
        return self._trainer.predict(net_dict['resolution_encoding'], net_dict['channel_encoding'], net_dict['op_encoding'], net_dict['ksize_encoding'])

    def search(self, model, top_k, target_acc,max_samples):
        self._searcher = Searcher(method='AS', sample_func=model.sample_function, get_net_acc= self.get_net_acc, convert_func=model.convert_encoding_list_to_dict)
        self._searcher.run(target_acc,max_samples)
        save = self._searcher.get_top_accuracy(top_k)
        self.topk_encoding = save

    
def set_logger_writer(save_path):
    create_exp_dir(save_path, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
    setup_logger(save_path, 'trainer')
    logger = logging.getLogger('trainer')
    writer =  SummaryWriter(save_path, 'summary')
    logger.info('Create logger and wirter in the path: %s ' % save_path)
    return logger, writer


def load_data(data_root, batch_size, num_workers):
    train_transform, valid_transform = cifar10_data_transform()
    train_data =  torchvision.datasets.CIFAR10(data_root, train=True, transform=train_transform)
    valid_data = torchvision.datasets.CIFAR10(data_root, train=False, transform=valid_transform)
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return train_queue, valid_queue



