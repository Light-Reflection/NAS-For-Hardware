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
            epoch = 350,
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

    def predict(self, x, batch_size=32, **kwargs):
        """Predict the output for a given testing data.
        # Arguments
            x: tf.data.Dataset or numpy.ndarray. Testing data.
            batch_size: Int. Defaults to 32.
            **kwargs: Any arguments supported by keras.Model.predict.
        """
        best_model = self.tuner.get_best_models(1)[0]
        best_trial = self.tuner.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters

        self.tuner.load_trial(best_trial)
        x = utils.prepare_preprocess(x, x)
        x = self.hypermodel.preprocess(best_hp, x)
        x = x.batch(batch_size)
        y = best_model.predict(x, **kwargs)
        y = self._postprocess(y)
        if isinstance(y, list) and len(y) == 1:
            y = y[0]
        return y


    def prepare_data(self, x, y, validation_data, validation_split):
        # Initialize HyperGraph model
        x = nest.flatten(x)
        y = nest.flatten(y)
        # TODO: check x, y types to be numpy.ndarray or tf.data.Dataset.
        # TODO: y.reshape(-1, 1) if needed.
        y = self._label_encoding(y)
        # Split the data with validation_split
        if (all([isinstance(temp_x, np.ndarray) for temp_x in x]) and
                all([isinstance(temp_y, np.ndarray) for temp_y in y]) and
                validation_data is None and
                validation_split):
            (x, y), (x_val, y_val) = utils.split_train_to_valid(
                x, y,
                validation_split)
            validation_data = x_val, y_val
        # TODO: Handle other types of input, zip dataset, tensor, dict.
        # Prepare the dataset
        dataset = x if isinstance(x, tf.data.Dataset) \
            else utils.prepare_preprocess(x, y)
        if not isinstance(validation_data, tf.data.Dataset):
            x_val, y_val = validation_data
            validation_data = utils.prepare_preprocess(x_val, y_val)
        return dataset, validation_dat

    def _label_encoding(self, y):
        self._label_encoders = []
        new_y = []
        for temp_y, output_node in zip(y, self.outputs):
            hyper_head = output_node
            if isinstance(hyper_head, node.Node):
                hyper_head = output_node.in_blocks[0]
            if (isinstance(hyper_head, head.ClassificationHead) and
                    utils.is_label(temp_y)):
                label_encoder = utils.OneHotEncoder()
                label_encoder.fit_with_labels(y)
                new_y.append(label_encoder.encode(y))
                self._label_encoders.append(label_encoder)
            else:
                new_y.append(temp_y)
                self._label_encoders.append(None)
        return new_y

    def _postprocess(self, y):
        y = nest.flatten(y)
        if not self._label_encoders:
            return y
        new_y = []
        for temp_y, label_encoder in zip(y, self._label_encoders):
            if label_encoder:
                new_y.append(label_encoder.decode(temp_y))
            else:
                new_y.append(temp_y)
        return new_y

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




class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
