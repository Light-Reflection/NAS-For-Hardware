import os
import sys
sys.path.append(os.getcwd())
import torch 
import torch.nn as nn
from .operations import OPS, ManualLinear
import numpy as np

from .automodel import AutoModel
from .utils import produce_channel_encoding, produce_resolution_encoding, produce_op_encoding, produce_ksize_encoding, print_dict
# able to changes layers
# able to changes search direction/ set diff encoding

# PRIMITIVES = ['MB6_3x3_se0.25', 'MB6_5x5_se0.25', 'MB3_3x3_se0.25', 'MB3_5x5_se0.25']
PRIMITIVES = ['Sep_3x3', 'Sep_5x5']

class  MixOPs(nn.Module):
    """docstring for  MixOPs"""
    # TODO: support two operations 
    def __init__(self, max_in_channels, max_out_channels, stride, affine, to_dispatch=False, init_op_index=None):
        super(MixOPs, self).__init__()
        self._to_dispatch = to_dispatch
        self._mix_ops = nn.ModuleList()
        if to_dispatch:
            # only support single op in the mix ops for now 
            # print('designed mix op')
            self._mix_ops.append(OPS[PRIMITIVES[init_op_index]](max_in_channels, max_out_channels, stride, affine))
        else:
            for primitive in PRIMITIVES:
                self._mix_ops.append(OPS[primitive](max_in_channels, max_out_channels, stride, affine))

    def forward(self, x, op_index=0, in_channels=None, out_channels=None, kernel_size=None):
        # print(x.shape)
        return self._mix_ops[op_index](x, in_channels, out_channels, kernel_size) 

class Cell(nn.Module):
    """docstring for Cell"""
    def __init__(self, max_in_channels, max_out_channels, stride, init_length, affine, to_dispatch=False, op_cfg=None):
        super(Cell, self).__init__()
        self._cell_nets = nn.ModuleList()
        # init: not to sample and choose the 0 op
        self._init_inc_list = [None]*init_length
        self._init_outc_list = [None]*init_length
        self._init_ksize_list = [None]*init_length
        self._init_index_list = op_cfg if op_cfg else [None]*init_length 
        if isinstance(max_in_channels, int) and isinstance(max_out_channels, int):
            max_in_channels = [max_in_channels]
            max_in_channels.extend([max_out_channels]*(init_length-1))
            max_out_channels = [max_out_channels]*init_length
        else:
            assert len(max_in_channels) == len(max_out_channels) == init_length, 'The length of in_channel list must be matched'
        # print(self._init_index_list)
        for i,init_op_index in enumerate(self._init_index_list):
            self._cell_nets.append(MixOPs(max_in_channels[i], max_out_channels[i], stride if i==0 else 1, affine, to_dispatch, init_op_index))

    def forward(self, x, index_list=None, inc_list=None, outc_list=None, ksize_list=None, sample_length=None): # ksize_list has problem
        index_list = index_list if index_list else self._init_index_list
        inc_list = inc_list if inc_list else self._init_inc_list
        outc_list = outc_list if outc_list else self._init_outc_list
        ksize_list = ksize_list if ksize_list else self._init_ksize_list  # depend on the sepcific opeartion

        for i,cell_net in enumerate(self._cell_nets[:sample_length]):
            x = cell_net(x, index_list[i], inc_list[i], outc_list[i], ksize_list[i])
        return x

class mobilenet_supernet(nn.Module,AutoModel):
    """docstring for supernet"""
    def __init__(   self, 
                    to_dispatch=False, 
                    resolution_encoding=None,
                     channel_encoding=None, 
                     op_encoding=None, 
                     ksize_encoding=None,
                     name = None,
                     constrain = None,
                     **kwargs): 

        super(mobilenet_supernet, self).__init__()
        # supernet cfg
        if to_dispatch:
            kwargs = kwargs['diction']
        self.constrain = constrain
        self.config = kwargs
        self.name = name
        self.build_supernet_cfg(kwargs,to_dispatch)
        self.combination = dict()
        print(resolution_encoding)
        self.manual_set_init_cfg(resolution_encoding, channel_encoding, op_encoding, ksize_encoding) # if all None it will do nothing

        self._to_dispatch = to_dispatch
        self._stem = nn.ModuleList()
        self._cells = nn.ModuleList()
        # self._stern = nn.ModuleList()
        self.subnets = []
        stem_inc = 3
        print(to_dispatch)
        stem_outc = self._channel_init_cfg[0] if to_dispatch else self._stem_cfg[0][0]
        self._stem.append(OPS['Conv3x3_BN_Act'](max_in_channels=stem_inc, max_out_channels=stem_outc, stride=self._stem_cfg[0][1], affine=self._affine, act_type='relu'))
        stem_inc = stem_outc
        stem_outc = self._channel_init_cfg[1] if to_dispatch else self._stem_cfg[1][0]
        self._stem.append(OPS['Sep_3x3'](max_in_channels=stem_inc, max_out_channels=stem_outc, stride=self._stem_cfg[1][1], affine=self._affine))

        init_super_cells = self._cells_layers - (len(self._cells_cfg) - 1)  # not include cell 0
        init_cell_inc = stem_outc
        has_layers = len(self._stem_cfg)
        for i in range(self._num_cells):
            if to_dispatch:
                init_cell_length = self._resolution_init_cfg[i]
                init_cell_inc = self._channel_init_cfg[(has_layers-1):(has_layers+init_cell_length-1)] 
                init_cell_outc = self._channel_init_cfg[has_layers:(has_layers+init_cell_length)] 
                init_op_cfg = self._op_init_cfg[sum(resolution_encoding[:i]):sum(resolution_encoding[:(i+1)])]
                has_layers += init_cell_length
            else:
                init_cell_length = init_super_cells if self._search_direction[0] else self._resolution_init_cfg[i]
                init_cell_outc = self._cells_cfg[i][0] # if not search set it in channel cfg
                # print(self._op_init_cfg)
                init_op_cfg = None 

            init_cell_stride = self._cells_cfg[i][1]
            self._cells.append(Cell(max_in_channels=init_cell_inc, max_out_channels=init_cell_outc, \
                stride=init_cell_stride, init_length=init_cell_length ,\
                affine=self._affine, to_dispatch=to_dispatch, op_cfg=init_op_cfg))

            init_cell_inc = init_cell_outc

        stern_outc = self._channel_init_cfg[-1] if to_dispatch else self._cells_cfg[-1][0]


        self._linear = ManualLinear(max_in_channels=stern_outc, max_out_channels=self._num_classes)

    def run(self, x, mode, resolution_encoding, channel_encoding, op_encoding, ksize_encoding):
        """     
        produce run cfg ; if all encoding is None it will execute the noraml forward
        print self._resolution_cfg, self._channel_cfg, self._op_cfg
        
        """
        self.set_run_cfg(mode, resolution_encoding, channel_encoding, op_encoding, ksize_encoding)

        if self.constrain and mode == 'train':
            # para = self.count_submodel_parameter(self._channel_cfg,self._op_cfg)
            while self.count_submodel_parameter(self._channel_cfg,self._op_cfg) > self.constrain:
                #print('~'*20,'The parameter amount of this model is:',para,'~'*20)
                #print('~'*20,'over sized model random again','~'*20)
                self.set_run_cfg(mode, resolution_encoding, channel_encoding, op_encoding, ksize_encoding)
                # para = self.count_submodel_parameter(self._channel_cfg,self._op_cfg)
        index_layer = 0
        for i,stem_layer in enumerate(self._stem):
            x = stem_layer(x, self._channel_cfg[index_layer], self._channel_cfg[index_layer+1], self._ksize_cfg[index_layer])
            index_layer += 1
            
        index_op_layer = 0
        for i,cell in enumerate(self._cells):
            sample_length = self._resolution_cfg[i]
            x = cell(x, self._op_cfg[index_op_layer:(index_op_layer+sample_length)], self._channel_cfg[index_layer:(index_layer+sample_length)], \
                self._channel_cfg[(index_layer+1):(index_layer+1+sample_length)], \
                self._ksize_cfg[index_layer:(index_layer+sample_length)], sample_length)

            index_layer += sample_length
            index_op_layer += sample_length

        # for stern_layer in self._stern:
        #     x = stern_layer(x, self._channel_cfg[index_layer], self._channel_cfg[index_layer+1], self._ksize_cfg[index_layer])
        #     index_layer += 1

        x = self._linear(x.mean(3).mean(2), self._channel_cfg[index_layer], self._num_classes) # include avg pool

        return x

    def forward(self, x):
        return self.run(x, 'train', None, None, None, None)

    def predict(self, x, resolution_encoding=None, channel_encoding=None, op_encoding=None, ksize_encoding=None):
        return self.run(x, 'search', resolution_encoding, channel_encoding, op_encoding, ksize_encoding)

    def set_run_cfg(self, mode, resolution_encoding=None, channel_encoding=None, op_encoding=None, ksize_encoding=None):
        if self._to_dispatch:
            self._resolution_cfg = self._resolution_init_cfg # for layers index
        else: 
            if mode == 'search':
                # TODO: set input encoding should be same as search direction
                assert (resolution_encoding or channel_encoding or op_encoding or ksize_encoding) != None
                self.manual_set_forwad_cfg(resolution_encoding, channel_encoding, op_encoding, ksize_encoding)

            elif mode == 'train':
                self.random_set_forward_cfg(self._search_direction)
                # TODO: Auto set the search direction 
            else:
                raise ValueError

            # self._channel_cfg.insert(0, 3) # Bug: will always insert 3 if not search channel 

    def manual_set_init_cfg(self, resolution_encoding=None, channel_encoding=None, op_encoding=None, ksize_encoding=None):
        # for dispatch module
        if resolution_encoding: # not search resolution and maual define the resolution cfg
            print('='*20,'Maunal set the resolution cfg in supernet','='*20)
            self._resolution_init_cfg = resolution_encoding
            print(self._resolution_init_cfg)
            print("Produce channel according to the resolution-encoding:", end='')
            self._channel_init_cfg = self.convert_resolution_encoding_to_basic_channel_cfg(resolution_encoding)
            #print(self._channel_init_cfg)
        if channel_encoding:
            assert channel_encoding is not None
            print('='*20,'Maunal set the channel_encoding cfg in supernet','='*20)
            self._channel_init_cfg = self.convert_channel_encoding_to_channel_cfg(channel_encoding, self._division,\
                self.convert_resolution_encoding_to_basic_channel_cfg(resolution_encoding))
        if op_encoding:
            assert op_encoding is not None
            #print('='*20,'Maunal set the op cfg in supernet','='*20)
            self._op_init_cfg = op_encoding
            #print(self._op_init_cfg)
        if ksize_encoding:
            #print('='*20,'Maunal set the ksize cfg in supernet','='*20) 
            self._ksize_init_cfg = ksize_encoding
            #print(self._ksize_init_cfg)


    def random_set_forward_cfg(self, search_direction):
        """        
        resolution & op: only serach in the part of cells
        channel: the whole net(expect the linear bcz its out channel depends on the num classes)
        assign the foward cfg of three direction 
        """
        if search_direction[0]:
            self._resolution_cfg = self.produce_resolution_encoding() 
        else:
            self._resolution_cfg = self._resolution_init_cfg # if not search, it will adopt the init cfg
        if search_direction[1]: 
            channel_encoding = self.produce_channel_encoding()
            basic_channel_cfg = self.convert_resolution_encoding_to_basic_channel_cfg(self._resolution_cfg)
            self._channel_cfg = self.convert_channel_encoding_to_channel_cfg(channel_encoding, self._division, basic_channel_cfg)
            self._channel_cfg.insert(0, 3)
        else: 
            pass # using the config channel #TODO: support the manual setting
        if search_direction[2]: # serach in op
            self._op_cfg = self.produce_op_encoding() 
        else:
            self._op_cfg = self._op_init_cfg
        if search_direction[3]:
            raise NotImplementedError
            # self._ksize_cfg = self.produce_ksize_encoding() 

    def manual_set_forwad_cfg(self, resolution_encoding=None, channel_encoding=None, op_encoding=None, ksize_encoding=None):
        # activate this method to get the subnet acc.
        if resolution_encoding:
            self._resolution_cfg = resolution_encoding
        if channel_encoding:
            basic_channel_cfg = self.convert_resolution_encoding_to_basic_channel_cfg(resolution_encoding)
            self._channel_cfg = self.convert_channel_encoding_to_channel_cfg(channel_encoding, self._division, basic_channel_cfg)
            self._channel_cfg.insert(0, 3)
        if op_encoding:
            self._op_cfg = op_encoding
        if ksize_encoding:
            self._ksize_cfg = ksize_encoding
    
    def dispatch(self,resolution_encoding=[2,2,3,3,4], \
         channel_encoding=None, op_encoding=None, ksize_encoding=None):
        to_dispatch = True
        # assert to_dispatch == True
        kwargs = self.config
        model = mobilenet_supernet(to_dispatch, resolution_encoding, channel_encoding, op_encoding, ksize_encoding,diction = kwargs)
        self.subnets.append(model.cuda())
        return model
        
    def dispatch_eval(self,resolution_encoding=[2,2,3,3,4], \
         channel_encoding=None, op_encoding=None, ksize_encoding=None):
        to_dispatch = True
        # assert to_dispatch == True
        kwargs = self.config
        return supernet(to_dispatch, resolution_encoding, channel_encoding, op_encoding, ksize_encoding,diction = kwargs).cuda()

    def build_supernet_cfg(self, kwargs,to_dispatch):
        # TODO: implement other info if needed
        if to_dispatch:
            pass
            # print("="*20, "Dispatching subnet from SuperNet", "="*20)
        else:
            print("="*20, "Start building supernet", "="*20)
        kwargs.setdefault('layers', 19)
        kwargs.setdefault('affine', True)
        kwargs.setdefault('num_of_ops', 4)
        kwargs.setdefault('division', 1)
        kwargs.setdefault('search_direction', [True, True, True, False]) # resolution/channel/op/ksize
        kwargs.setdefault('channels', [(32,2), (16,1), [(24,2),(40,2),(80,2),(112,1),(192,2)], (320,1), (1280,1)]) #[stem, [cells], stern] , (outc, stride)
        kwargs.setdefault('num_of_classes',1000)
        for i,channel_setting in enumerate(kwargs['channels']):
            if isinstance(channel_setting, list):
                self._cells_cfg = channel_setting
                self._cells_index = i

        self._stem_cfg = kwargs['channels'][:self._cells_index]
        # self._stern_cfg = kwargs['channels'][(self._cells_index+1):]
        self._num_classes = kwargs['num_of_classes']
        self._channels = kwargs['channels']
        self._layers = kwargs['layers']
        self._affine = kwargs['affine']
        self._num_ops = kwargs['num_of_ops']
        self._division = kwargs['division']
        self._search_direction = kwargs['search_direction']
        self._num_cells = len(self._cells_cfg) 
        self._cells_layers = self._layers - len(self._stem_cfg)  - 1 # 1 for classifier

        # defalut 
        self._resolution_cfg = [None]*self._num_cells 
        self._op_cfg = [0]*self._cells_layers
        self._ksize_cfg = [None]*self._layers
        self._channel_cfg = [None]*self._layers

        print_dict(kwargs)
        print("="*24, "End", "="*24)

    
    def produce_resolution_encoding(self):
        return produce_resolution_encoding(self._cells_layers, self._num_cells)
    def produce_channel_encoding(self):
        return produce_channel_encoding(self._layers-1, self._division)  # -1: the last channel depends on the num_classes
    def produce_op_encoding(self):
        return produce_op_encoding(self._cells_layers, self._num_ops)
    def produce_kszie_encoding(self):
        return produce_ksize_encoding(self._layers)

    def sample_function(self):
        def helper():
            length=self._num_cells+(self._layers-1)+self._cells_layers
            # [0:self._num_cells,self._num_cells:self._num_cells+self._layers-1,self._num_cells+self._layers-1:self._num_cells+self._layers=1+self._cells_layers]
            net_encoding_list = [-1]*length
            for i,direction in enumerate(self._search_direction):
                if direction is True:
                    if i == 0: 
                        resolution_encoding = self.produce_resolution_encoding()
                        net_encoding_list[:self._num_cells] = resolution_encoding
                    if i == 1:
                        channel_encoding = self.produce_channel_encoding()
                        net_encoding_list[self._num_cells:self._num_cells+self._layers-1]= channel_encoding
                    if i == 2: 
                        op_encoding = self.produce_op_encoding()
                        net_encoding_list[self._num_cells+self._layers-1:self._num_cells+self._layers-1+self._cells_layers]=op_encoding
                    if i == 3: 
                        raise NotImplementedError
            return resolution_encoding,channel_encoding,op_encoding,np.array(net_encoding_list)
        
        resolution_encoding,channel_encoding,op_encoding,net_encoding_list =helper()
        parameter = get_model_parameters_number(self.dispatch_eval(resolution_encoding,channel_encoding, op_encoding))
        
        if self.constrain:
            while parameter > self.constrain:
                print('*'*24,'over sized parameters')
                resolution_encoding,channel_encoding,op_encoding,net_encoding_list =helper()
                parameter = get_model_parameters_number(self.dispatch_eval(resolution_encoding,channel_encoding, op_encoding))
        return net_encoding_list

    def convert_encoding_list_to_dict(self, net_encoding_list):
        # TODO: support channel_encoding
        net_encoding_dict = {'resolution_encoding':net_encoding_list[:self._num_cells].tolist() if self._search_direction[0] else self._resolution_init_cfg, 
        'channel_encoding':net_encoding_list[self._num_cells:self._num_cells+self._layers-1].tolist()  if self._search_direction[1] else None, \
        'op_encoding':net_encoding_list[self._num_cells+self._layers-1:self._num_cells+self._layers-1+self._cells_layers].tolist() if self._search_direction[2] else self._op_init_cfg, \
        'ksize_encoding':None}

        # print(net_encoding_dict)
        return net_encoding_dict


    def convert_channel_encoding_to_channel_cfg(self, channel_encoding, division, basic_channel_cfg):
        assert max(channel_encoding) <= division and len(channel_encoding) == len(basic_channel_cfg), 'cfg conflicts, ' + str(max(channel_encoding)) + 'vs'+ str(division) + '\n' +  str(len(channel_encoding)) + 'vs' + str(len(basic_channel_cfg))
        channel_cfg = []
        for i,encoding in enumerate(channel_encoding):
            channel_cfg.append(encoding * basic_channel_cfg[i]//division)
        return channel_cfg

    def convert_resolution_encoding_to_basic_channel_cfg(self, resolution_encoding):
        # basic_channel_cfg = [3]
        # print(resolution_encoding)
        assert sum(resolution_encoding) == self._cells_layers, \
        'your resultion_encoding is {}, But the sum(resolution_encoding) should be {}'.format(resolution_encoding, self._cells_layers)
        basic_channel_cfg = [stem_cfg[0] for stem_cfg in self._stem_cfg]
        for i,encoding in enumerate(resolution_encoding):
            for _ in range(encoding):
                basic_channel_cfg.append(self._cells_cfg[i][0])
        # basic_channel_cfg.extend([stern_cfg[0] for  stern_cfg in self._stern_cfg])
        return basic_channel_cfg
    
    def count_submodel_parameter(self,channellist,operationlist):
        """
        : return: The parameter amount of submodel
        """
        ops0 = OPS['MB6_3x3_se0.25']
        ops1 = OPS['MB6_5x5_se0.25']
        count = 0
        for i in range(len(operationlist)):
            if operationlist[i] == 0:
                if (0,channellist[i+1],channellist[i+2]) not in self.combination:
                    kernel = ops0(channellist[i+1],channellist[i+2],1,True)
                    self.combination[(0,channellist[i+1],channellist[i+2])] = get_model_parameters_number(kernel)         
                count += self.combination[(0,channellist[i+1],channellist[i+2])]
            elif operationlist[i] == 1:
                if (1,channellist[i+1],channellist[i+2]) not in self.combination:
                    kernel = ops1(channellist[i+1],channellist[i+2],1,True)
                    self.combination[(1,channellist[i+1],channellist[i+2])] = get_model_parameters_number(kernel)         
                count += self.combination[(1,channellist[i+1],channellist[i+2])]
        count+=get_model_parameters_number(self._stem)
        # count+=get_model_parameters_number(self._stern)
        count+=get_model_parameters_number(self._linear)
        return count

def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num











        
