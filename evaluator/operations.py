import torch 
import torch.nn as nn
import torch.nn.functional as F

OPS = {'MB6_3x3': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=6, max_kernel_size=3, stride=stride, padding=1, bias=False, affine=affine, act_type='relu6'),
    'MB6_5x5': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=6, max_kernel_size=5, stride=stride, padding=2, bias=False, affine=affine, act_type='relu6'),
    'MB3_3x3': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=3, max_kernel_size=3, stride=stride, padding=1, bias=False, affine=affine, act_type='relu6'),
    'MB3_5x5': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=3, max_kernel_size=5, stride=stride, padding=2, bias=False, affine=affine, act_type='relu6'),    
    'Conv3x3_BN_ReLU6': lambda max_in_channels, max_out_channels, stride, affine: \
    ConvBNActi(max_in_channels=max_in_channels, max_out_channels=max_out_channels, max_kernel_size=3, stride=stride, padding=1, bias=False, affine=affine, act_type='relu6'),
    'MB1_3x3': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=1, max_kernel_size=3, stride=stride, padding=1, bias=False, affine=affine, act_type='relu6'),
    'Conv1x1_BN_ReLU6': lambda max_in_channels, max_out_channels, stride, affine: \
    ConvBNActi(max_in_channels=max_in_channels, max_out_channels=max_out_channels, max_kernel_size=1, stride=stride, padding=0, bias=False, affine=affine, act_type='relu6'),
    'MB6_3x3_se0.25': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=6, max_kernel_size=3, stride=stride, padding=1, bias=False, affine=affine, act_type='relu6', se=0.25),
    'MB6_5x5_se0.25': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=6, max_kernel_size=5, stride=stride, padding=2, bias=False, affine=affine, act_type='relu6', se=0.25),
    'MB1_3x3_se0.25': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=1, max_kernel_size=3, stride=stride, padding=1, bias=False, affine=affine, act_type='relu6', se=0.25),

}

def activation(func='relu'):
    """activate function"""
    acti = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU()],
        ['prelu', nn.PReLU()],
        ['relu', nn.ReLU()],
        ['none', Identity()],
        ['relu6', nn.ReLU6()]])

    return acti[func]

def drop_connect(input_, p, training):
    """Build for drop connetc"""
    if not training: return input_
    bs = input_.shape[0] 
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([bs, 1, 1, 1], dtype=input_.dtype, devices=input_.device)
    binary_tensor = torch.floor(random_tensor)
    output = input_ / keep_prob * binary_tensor
    return output

# Custom Convolution 
class ManualConv2d(nn.Conv2d):
    """docstring for ManualConv2d"""
    def __init__(self, max_in_channels, max_out_channels, max_kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ManualConv2d, self).__init__(max_in_channels, max_out_channels, max_kernel_size, stride, padding, dilation, groups, bias)
        self._has_bias = bias
        self._max_ksize = max_kernel_size
        self._padding = padding
        self._max_inc = max_in_channels
        self._max_outc = max_out_channels
        self._max_ksize = max_kernel_size

    def forward(self, x, in_channels=None, out_channels=None, kernel_size=None, groups=None):
        kernel_size = kernel_size if kernel_size  else self._max_ksize
        if in_channels:
            assert in_channels <= self._max_inc, 'in_channels:{}, max_in_channels:{}'.format(in_channels, self._max_inc)
        if out_channels:
            assert out_channels <= self._max_outc, 'out_channels:{}, max_out_channels:{}'.format(out_channels, self._max_outc)
        if kernel_size:
            assert kernel_size <=self._max_ksize, 'kernel_size:{}, max_kernel_size:{}'.format(kernel_size, self._max_ksize)
        # only support when kernel size is odd
        rbound = (kernel_size+self._max_ksize)//2
        lbound = (self._max_ksize-kernel_size)//2

        return F.conv2d(x, self.weight[:out_channels, :in_channels, lbound:rbound, lbound:rbound],
            self.bias[:out_channels] if self._has_bias else None, self.stride, self.padding, self.dilation, int(groups) if groups else self.groups)

class ManualBN2d(nn.BatchNorm2d):
    def __init__(self, max_num_features, eps=1e-5, momentum=0.1, affine=True, tracking_running_stats=True):
        super(ManualBN2d, self).__init__(max_num_features, eps, momentum, affine, tracking_running_stats)
        # self._max_nf = max_num_features
    def forward(self, x, num_features=None):
        # original define in BatchNorm2d
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        
        # Adjust num features and avoid the condiction that self.bias & self.weight is None
        return F.batch_norm(x, self.running_mean[:num_features], self.running_var[:num_features], \
            self.weight[:num_features] if self.affine else self.weight, self.bias[:num_features] if self.affine else self.bias, \
            self.training or not self.track_running_stats, exponential_average_factor, self.eps)

class ManualLinear(nn.Linear):
    """docstring for ManualLinear"""
    def __init__(self, max_in_channels, max_out_channels, bias=True):
        super(ManualLinear, self).__init__(max_in_channels, max_out_channels, bias)
        self._has_bias = bias
    
    def forward(self, x, in_channels=None, out_channels=None):
        return F.linear(x, self.weight[:out_channels, :in_channels], self.bias[:out_channels] if self._has_bias else self.bias)

class ConvBNActi(nn.Module):
    """docstring for ConvBNActi"""
    def __init__(self, max_in_channels, max_out_channels, max_kernel_size, stride, padding, bias, affine, act_type):
        super(ConvBNActi, self).__init__()
        self._conv = ManualConv2d(max_in_channels, max_out_channels, max_kernel_size, stride=stride, padding=padding, bias=bias)
        self._bn = ManualBN2d(max_out_channels, affine=affine)
        self._acti = activation(act_type)
        self._max_outc = max_out_channels
        self._affine = affine
    def forward(self, x, in_channels=None, out_channels=None, kernel_size=None):
        x = self._conv(x, in_channels, out_channels, kernel_size)
        x = self._bn(x, out_channels)
        x = self._acti(x)
        return x

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class DwConvBNActi(nn.Module):
    """docstring for ConvBNReLU"""
    def __init__(self, max_in_channels, max_out_channels, max_kernel_size, stride, padding, bias, affine, act_type):
        super(DwConvBNActi, self).__init__()
        assert max_in_channels==max_out_channels
        self._dwconv = ManualConv2d(max_in_channels, max_in_channels, max_kernel_size, stride=stride, padding=padding, groups=max_in_channels, bias=bias)
        self._bn = ManualBN2d(max_in_channels, affine=affine)
        self._acti = activation(act_type)
        self._affine = affine
        self._max_outc = max_out_channels


    def forward(self, x, in_channels=None, out_channels=None, kernel_size=None):
        x = self._dwconv(x, in_channels, out_channels, kernel_size, in_channels)
        x = self._bn(x, out_channels)
        x = self._acti(x)
        return x

class PwConvBNActi(nn.Module):
    """docstring for PwConvBNActi"""
    def __init__(self, max_in_channels, max_out_channels, bias, affine, act_type):
        super(PwConvBNActi, self).__init__()
        self._pwconv = ManualConv2d(max_in_channels, max_out_channels, 1, bias=bias)
        self._bn = ManualBN2d(max_out_channels, affine=affine)
        self._acti = activation(act_type)
        self._affine = affine
        self._max_outc = max_out_channels

    def forward(self, x, in_channels=None, out_channels=None, kernel_size=None):
        x = self._pwconv(x, in_channels, out_channels, kernel_size)
        x = self._bn(x, out_channels)
        x = self._acti(x)

        return x

class MBConv(nn.Module):
    """
    Enable to choose : expand ratio / kernel_size / se / skip / drop connect
    """
    def __init__(self, max_in_channels, max_out_channels, expand_ratio, max_kernel_size, stride, padding, bias, affine, act_type, se=False, skip=False, drop=False):
        # TODO: Support the drop and skip connect when training supernet
        super(MBConv, self).__init__()
        self._stride = stride 
        self._expand_ratio =  expand_ratio
        self._bias = bias 
        self._affine = affine
        self._padding = padding 
        self._act_type = act_type
        self._se = se

        max_hidden_dim = round(expand_ratio * max_in_channels)
        if expand_ratio != 1:
            self._econv = PwConvBNActi(max_in_channels, max_hidden_dim, bias, affine, act_type) # expand channels
        self._sconv = DwConvBNActi(max_hidden_dim, max_hidden_dim, max_kernel_size, stride, padding, bias, affine, act_type) # sepconv
        if se:
            num_squeezed_channels = max(1, int(max_in_channels*se))
            self._se_rconv = ManualConv2d(max_hidden_dim, num_squeezed_channels, 1)
            self._se_acti = activation('relu')
            self._se_econv = ManualConv2d(num_squeezed_channels, max_hidden_dim, 1)
            self._num_squeezed_channels = num_squeezed_channels

        self._rconv = PwConvBNActi(max_hidden_dim, max_out_channels, bias, affine, act_type='none') # reduce channels
        self._hidden_dim = max_hidden_dim

    def forward(self, x, in_channels=None, out_channels=None, kernel_size=None):
        hidden_dim = round(in_channels * self._expand_ratio) if in_channels else self._hidden_dim

        if self._expand_ratio != 1:
            x = self._econv(x, in_channels, hidden_dim)
        x = self._sconv(x, hidden_dim, hidden_dim, kernel_size)
        x = self._rconv(x, hidden_dim, out_channels)
        return x
