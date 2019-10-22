import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

# global params
BIAS = False 
IMAGE_SIZE = 32
OPS = {'MB6_3x3': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=6, max_kernel_size=3, stride=stride, bias=False, affine=affine, act_type='swish'),
    'MB6_5x5': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=6, max_kernel_size=5, stride=stride, bias=False, affine=affine, act_type='swish'),
    'MB3_3x3': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=3, max_kernel_size=3, stride=stride, bias=False, affine=affine, act_type='swish'),
    'MB3_5x5': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=3, max_kernel_size=5, stride=stride, bias=False, affine=affine, act_type='swish'),    
    'Conv3x3_BN_Act': lambda max_in_channels, max_out_channels, stride, affine, act_type: \
    ConvBNActi(max_in_channels=max_in_channels, max_out_channels=max_out_channels, max_kernel_size=3, stride=stride, bias=False, affine=affine, act_type=act_type),
    'MB1_3x3': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=1, max_kernel_size=3, stride=stride, bias=False, affine=affine, act_type='swish'),
    'Conv1x1_BN_Act': lambda max_in_channels, max_out_channels, stride, affine, act_type: \
    ConvBNActi(max_in_channels=max_in_channels, max_out_channels=max_out_channels, max_kernel_size=1, stride=stride, bias=False, affine=affine, act_type=act_type),
    'MB6_3x3_se0.25': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=6, max_kernel_size=3, stride=stride, bias=False, affine=affine, act_type='swish', se=0.25),
    'MB6_5x5_se0.25': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=6, max_kernel_size=5, stride=stride, bias=False, affine=affine, act_type='swish', se=0.25),
    'MB1_3x3_se0.25': lambda max_in_channels, max_out_channels, stride, affine: \
    MBConv(max_in_channels=max_in_channels, max_out_channels=max_out_channels, expand_ratio=1, max_kernel_size=3, stride=stride, bias=False, affine=affine, act_type='swish', se=0.25),
    'Sep_3x3_N': lambda max_in_channels, max_out_channels, stride, affine:\
    SepConv_NoMidReLU(max_in_channels, max_out_channels, 3, stride, bias=False, affine=affine, act_type='relu'),
    'Sep_3x3': lambda max_in_channels, max_out_channels, stride, affine:\
    SepConv(max_in_channels, max_out_channels, 3, stride, bias=False, affine=affine, act_type='relu'),
}

class swish(nn.Module):
    # swish activation 
    def __init__(self):
        super(swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

def activation(func='relu'):
    """activate function"""
    acti = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU()],
        ['prelu', nn.PReLU()],
        ['relu', nn.ReLU()],
        ['none', Identity()],
        ['relu6', nn.ReLU6()],
        ['swish', swish()]])

    return acti[func]

def drop_connect(input_, p, training):
    """Build for drop connetc"""
    if not training: return input_
    bs = input_.shape[0] 
    keep_prob = 1 - p
    random_tensor = keep_prob
    print(input_.dtype, input_.device)
    random_tensor += torch.rand([bs, 1, 1, 1], dtype=input_.dtype, devices=input_.device)
    binary_tensor = torch.floor(random_tensor)
    output = input_ / keep_prob * binary_tensor
    return output

from functools import partial
def get_same_padding_conv2d(image_size=None):
    return partial(ManualConv2dPad, image_size=image_size)

# Custom Convolution 
class ManualConv2dPad(nn.Conv2d):
    """docstring for ManualConv2d"""
    def __init__(self, max_in_channels, max_out_channels, max_kernel_size, image_size=None, stride=1, dilation=1, groups=1, bias=True):
        super(ManualConv2dPad, self).__init__(max_in_channels, max_out_channels, max_kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias)

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh , sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

        self._has_bias = bias
        self._max_ksize = max_kernel_size
        # self._padding = padding
        self._max_inc = max_in_channels
        self._max_outc = max_out_channels
        self._max_ksize = max_kernel_size

    def forward(self, x, in_channels=None, out_channels=None, kernel_size=None, groups=None):
        x = self.static_padding(x)
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
        # in group conv :in_channels = 1
        return F.conv2d(x, self.weight[:out_channels, :in_channels, lbound:rbound, lbound:rbound],
            self.bias[:out_channels] if self._has_bias else None, self.stride, self.padding, self.dilation, int(groups) if groups else self.groups)

class ManualBN2d(nn.BatchNorm2d):
    # effi eps=0.001 / momentum=0.01

    def __init__(self, max_num_features, eps=0.001, momentum=0.01, affine=True, tracking_running_stats=True):
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



ManualConv2d = get_same_padding_conv2d(IMAGE_SIZE)

class ConvBNActi(nn.Module):
    """docstring for ConvBNActi"""
    def __init__(self, max_in_channels, max_out_channels, max_kernel_size, stride, bias, affine, act_type):
        super(ConvBNActi, self).__init__()
        self._conv = ManualConv2d(max_in_channels, max_out_channels, max_kernel_size, stride=stride, bias=bias)
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
    def __init__(self, max_in_channels, max_out_channels, max_kernel_size, stride, bias, affine, act_type):
        super(DwConvBNActi, self).__init__()
        assert max_in_channels==max_out_channels
        self._dwconv = ManualConv2d(max_in_channels, max_in_channels, max_kernel_size, stride=stride, groups=max_in_channels, bias=bias)
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

class  SepConv(nn.Module):
    """docstring for  SepConv"""
    def __init__(self, max_in_channels, max_out_channels, max_kernel_size, stride, bias, affine, act_type):
        super( SepConv, self).__init__()
        self._max_inc = max_in_channels
        self._max_outc = max_out_channels
        self._bias = bias
        self._affine = affine
        self._sconv = DwConvBNActi(max_in_channels, max_in_channels, max_kernel_size, stride, bias, affine, act_type) # previous not have this activation
        self._pconv = PwConvBNActi(max_in_channels, max_out_channels, bias, affine, act_type)
    def forward(self, x, in_channels=None, out_channels=None, kernel_size=None):
        x = self._sconv(x, in_channels, in_channels, kernel_size)
        x = self._pconv(x, in_channels, out_channels, kernel_size)
        return x
        
class  SepConv_NoMidReLU(nn.Module):
    """docstring for  SepConv"""
    def __init__(self, max_in_channels, max_out_channels, max_kernel_size, stride, bias, affine, act_type):
        super(SepConv_NoMidReLU, self).__init__()
        self._max_inc = max_in_channels
        self._max_outc = max_out_channels
        self._bias = bias
        self._affine = affine
        self._sconv = DwConvBNActi(max_in_channels, max_in_channels, max_kernel_size, stride, bias, affine, act_type='none') # previous not have this activation
        self._pconv = PwConvBNActi(max_in_channels, max_out_channels, bias, affine, act_type)
    def forward(self, x, in_channels=None, out_channels=None, kernel_size=None):
        x = self._sconv(x, in_channels, in_channels, kernel_size)
        x = self._pconv(x, in_channels, out_channels, kernel_size)
        return x
        

class MBConv(nn.Module):
    """
    Enable to choose : expand ratio / kernel_size / se / skip / drop connect
    """
    def __init__(self, max_in_channels, max_out_channels, expand_ratio, max_kernel_size, stride, bias, affine, act_type, se=False, skip=True, drop=False):
        # TODO: Support the drop  connect when training supernet
        super(MBConv, self).__init__()
        self._stride = stride 
        self._expand_ratio =  expand_ratio
        self._bias = bias 
        self._affine = affine
        # self._padding = padding 
        self._act_type = act_type
        self._se = se
        self._skip = skip
        self._drop_prob = drop
        self._max_inc = max_in_channels
        self._max_outc = max_out_channels

        max_hidden_dim = round(expand_ratio * max_in_channels)
        if expand_ratio != 1:
            self._econv = PwConvBNActi(max_in_channels, max_hidden_dim, bias, affine, act_type) # expand channels
        self._sconv = DwConvBNActi(max_hidden_dim, max_hidden_dim, max_kernel_size, stride, bias, affine, act_type) # sepconv
        if se:
            num_squeezed_channels = max(1, int(max_in_channels*se))
            self._se_rconv = ManualConv2d(max_hidden_dim, num_squeezed_channels, 1)
            self._se_acti = activation(act_type)
            self._se_econv = ManualConv2d(num_squeezed_channels, max_hidden_dim, 1)
            self._num_squeezed_channels = num_squeezed_channels

        self._rconv = PwConvBNActi(max_hidden_dim, max_out_channels, bias, affine, act_type='none') # reduce channels
        self._hidden_dim = max_hidden_dim

    def forward(self, x, in_channels=None, out_channels=None, kernel_size=None):
        inputs = x
        hidden_dim = round(in_channels * self._expand_ratio) if in_channels else self._hidden_dim
        inc = in_channels if in_channels else self._max_inc
        outc = out_channels if out_channels else self._max_outc
        if self._expand_ratio != 1:
            x = self._econv(x, in_channels, hidden_dim)
        x = self._sconv(x, hidden_dim, hidden_dim, kernel_size)
        if self._se:
            num_squeezed_channels = max(1, int(in_channels * self._se)) if in_channels else self._num_squeezed_channels
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x = self._se_rconv(x, hidden_dim, num_squeezed_channels)
            x = self._se_acti(x)
            x = self._se_econv(x, num_squeezed_channels, hidden_dim)
            x = torch.sigmoid(x_squeezed) * x

        x = self._rconv(x, hidden_dim, out_channels)
        if self._skip and self._stride == 1 and inc == outc:
            # if self._drop_prob:
            #     x = drop_connect(x , self._drop_prob, training=self.training) # Error: Drop Connect 
            x = x + inputs

        return x


if __name__ == '__main__':
    # x = torch.ones((1,3,6,6))
    # conv2d = nn.Conv2d(5, 5, 3, groups=5)
    # # print(conv2d.weight.shape)
    # mconv2d = ManualConv2d(6, 6, 3, groups=6)
    # print(mconv2d.weight.shape)
    # y = mconv2d(x, 3, 3)
    # print(mconv2d.weight)
    op = MBConv(3, 3, 3, 3, 1, 1, False, False, 'relu6', 0.25, True, 0.1)
    print(op(torch.rand(3,3,6,6)))
