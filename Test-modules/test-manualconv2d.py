import torch 
import torch.nn as nn
import torch.nn.functional as F

class ManualConv2d(nn.Conv2d):
    """docstring for ManualConv2d"""
    def __init__(self, max_in_channels, max_out_channels, max_kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(max_in_channels, max_out_channels, max_kernel_size, stride, padding, dilation, groups, bias)
        # self._inc = in_channels
        # self._outc = out_channels
        self._has_bias = bias
        self._max_inc = max_in_channels
        self._max_outc = max_out_channels
        self._max_ksize = max_kernel_size

        self.padding = padding

    def forward(self, x, in_channels=None, out_channels=None, kernel_size=None):
        in_channels = self._max_inc if in_channels is None else in_channels
        out_channels = self._max_outc if out_channels is None else out_channels
        kernel_size = self._max_ksize if kernel_size is None else kernel_size

        # process the weight kernel  
        # only support when kernel size is odd
        rbound = (kernel_size+self._max_ksize)//2
        lbound = (self._max_ksize-kernel_size)//2

        return F.conv2d(x, self.weight[:out_channels, :in_channels, lbound:rbound, lbound:rbound],
            self.bias[:out_channels] if self._has_bias else None, self.stride, self.padding, self.dilation, self.groups)
       

class net(nn.Module):
    """docstring for net"""
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = ManualConv2d(max_in_channels=3, max_out_channels=8, max_kernel_size=5, padding=1)
        self.conv2 = ManualConv2d(max_in_channels=8, max_out_channels=12, max_kernel_size=5, padding=1)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x, out_channels=4, kernel_size=3)
        # print(x.shape)
        x = self.conv2(x, in_channels=4, out_channels=8)
        print(x.shape)
        return x


x = torch.randn(1,3,7,7)
y = torch.randn(1,8,5,5)
model = net()
print(model.conv1.weight)
loss_fn =torch.nn.MSELoss(reduction='sum')
lr = 1
optim = torch.optim.Adam(model.parameters(), lr=lr)
for t in range(5):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optim.zero_grad()
    loss.backward()
    optim.step()

print(model.conv1.weight)
