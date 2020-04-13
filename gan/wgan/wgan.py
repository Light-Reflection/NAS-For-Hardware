import argparse
import os
import numpy as np
import math
import sys
sys.path.append(os.getcwd())

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=200, help="interval betwen image samples")
parser.add_argument("--save_interval", type=int, default=20, help='interval save model')
parser.add_argument("--save_path", type=str, default='./logs', help='interval save model')

opt = parser.parse_args()
print(opt)
if not os.path.exists(opt.save_path ):

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
from generator.operations import ManualConv2d, ManualLinear, ManualBN1d



class manual_block(nn.Module):
    def __init__(self, max_inc, max_outc, normalize=True):
        super(manual_block, self).__init__()
        self._linear = ManualLinear(max_in_channels=max_inc, max_out_channels=max_outc)
        self._norm = normalize
        if self._norm:
            self._bn = ManualBN1d(max_outc, 0.8)
        self._acti = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, inc=None, outc=None):
        # print(inc, outc)
        x = self._linear(x, inc, outc)
        if self._norm:
            x = self._bn(x, outc)
        x = self._acti(x)
        return x



class SuperGen(nn.Module):
    def __init__(self, width=1.0):
        super(SuperGen, self).__init__()
        self.cfg = [128, 256, 512, 1024]
        if width>1:
            self.cfg = self.expand_cfg(width)

        self.model = nn.ModuleList()

        max_inc = opt.latent_dim
        for i, max_outc in enumerate(self.cfg):
            if i == 0:
                self.model.append(manual_block(max_inc, max_outc, normalize=False))
            else:
                self.model.append(manual_block(max_inc, max_outc))
            max_inc = max_outc

        last_max_channel = self.cfg[-1]
        self.linear = ManualLinear(last_max_channel, int(np.prod(img_shape)))
        self.tanh = nn.Tanh()

    def forward(self, x, mode='part', nums=2):
        if mode == 'part':
            c_encoding = self.produce_encoding(nums) # or using some convert function
        else:
            c_encoding = self.cfg
        inc = opt.latent_dim
        for block, outc in zip(self.model, c_encoding):
            x = block(x, int(inc), int(outc))
            inc = outc

        last_channel = int(c_encoding[-1])
        x = self.linear(x, in_channels=last_channel)
        x = self.tanh(x)
        x = x.view(x.shape[0], *img_shape)
        return x

    def expand_cfg(self, factor=1):
        new_cfg = []
        for chs in self.cfg:
            new_cfg.append(chs*factor)
        return new_cfg

    def produce_encoding(self, nums=2):
        lens = len(self.cfg)
        ori_enc = np.random.randint(1, nums+1, size=lens)
        forword_cfg = []
        for chs, multi in zip(self.cfg, ori_enc):
            chs = chs * multi / nums
            forword_cfg.append(chs)
        return forword_cfg


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
# generator = SuperGen()
discriminator = Discriminator()


if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )
"""load asian celeba """
#mean = np.array([0.485, 0.456, 0.406])
#std = np.array([0.229, 0.224, 0.225])
#data_dir = './'
#import torchvision
#dataset = datasets.ImageFolder(data_dir,
#                               transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]),
#                               )


"""load lfw"""
from gan.dataloader import LFWDataset
from torch.utils.data import DataLoader
dataset = LFWDataset(root_dir='/media/dm/d/Projects/Brandom_Amos/data/dcgan-completion.tensorflow/data/your-dataset/celeba_aligned/img_align_celeba', transforms=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    for i, imgs in enumerate(dataloader):

        # Configure input
        # print(type(imgs))
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        # print(type(generator))
        # print(generator(z))
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )
        if batches_done % opt.save_interval == 0:
            torch.save({'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict()
                }, )

        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "./gan/wgan/images-{}/{}.png".format(2,batches_done), nrow=5, normalize=True)
        batches_done += 1
