import argparse

import torch

from GAN.competitiongan import _competitionGan
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--x_dim', default=784, help='channels of input data')
parser.add_argument('--z_dim', default=100, help='noise channels')
parser.add_argument('--mb_size', default=64, help='the size of each batch')
parser.add_argument('--img_size', default=28, help='for mnsit datasets, it is 28px')
parser.add_argument('--savepath', default='out/', help='the folder saved .pth and img results')
parser.add_argument('--display_it', default=100, help='iter how many times, display the result')
parser.add_argument('--g_model', default='fc_competition', help='what kind of netG')
parser.add_argument('--d_model', default='fc_competition', help='what kind of netD')
parser.add_argument('--niter', default=5, help='iter how many times')
parser.add_argument('--train', default=True, help='trian or test')
parser.add_argument('--cc', default=True, help='use tensorboard or not')
parser.add_argument('--cuda', default=False, help='use cuda or not')
parser.add_argument('--nums',  default=5, help='how many netG to compete!')
parser.add_argument('--Lambda',  default=10, help='belong to competition')
parser.add_argument('--random', default=True, help='random the best netG index')

opt = parser.parse_args()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data/torch_mnistdata', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=opt.mb_size, shuffle=True, **{})

#GAN = _competitionGan(opt)

for it in range(opt.niter):
    for index, (data, target) in enumerate(train_loader):
        print data.size()
        print target
