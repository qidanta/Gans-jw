import argparse

import torch

from GAN.competitiongan import _competitionGan
from torchvision import datasets, transforms

# for test
from util.train_util import link2condition_data

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
parser.add_argument('--train', default=False, help='trian or test')
parser.add_argument('--cc', default=False, help='use tensorboard or not')
parser.add_argument('--cuda', default=False, help='use cuda or not')
parser.add_argument('--nums',  default=10, help='how many netG to compete!')
parser.add_argument('--Lambda',  default=10, help='belong to competition')
parser.add_argument('--random', default=False, help='random the best netG index')
parser.add_argument('--find_version', default='v1.1', help='v1.0: index by maxium fake_prob | v1.1: minium L1 distance between real and fake img |v1.2: index by minium distance between real and fake prob')
parser.add_argument('--condition_D', default=True)
parser.add_argument('--g_network_path', default='./result/competitionganv1.4/out/netG8_epoch_3900.pth')
parser.add_argument('--d_network_path', default='/home/eric/Desktop/Project-PY/pro-py27/01GANs/02competitiongan/result/competitionganv1.4/out/netD_epoch_3200.pth')

opt = parser.parse_args()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data/torch_mnistdata', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=opt.mb_size, shuffle=True, **{})

GAN = _competitionGan(opt)

for it in range(opt.niter):
    GAN.test(it)
