import argparse

import torch

from GAN.competiion_gan3_2 import _competitionGan
from torchvision import datasets, transforms

# for test
from util.train_util import link2condition_data

parser = argparse.ArgumentParser()
parser.add_argument('--x_dim', default=1, help='channels of input data')
parser.add_argument('--z_dim', default=100, help='noise channels')
parser.add_argument('--mb_size', default=64, help='the size of each batch')
parser.add_argument('--img_size', default=64, help='for mnsit datasets, it is 28px')
parser.add_argument('--fine_size', default=28, help='img final size')
parser.add_argument('--savepath', default='out/', help='the folder saved .pth and img results')
parser.add_argument('--display_it', default=500, help='iter how many times, display the result')
parser.add_argument('--g_model', default=['dcgans_sub1', 'dcgans_sub2', 'dcgans_sub3', 'dcgans_sub4', 'dcgans_remain'], help='what kind of netG')
parser.add_argument('--d_model', default=['dcgans_sub1', 'dcgans_sub2', 'dcgans_sub3', 'dcgans_sub4', 'dcgans_remain']  , help='what kind of netD')
parser.add_argument('--c_model', default='mnist_classfier', help='what kind of classfier netD')
parser.add_argument('--niter', default=5, help='iter how many times')
parser.add_argument('--train', default=True, help='trian or test')
parser.add_argument('--cc', default=True, help='use tensorboard or not')
parser.add_argument('--cuda', default=True, help='use cuda or not')
parser.add_argument('--nums',  default=10, help='how many netG to compete!')
parser.add_argument('--Lambda',  default=10, help='belong to competition')
parser.add_argument('--random', default=False, help='random the best netG index')
parser.add_argument('--find_version', default='v1.1', help='v1.0: index by maxium fake_prob | v1.1: minium L1 distance between real and fake img |v1.2: index by minium distance between real and fake prob')
parser.add_argument('--condition_D', default=0)
parser.add_argument('--g_network_path', default='/home/eric/Desktop/Project-PY/pro-py27/01GANs/02competitiongan/result/competitionganv1.4/out/netD_epoch_4600.pth')
parser.add_argument('--d_network_path', default='/home/eric/Desktop/Project-PY/pro-py27/01GANs/02competitiongan/result/competitionganv1.4/out/netD_epoch_3200.pth')

opt = parser.parse_args()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data/torch_mnistdata', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Scale(opt.img_size),
                       transforms.ToTensor()
                   ])),
    batch_size=opt.mb_size, shuffle=True, **{})

GAN = _competitionGan(opt)
print GAN
print GAN.info()


for it in range(opt.niter):
    for index, (data, target) in enumerate(train_loader):
        GAN.train(data, target)
        if GAN.cnt % opt.display_it == 0:
             GAN.store(it)
