import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from cfg import *
from D.netD import build_netD
from G.netG import build_netG
from torch.autograd import Variable

# global setting
mb_size = 64
z_dim = 100
h_dim = 128
x_dim = 784
niter = 10
cuda = False
model_path = './model/netG_epoch_22.pth'

# build gans
netG = build_netG(config['G'][1], z_dim)

# load weights for netG
netG.load_state_dict(torch.load(model_path))

# noise
z = torch.FloatTensor(1, z_dim)

# init cuda
if cuda:
    netG.cuda()
    z = z.cuda()

z = Variable(z)

for it in range(niter):
    z.data.resize_(1, z_dim).normal_(0, 1)
    sample = netG(z).data.numpy()
    plt.imshow(sample.reshape(28, 28))
    if not os.path.exists('result/'):
            os.makedirs('result/')
    plt.savefig('result/{}.png'.format(str(it).zfill(3)))
