import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nn
import torch.optim as optim
from pycrayon import CrayonClient
from tensorflow.examples.tutorials.mnist import input_data
from torch.autograd import Variable

from cfg import *
from D.netD import build_netD
from G.netG import build_netG
from util.network_util import create_nets, init_network, weight_init
from util.solver_util import create_couple2one_optims, create_optims
from util.train_util import (compute_loss, create_netG_indeps_sample,
                             mutil_steps, netD_fake)
from util.vision_util import(create_experiments, create_sigle_experiment, add2experiments)

mnist = input_data.read_data_sets('./mnist_data/data', one_hot=True)
mb_size = 64
z_dim = 100
h_dim = 128
x_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
lr = 1e-3
cnt = 0
nets_num = 10


cc = CrayonClient(hostname="localhost")
cc.remove_all_experiments()
D_exp = create_sigle_experiment(cc, 'D_loss')
G_exps = create_experiments(cc, 10)

netG_share = build_netG(config['G'][2], z_dim)
netG_indeps = create_nets(config['G'], h_dim, nets_num)
netD = build_netD(config['D'][2], x_dim)

init_network(netG_share)
init_network(netG_indeps)
init_network(netD)

D_solver = optim.Adam(netD.parameters(), lr=lr)
G_share_solver = optim.Adam(netG_share.parameters(), lr=lr)
G_indep_solver = create_optims(netG_indeps, [lr,])
G_solvers = create_couple2one_optims(netG_share, netG_indeps, [lr,])


'''
params = [item.size() for item in list(netG.parameters())]
print params
'''
for it in range(10):
    z = Variable(torch.randn(mb_size, z_dim))
    X, c = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))
    c = Variable(torch.from_numpy(c.astype('float32')))

    G_share_sample = netG_share(z)
    G_indep_sample = create_netG_indeps_sample(netG_indeps, G_share_sample)

    D_real = netD(X)
    D_fake = netD_fake(G_indep_sample, netD)
    D_loss, G_losses, index = compute_loss(D_real, D_fake)

    D_exp.add_scalar_value('D_loss', D_loss.data[0], step=it)
    add2experiments(G_losses, G_exps, step=it)

    D_loss.backward(retain_variables=True)
    D_solver.step()
    D_solver.zero_grad()

    mutil_steps(G_losses, G_share_solver, G_indep_solver, index)

    if it % 1000 == 0:
        G_share_sample = netG_share(z)
        G_indep_sample = create_netG_indeps_sample(netG_indeps, G_share_sample)
        for index_of_sampels, samples in enumerate(G_indep_sample):
            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            prefix = '{}netG_{}st_'.format(cnt, index_of_sampels)
            samples = samples.data.numpy()[:16]
            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

            if not os.path.exists('out/'):
                os.makedirs('out/')
            
            plt.savefig('out/{}.png'.format(prefix + str(cnt).zfill(3)), bbox_inches='tight')
            plt.close(fig)
            torch.save(netG_indeps[index_of_sampels].state_dict(), '%s/netG_indep_epoch_%d.pth' % ('./out', it))
            torch.save(netG_share.state_dict(), '%s/netG_share_epoch_%d.pth' % ('./out', it))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('./out', it))
        cnt += 1

