import argparse
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

import torchvision.utils as vutils
from baseModel import _baseModel
from D.netD import build_netD
from G.netG import build_netG
from util.network_util import create_nets, init_network
from util.solver_util import create_optims
from util.train_util import find_best_netG, find_best_netG_v1dot1, find_best_netG_v1dot2
from util.vision_util import create_sigle_experiment
import random


class _competitionGan(_baseModel):
    def __init__(self, opt):
        super(_competitionGan, self).__init__(opt)
        self.opt = opt
        self.x_dim = opt.x_dim
        self.z_dim = opt.z_dim
        self.img_size = opt.img_size
        self.nums = opt.nums
        self.Lambda = opt.Lambda
        self.savepath = opt.savepath
        self.cnt = 0

        self.netGs = create_nets(self.g_model, opt.x_dim, opt.nums, type='G')
        self.netD = build_netD(opt.d_model, opt.x_dim)


        X = torch.FloatTensor(opt.mb_size, opt.x_dim, opt.img_size, opt.img_size)
        Z = torch.FloatTensor(opt.mb_size, opt.z_dim, 1, 1)
        real_like_sample = torch.FloatTensor(opt.mb_size, opt.x_dim, opt.img_size, opt.img_size)
        fake_like_sample = torch.FloatTensor(opt.mb_size, opt.x_dim, opt.img_size, opt.img_size)
        label = torch.FloatTensor(opt.mb_size)
        self.criterionGAN = torch.nn.BCELoss()
        self.criterionL1 = torch.nn.L1Loss()

        if self.cuda:
            netD.cuda()
            netG.cuda()
            self.criterionGAN.cuda()
            self.L1loss.cuda()
            X, Z = X.cuda(), Z.cuda()
            real_like_sample, fake_like_sample = real_like_sample.cuda(), fake_like_sample.cuda()
            label = label.cuda()

        self.X = Variable(X)
        self.Z = Variable(Z)
        self.real_like_sample = Variable(real_like_sample)
        self.fake_like_sample = Variable(fake_like_sample)
        self.label= Variable(label)

        self.create_tensorboard()
        self.index_exp = create_sigle_experiment(self.cc, 'index')
        self.D_solver = torch.optim.Adam(self.netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.G_solvers = create_optims(self.netGs, [2e-4, (0.5, 0.999)])

        if opt.train == False:
            self.load_networkG(self.opt.g_network_path)
            self.load_networkD(self.opt.d_network_path)
        else:
            init_network(self.netD)
            init_network(self.netGs)
    
    def draft_data(self, input):
        self.mb_size = input.size(0)
        self.X.data.resize_(input.size()).copy_(input)
        self.X.data.resize_(self.mb_size, self.x_dim, self.img_size, self.img_size)
        self.Z.data.resize_(self.mb_size, self.z_dim).normal_(0, 1)
        self.label.data.resize_(self.mb_size)

    def generate_samples(self):
        '''belong to cometitiongan, generate samples by self.netGs

        - Returns
        a list of fake_samples create by netGs
        '''
        fake  = []
        for netG in self.netGs:
            fake.append(netG(self.Z))
        return fake

    def backward_D(self):
        self.D_real = self.netD(self.X)

        self.D_fake = []
        self.fake = self.generate_samples()
        for sample in self.fake:
            self.D_fake.append(self.netD(sample))
        
        # best index of netG
        if self.opt.find_version == 'v1.2':
            index = find_best_netG_v1dot2(self.D_real, self.D_fake)
        if self.opt.find_version == 'v1.1':
            index = find_best_netG_v1dot1(self.criterionL1, self.fake, self.X)
        else:
            index = find_best_netG(self.D_fake)
        
        if self.opt.random and self.cnt % self.opt.display_it == 0:
            index = random.randint(0, self.nums-1)
        self.real_like_prob = self.D_fake[index]
        self.real_like_sample.data.copy_(self.fake[index].data)

        # real
        self.label.data.fill_(1)
        self.loss_D_real = self.criterionGAN(self.D_real, self.label)
        self.loss_D_real.backward(retain_variables=True)

        # fake
        self.label.data.fill_(0)
        self.loss_D_real_like = self.criterionGAN(self.real_like_prob, self.label)
        self.loss_D_real_like.backward(retain_variables=True)

        # lambda
        preb_D_gap = []
        for i in range(self.nums):
            if i != index:
                gap = self.criterionL1(self.fake[i], self.real_like_sample) * self.Lambda
                gap.backward(retain_variables=True)
                preb_D_gap.append(gap)
        self.loss_D_lambda = sum(preb_D_gap)/len(preb_D_gap)

        self.loss_D = self.loss_D_real + self.loss_D_real_like + self.loss_D_lambda
        self.cnt = self.cnt + 1
        self.D_exp.add_scalar_value('D_loss', self.loss_D.data[0], step=self.cnt)

    def backward_G(self):
        D_fake = []
        for sample in self.fake:
            D_fake.append(self.netD(sample))
        
        # best index of netG
        if self.opt.find_version == 'v1.2':
            index = find_best_netG_v1dot2(self.D_real, self.D_fake)
        if self.opt.find_version == 'v1.1':
            index = find_best_netG_v1dot1(self.criterionL1, self.fake, self.X)
        else:
            index = find_best_netG(self.D_fake)
        
        if self.opt.random and self.cnt % self.opt.display_it == 0:
            index = random.randint(0, self.nums-1)
        self.index_exp.add_scalar_value('index', index, step=self.cnt)
        self.fake_like_prob = self.D_fake[index]
        self.fake_like_sample.data.copy_(self.fake[index].data)

        self.label.data.fill_(1)
        self.loss_G_fake_like = self.criterionGAN(self.fake_like_prob, self.label)
        self.loss_G_fake_like.backward(retain_variables=True)

        preb_G_gap = []
        for i in range(self.nums):
            if i != index:
                gap = self.criterionL1(self.fake[i], self.fake_like_sample) * self.Lambda
                gap.backward(retain_variables=True)
                preb_G_gap.append(gap)
        self.loss_G_lambda = sum(preb_G_gap)/len(preb_G_gap)

        self.loss_G = self.loss_G_fake_like + self.loss_G_lambda
        self.best_netG_index = index
        self.G_exp.add_scalar_value('G_loss', self.loss_G.data[0], step=self.cnt)
    
    def train(self, input):
        self.draft_data(input)

        self.netD.zero_grad()
        self.backward_D()
        self.D_solver.step()

        for netG in self.netGs:
            netG.zero_grad()
        self.backward_G()
        for solver in self.G_solvers:
            solver.step()

    def continue_train(self, input):
        '''use competition mode in continue_train func for try, 
        ...not in train func
        '''
        self.draft_data(input)

        self.netD.zero_grad()
        self.backward_D()
        self.D_solver.step()

        for netG in self.netGs:
            netG.zero_grad()
        self.backward_G()
        for solver in self.G_solvers:
            solver.step()
    
    def test(self, input, cnt):
        self.draft_data(input)
        fake = self.netGs[0].forward(self.Z)
        self.save_image(fake, cnt, self.savepath)


    def load_networkG(self, g_network_path):
        '''load network parameters of netG and netD

        - Params:
        @g_network_path: the path of netG
        '''
        for netG in self.netGs:
            netG.load_state_dict(torch.load(g_network_path))
    
    def save_network(self, it, savepath):
        torch.save(self.netGs[self.best_netG_index].state_dict(), '{}/netG{}_epoch_{}.pth'.format(savepath, self.best_netG_index, it))
        torch.save(self.netD.state_dict(), '{}/netD_epoch_{}.pth' .format(savepath, it))

    def save_image(self, fake, it , savepath):
        '''save result of netG output

        - Params:
        @fake: the output of netG
        @it: number of iterations
        @savepath: in savepath, save network parameter
        '''
        samples = fake.data.resize_(self.mb_size, 1, self.opt.img_size, self.opt.img_size).numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for index, sample in  enumerate(samples):
            ax = plt.subplot(gs[index])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.img_size, self.img_size), cmap='Greys_r')

        plt.savefig(self.savepath+ '/{}.png'.format(str(self.cnt)), bbox_inches='tight')


    def store(self):
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        self.save_network(self.cnt, self.savepath)
        self.save_image(self.fake_like_sample, self.cnt, self.savepath)

    def __str__(self):
        netG = ''
        for G in self.netGs:
            netG += G.__str__()
        netD = self.netD.__str__()
        return 'Gan:\n' + '{}{}'.format(netG, netD)

    def gan_type(self):
        '''print what the gan's type
        '''
        return {'G': self.opt.g_model, 'D': self.opt.d_model}

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_dim', default=784, help='channels of input data')
    parser.add_argument('--z_dim', default=100, help='noise channels')
    parser.add_argument('--mb_size', default=64, help='the size of each batch')
    parser.add_argument('--img_size', default=28, help='for mnsit datasets, it is 28px')
    parser.add_argument('--savepath', default='out/', help='the folder saved .pth and img results')
    parser.add_argument('--g_model', default='fc_competition', help='what kind of netG')
    parser.add_argument('--d_model', default='fc_competition', help='what kind of netD')
    parser.add_argument('--train', default=True, help='trian or test')
    parser.add_argument('--cc', default=True, help='use tensorboard or not')
    parser.add_argument('--cuda', default=False, help='use cuda or not')
    parser.add_argument('--nums',  default=5, help='how many netG to compete!')
    parser.add_argument('--Lambda',  default=10, help='belong to competition')
    parser.add_argument('--find_version', default='v1.0', help='v1.0: index by maxium fake_prob | v1.2: index by minium distance bwteen real and fake')


    opt = parser.parse_args()
    if opt.find_version == 'v1.0':
        print True
    gans = _competitionGan(opt)
    print gans
