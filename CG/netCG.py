import torch
import torch.nn as nn
from cfg import *

class _netCG(nn.Module):
    def __init__(self, layersHead, layersBody, nums, ngpu = 0):
        super(_netCG, self).__init__()
        self.ngpu = ngpu
        self.nums = nums
        self._size = []
        self.type = 'netCG'
        self.sub = []
        self.sub.append(nn.Sequential(*layersHead))
        for i in range(nums-1):
            self.sub.append(nn.Sequential(*layersBody))
        self.main = nn.Sequential(*self.sub)
    
    def forward(self, input = 0, condition = 0):
        z = input
        c = condition
        self.samples = []
        for i in range(self.nums):
            z = self.main[i](z)
            self.samples.append(z)
        return self.samples
    
    def layers_size(self):
        return self._size

    def info(self):
        return self.type




def create_convnets_G(cfg, x_dim = 0, c_dim = 0, batch_norm = False):
    layers = []
    i_dim = x_dim + c_dim
    for v in cfg:
        if v == 'R':
            layers += [nn.ReLU(inplace=True)]
        elif v == 'LR':
            layers += [nn.LeakyReLU(0.2, inplace=False)]
        elif v == 'S':
            layers += [nn.Sigmoid()]
        elif v == 'TH':
            layers += [nn.Tanh()]
        elif v == 'B':
            layers += [nn.BatchNorm2d(i_dim)]
        elif type(v) == tuple:
            o_dim, k, s, p = v
            layers += [nn.ConvTranspose2d(i_dim, o_dim, kernel_size=k, stride=s, padding=p, bias=False)]
            i_dim = o_dim
        else:
            if v[-1] == 'd':
                o_dim = int(v[:-1])
                layers += [nn.Linear(i_dim, o_dim, bias=True)]
                i_dim = o_dim + c_dim
            else:
                o_dim = int(v)
                layers += [nn.Linear(i_dim, o_dim, bias=True)]
                i_dim = o_dim
    return layers


def create_convnets_D(cfg, x_dim = 0, c_dim = 0, batch_norm = False):
    layers = []
    i_dim = x_dim + c_dim
    for v in cfg:
        if v == 'R':
            layers += [nn.ReLU(inplace=True)]
        elif v == 'LR':
            layers += [nn.LeakyReLU(0.2, inplace=True)]
        elif v == 'S':
            layers += [nn.Sigmoid()]
        elif v == 'Softmax':
            layers += [nn.Softmax()]
        elif v == 'B':
            layers += [nn.BatchNorm2d(i_dim)]
        elif type(v) == tuple:
            o_dim, k, s, p = v
            layers += [nn.Conv2d(i_dim, o_dim, kernel_size=k, stride=s, padding=p, bias=False)]
            i_dim = o_dim
        else:
            if v[-1] == 'd':
                o_dim = int(v[:-1])
                layers += [nn.Linear(i_dim, o_dim, bias=True)]
                i_dim = o_dim + c_dim
            else:
                o_dim = int(v)
                layers += [nn.Linear(i_dim, o_dim, bias=True)]
                i_dim = o_dim
    return layers



def build_netCG(cfg_index, z_dim = 100, x_dim = 784, nums = 2, c_dim = 0, batch_norm=False):
    '''build sigle netCG
    - Params:
    @cfg_index: what kind of netG
    @z_dim: nosie dim
    @x_dim: real img channels
    @c_dim: conditions dim

    - Returns:
    the class netG
    '''
    network_G=create_convnets_G(netNumGConfig[cfg_index], z_dim, c_dim)
    network_D = create_convnets_D(netDNumConfig[cfg_index], x_dim, c_dim)
    return _netCG(layersHead=network_G, layersBody=network_D + network_G, nums=10)


if __name__ == '__main__':
    netG = build_netCG('fc_competition2', nums=10)
    print netG.main[2]