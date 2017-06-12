import torch
import torch.nn as nn
from cfg import *

class _netG(nn.Module):
    def __init__(self, layers, ngpu = 0):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self._size = []
        self.type = 'netG'
        self.main = nn.ModuleList(layers)
    
    def forward(self, input = 0, condition = 0):
        z = input
        c = condition
        for index in range(len(self.main)):
                z = self.main[index](z)
                self._size.append(z.size())
        return z
    
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


def build_netG(cfg_index, z_dim = 0, c_dim = 0, batch_norm=False):
    '''build sigle netG
    - Params:
    @netGS: index in cfg, shared layers
    @netGI: index in cfg, independ layers
    @z_dim: nosie dim
    @c_dim: conditions dim

    - Returns:
    the class netG
    '''
    network=create_convnets_G(netNumGConfig[cfg_index], z_dim, c_dim)
    return _netG(layers=network)


if __name__ == '__main__':
    network = create_convnets_G(netNumGConfig['dcgans'], 100, 0)
    netG = _netG(layers = network)
    print netG