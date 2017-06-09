import torch
import torch.nn as nn
from cfg import *

class _netGFM(nn.Module):
    def __init__(self, layers):
        super(_netGFM, self).__init__()
        self._size = []
        self.type = 'netG'
        self.sub = nn.Sequential(*layers[0])
        self.remain = nn.Sequential(*layers[1])
    
    def forward(self, input = 0, condition = 0):
        z = input
        c = condition
        self.fm = []
        z = self.sub(z)
        self.fm.append(z)
        out = self.remain(z)
        return out, self.fm
    
    def layers_size(self):
        return self._size

    def info(self):
        return self.type



def create_convnets_GFM(cfg, x_dim = 0, c_dim = 0, batch_norm = False):
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


def build_netGFM(cfg_index, z_dim = 0, c_dim = 0, batch_norm=False):
    '''build sigle netG
    - Params:
    @cfg_index: index in cfg
    @z_dim: nosie dim
    @c_dim: conditions dim

    - Returns:
    the class netG
    '''
    network = []
    for index in cfg_index:
        network.append(create_convnets_GFM(netNumGConfig[index], z_dim, c_dim))
    return _netGFM(layers=network)


if __name__ == '__main__':
    netG = build_netGFM(['fc_layer_sub', 'fc_layer_sub_remain'], z_dim=100, c_dim = 0)
    print netG