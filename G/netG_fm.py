import torch
import torch.nn as nn
from cfg import *

class _netGFM(nn.Module):
    def __init__(self, layers, type):
        super(_netGFM, self).__init__()
        self._size = []
        self.type = type
        self.length = len(layers)
        if self.type == 'fc_linear':
            self.sub = nn.Sequential(*layers[0])
            self.remain = nn.Sequential(*layers[1])
        if self.type == 'dcgans':
            self.sub0 = nn.Sequential(*layers[0])
            self.sub1 = nn.Sequential(*layers[1])
            self.sub2 = nn.Sequential(*layers[2])
            self.sub3 = nn.Sequential(*layers[3])
            self.remain = nn.Sequential(*layers[4])
        if self.type == 'dcgans_nob':
            self.sub0 = nn.Sequential(*layers[0])
            self.sub1 = nn.Sequential(*layers[1])
            self.sub2 = nn.Sequential(*layers[2])
            self.sub3 = nn.Sequential(*layers[3])
            self.remain = nn.Sequential(*layers[4])
    
    def forward(self, input = 0, condition = 0):
        z = input
        c = condition
        self.fm = []
        if self.type == 'fc_linear':
            z = self.sub(z)
            self.fm.append(z)
            out = self.remain(z)
            return out, self.fm
        if self.type == 'dcgans':
            z = self.sub0(z)
            self.fm.append(z)
            z = self.sub1(z)
            self.fm.append(z)
            z = self.sub2(z)
            self.fm.append(z)
            z = self.sub3(z)
            self.fm.append(z)
            out = self.remain(z)
            return out, self.fm
        if self.type == 'dcgans_nob':
            z = self.sub0(z)
            self.fm.append(z)
            z = self.sub1(z)
            self.fm.append(z)
            z = self.sub2(z)
            self.fm.append(z)
            z = self.sub3(z)
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
    return layers, i_dim


def build_netGFM(cfg_index, z_dim = 0, c_dim = 0, batch_norm=False, gans_type='dcgans'):
    '''build sigle netG
    @Params:
    - cfg_index: index in cfg
    - z_dim: nosie dim
    - c_dim: conditions dim
    - gans_type: what's the type of netG

    @Returns:
    the class netG
    '''
    networks = []
    for index in cfg_index:
        network, z_dim = create_convnets_GFM(netNumGConfig[index], z_dim, c_dim)
        networks.append(network)
    return _netGFM(layers=networks, type=gans_type)


if __name__ == '__main__':
    netG = build_netGFM(['dcgans_nob_sub1', 'dcgans_nob_sub2', 'dcgans_nob_sub3', 'dcgans_nob_sub4', 'dcgans_remain'], z_dim=100, c_dim = 0, gans_type='dcgans_nob')
    print netG