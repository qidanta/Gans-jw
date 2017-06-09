import torch.optim as optim

def create_optims(nets, config):
    '''create optim for each net in nets

    - Params:
    @nets: each one in nets is class netG
    @config: config for optim

    - Return:
    mutil optims
    '''
    num = len(nets)
    optims = []
    for net in nets:
        optims.append(optim.Adam(net.parameters(), *config))
    return optims

def create_couple2one_optims(net_share, net_indeps, config):
    '''v1.0 create mutil Adam optims by couple net_share and net_indep's parameters

    - Params:
    @net_share: single shared netG for sharing
    @net_indep: mutil indepently netG_indep

    - Returns:
    a list of solver by adding netG_share's parameters and net_indep's parameters
    '''
    optims = []
    num = len(net_indeps)
    net_shares = [net_share] * num
    optims = [ optim.Adam(list(net_share.parameters()) + list(net_indep.parameters()), *config) for net_share, net_indep in zip(net_shares, net_indeps) ]
    return optims

def create_couple_optims(net_share, net_indep):
    '''create mutil optims by couple net_share and net_indep

    - Params:
    @net_share: single netG_solver for sharing
    @net_indep: mutil indepently netG_solvers

    - Returns:
    a list of (net_share_optim, net_indep_optim)
    '''
    optims = []
    num = len(net_indep)
    net_shares = [net_share] * num 
    optims = zip(net_shares, net_indep)
    return optims

