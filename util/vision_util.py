
def create_sigle_experiment(client, name):
    '''create experiment named name

    - Params:
    @client: CrayonClient
    @name: experiment name

    - Returns:
    a experiment named name
    '''
    try:
        exp = client.open_experiment(name)
    except Exception, e:
        exp = client.create_experiment(name)
    return exp

def create_experiments(client, num, prefix='G_loss'):
    '''create num experiments

    - Params:
    @client: Crayonclinet
    @num: how many experiments created
    @prefix: the name of experiment's prefix

    - Returns:
    contain mutil G_loss experiment
    '''
    exps = []
    for index in range(num):
        name = prefix + '_{}'.format(index)
        exps.append(create_sigle_experiment(client, name))
    return exps

def add2experiments(losses, exps, step, prefix='G_loss'):
    '''add G_loss's data to exps

    - Params:
    @lossess: G_losses
    @exps: experiment created by crayonclient
    @step: run times
    @prefix: the prefix of experiment's name
    '''
    couples = zip(losses, exps)
    for i, couple in enumerate(couples):
        name = prefix + '_{}'.format(i)
        couple[1].add_scalar_value(name, couple[0].data[0], step)
