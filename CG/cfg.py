'''=====_netD===='''
# the define of net
netDNumConfig = {
    'fc_noncondition': ['200', 'R', '1', 'S'],
    'fc_condition': ['128', 'R', '200', 'R', '128d', 'R', '200d', 'R', '1', 'S'],
    'fc_competition': ['128', 'R', '1', 'S' ],
    'fc_competition2': ['128', 'R', '100', 'R'],
    'mnist_classfier': ['128', 'R', '10', 'Softmax'],
    'dcgans': [(64, 4, 2, 1), 'LR', (128, 4, 2, 1), 'B', 'LR', (256, 4, 2, 1), 'B', 'LR', (512, 4, 2, 1), 'B', 'LR', (1, 4, 1, 0), 'S'],
    'dcgans-mnist': [(64, 4, 2, 1), 'LR', (128, 4, 2, 1), 'B', 'LR', (256, 4, 2, 1), 'B', 'LR', (512, 4, 2, 1), 'B', 'LR', (1, 4, 1, 0), 'S'],
    'chainer-dcgans': [(32, 3, 2, 1), 'LR', (32, 3, 2, 2), 'B', 'LR', (32, 3, 2, 1), 'B', 'LR', (32, 3, 2, 1), 'B', 'LR', (1, 3, 2, 1), 'S']
}
netNumDCConfig = {
    'fc_condition_out': [0, 6, 8],
    'fc_competition_out': [999]
}

'''=====_netG===='''
# the define of net
netNumGConfig = {
    'fc': ['200d', 'R', '128d', 'R', '200', 'R', '128', 'R', '784', 'S'],
    'fc_competition': ['128', 'R', '784', 'S'],
    'fc_competition2': ['128', 'R', '784', 'S'],
    'fc_cs': ['128', 'R'],
    'fc_ci': ['784', 'S'],
    'dcgans': [(512, 4, 1, 0), 'B', 'R', (256, 4, 2, 1), 'B', 'R', (128, 4, 2, 1), 'B', 'R', (64, 4, 2, 1), 'B', 'R', (3, 4, 2, 1), 'TH'],
    'dcgans-mnist': [(512, 4, 1, 0), 'B', 'R', (512, 4, 2, 1), 'B', 'R', (256, 4, 2, 1), 'B', 'R', (128, 4, 2, 1), 'B', 'R', (1, 4, 2, 1), 'TH'],
    'chanier-dcgans': [(128, 3, 2, 0), 'B', 'R', (128, 3, 2, 1), 'B', 'R', (128, 3, 2, 1), 'B', 'R', (128, 3, 2, 2), 'B', 'R', (1, 3, 2, 2), 'TH'],
}
# the define which layers to change
netNumGCConfig = {
    'fc_condition_out': [0, 2, 4],
    'fc_competition_out': [999]
}
