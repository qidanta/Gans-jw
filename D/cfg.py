'''=====_netD===='''
# the define of net
netDNumConfig = {
    'fc_noncondition': ['200', 'R', '1', 'S'],
    'fc_condition': ['128', 'R', '200', 'R', '128d', 'R', '200d', 'R', '1', 'S'],
    'fc_competition': ['128', 'R', '1', 'S' ],
    'fc_layer_sub': ['128', 'R'],
    'fc_layer_sub_remain': ['1', 'S'],
    'mnist_classfier': ['128', 'R', '10', 'Softmax'],
    'dcgans': [(64, 4, 2, 1), 'LR', (128, 4, 2, 1), 'B', 'LR', (256, 4, 2, 1), 'B', 'LR', (512, 4, 2, 1), 'B', 'LR', (1, 4, 1, 0), 'S'],
    'dcgans-mnist': [(64, 4, 2, 1), 'LR', (128, 4, 2, 1), 'B', 'LR', (256, 4, 2, 1), 'B', 'LR', (512, 4, 2, 1), 'B', 'LR', (1, 4, 1, 0), 'S'],
    'chainer-dcgans': [(32, 3, 2, 1), 'LR', (32, 3, 2, 2), 'B', 'LR', (32, 3, 2, 1), 'B', 'LR', (32, 3, 2, 1), 'B', 'LR', (1, 3, 2, 1), 'S']
}
netNumDCConfig = {
    'fc_condition_out': [0, 6, 8],
    'fc_competition_out': [999]
}