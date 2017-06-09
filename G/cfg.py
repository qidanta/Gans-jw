'''=====_netG===='''
# the define of net
netNumGConfig = {
    'fc': ['200d', 'R', '128d', 'R', '200', 'R', '128', 'R', '784', 'S'],
    'fc_competition': ['128', 'R', '784', 'S'],
    'fc_layer_sub': ['128', 'R'],
    'fc_layer_sub_remain': ['784', 'S'],
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
