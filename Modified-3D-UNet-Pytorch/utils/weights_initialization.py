import torch.nn.init  as init
import torch.nn as nn

def xavier_uniform_init_weights(m):
    if isinstance(m,nn.Conv3d):
        # leaky_relu default slope 0.01
        init.xavier_uniform_(m.weight.data, gain=init.calculate_gain('leaky_relu', 0.01))

def xavier_normal_init_weights(m):
    if isinstance(m,nn.Conv3d):
        # leaky_relu default slope 0.01
        init.xavier_normal_(m.weight.data, gain=init.calculate_gain('leaky_relu', 0.01))

def kaiming_init_weights(m):
    if isinstance(m,nn.Conv3d):
        # leaky_relu default slope 0.01
        init.kaiming_uniform_(m.weight.data, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

def init_weights(model, init_type='xavier_unifom'):
    if init_type == 'xavier_uniform':
        model.apply(xavier_uniform_init_weights)
    elif init_type == 'xavier_normal':
        model.apply(xavier_normal_init_weights)
    elif init_type == 'kaiming':
        model.apply(kaiming_init_weights)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)