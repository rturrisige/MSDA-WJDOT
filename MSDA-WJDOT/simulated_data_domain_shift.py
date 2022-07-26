'''
SIMULATED DATA

We consider a classification problem with 3 classes.
For sources and target we generate config.Ns_samples and config.N_samples samples
from config.S + 1 3D-Gaussian distributions obtained by applying a rotation.
The source angles of rotation are uniformely taken between 0 and config.amax.
The target angle config.theta is randomly chosen in the same interval.
The classification function is supposed to be f(x) = sigmoid(x * w + b)

'''

import ot
import sys
import os
import numpy as np
from WJDOT import *

tonpy = lambda x: x.detach().cpu().numpy()


# ################
# Utilities    ##
# ###############

class Configuration(object):
    '''
    This class contains the network and wjdot pararmeters.

    Arguments
    ---------

    sigma: float
           standard deviation of the gaussian distribution
    Ns_samples: int
           number of source training samples
    N_samples: int
           number of target training samples
    testing_samples: int
           number of target testing samples
    amax: float
           biggest rotation angle
    S: int
           number of sources
    theta_s: [float, ..., float]
           list of sources rotation angle
    theta: float
           target rotation angle
    embedding_dim: int
            dimension of the embedding space
    num_classes: int
            number of classes
    num_epochs: int
            number of training epochs
    lr: float
            learning rate
    lr_decay: float
            learning rate decay
    l2_reg: float
            weigt of the l2 regularization term
    bures_reg: float
            regularization weight in bures algorithm
    beta: float
            weight of feature loss in the cost distance
    '''

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Data parameters
        self.sigma = 0.8
        self.Ns_samples = 300
        self.N_samples = 300
        self.testing_samples = 1000
        self.amax = 2 * np.pi / 3
        self.S = 20
        self.theta_s = torch.tensor(list(np.linspace(0, self.amax, self.S)))
        self.theta = float(np.random.rand(1) * self.amax)
        # Network parameters
        self.embedding_dim = 3
        self.num_classes = 3
        self.num_epochs = 1000
        self.lr = 0.01
        self.lr_decay = 1.0
        self.l2_reg = 0.0
        # DA parameters
        self.bures_reg = 0.003
        self.beta = 1


def rotation_3D(x, theta, axis, device='cpu'):
    """
    Returns the input data rotated of an angle theta.

    Parameters
    ----------
    x : torch array, shape (n samples, n features)
        input matrix
    theta : float
        angle
    axis :  int
        axis of rotation
    device : 'cpu' or 'cuda'


    Returns
    -------
    Rx: torch array, shape (n samples, n features)
    """
    if axis == 0:
        R = torch.tensor([[1.0, 0.0, 0.0],
                          [0.0, np.cos(theta), -np.sin(theta)],
                          [0.0, np.sin(theta), np.cos(theta)]], device=device)
    elif axis == 1:
        R = torch.tensor([[np.cos(theta), 0.0, np.sin(theta)],
                          [0.0, 1.0, 0.0],
                          [-np.sin(theta), 0.0, np.cos(theta)]], device=device)
    else:
        R = torch.tensor([[np.cos(theta), -np.sin(theta), 0.0],
                          [np.sin(theta), np.cos(theta), 0.0],
                          [0.0, 0.0, 1.0]], device=device)
    Rx = x.mm(R)
    return Rx


def get_3D_data(n, theta, sigma=0.8, device='cpu', axis=0):
    """
    Returns a dataset samples from a 3D-Gaussian distribution

    Parameters
    ----------
    n : int
        number of samples
    theta : float
        angle
    sigma :  float
        standard deviation
    device : 'cpu' or 'cuda'
        device
    axis: int
        axis of rotation

    Returns
    -------
    x: torch array, shape (n samples, n features)
    y: torch array, shape (nsamples, n classes)
    """
    y = torch.zeros([n, 3], device=device)
    y[:n // 3, 0] = 1
    y[n // 3:(2 * n) // 3, 1] = 1
    y[(2 * n) // 3:, 2] = 1
    x = torch.randn(n, 3, device=device) * sigma
    x -= torch.mean(x, 0)
    x += y
    x = rotation_3D(x, theta, int(axis), device)
    return x, y


# ################
# Generate  data #
##################

config = Configuration()
wasserstein_cost = 'emd'
print('\n')
print('SIMULATED DATA')
print('Number of sources:', config.S)
print('Sources samples:', config.Ns_samples)
print('Target samples:', config.N_samples)
print('Feature dimension:', config.embedding_dim)
print('Number of classes:', config.num_classes)
print('\n')
print('Wasserstein function:', wasserstein_cost, '(Allowed choices: emd and bures)')
print('\n')

sources_train_data = [get_3D_data(config.Ns_samples, angle, config.sigma, config.device) for angle in config.theta_s]
target_x, target_y = get_3D_data(config.N_samples, config.theta, config.sigma, config.device)
Ns_list = [config.Ns_samples] * config.S
source_xy = get_xy_matrix(sources_train_data, config.beta)  # all training data

target_test_x, target_test_y = get_3D_data(config.testing_samples, config.theta, config.sigma, config.device)

# ################
# MSDA via WJDOT #
# ################

net = ClassifierLayer(config).to(config.device)

print('MSDA-WJDOT:')
epochs_alphas, epochs_cost = wjdot(net, config, source_xy, Ns_list, target_x, wasserstein_cost)
target_accuracy = inference(net, target_test_x, torch.max(target_test_y, 1)[1])

print('\n Testing target accuracy: {:.4f}'.format(target_accuracy))
