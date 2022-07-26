'''
SIMULATED DATA: TARGET SHIFT

We consider a classification problem with 2 classes.
For sources and target we generate config.Ns_samples and config.N_samples samples
from config. from a 2D-Gaussian distribution.
Sources and target have different proportions of classes that are randomly generated.
The classification function is supposed to be f(x) = x * w + b

'''
import ot
import numpy as np
from WJDOT import *
from numpy import round
from ot.datasets import make_data_classif
torch.manual_seed(0)

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
    p_s: [float, ..., float]
           list of sources class proportions
    pt: float
           target class proportion
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
        self.sigma = 0.5
        self.Ns_samples = 50
        self.N_samples = 50
        self.testing_samples = 1000
        self.S = 20
        self.p_s = sorted(round(np.random.rand(self.S), 2))
        self.pt = round(np.random.rand(1), 1)[0]
        # Network parameters
        self.embedding_dim = 2
        self.num_classes = 2
        self.num_epochs = 1000
        self.lr = 0.001
        self.lr_decay = 1.0
        self.l2_reg = 0.0
        self.maxerror = 300
        # DA parameters
        self.bures_reg = 0.003
        self.beta = 1


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
print('Label proportion in Sources:', config.p_s)
print('Label proportion in Target:', config.pt)
print('\n')
print('MSDA-WJDOT:')

##

source_dataset, Nsample  = [], []
for i in range(config.S):
    xs, ys = make_data_classif('2gauss_prop', config.Ns_samples + 1, nz=config.sigma, p=config.p_s[i])
    Nsample.append(xs.shape[0])
    source_dataset.append((torch.tensor(xs).float(), get_onehot_label(ys, config.num_classes)))


xy_all = get_xy_matrix(source_dataset, 1)
xnp, ynp = make_data_classif('2gauss_prop', config.N_samples, nz=config.sigma, p=config.pt)
x = torch.tensor(xnp).float()
y = torch.tensor(ynp)

xt, yt = make_data_classif('2gauss_prop', config.N_samples, nz=config.sigma, p=config.pt)
xt = torch.tensor(xt).float()
yt = torch.tensor(yt)

net = LinearLayer(config)
epochs_alpha, epochs_loss = wjdot(net, config, xy_all, Nsample, x)
accuracy = inference(net, xt, yt)

print('\n Testing target accuracy: {:.4f}'.format(accuracy))
