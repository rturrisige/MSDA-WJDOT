import numpy as np
import pylab as pl
import scipy as sp
import torch
import ot_torch as ott

import WJDOT

import matplotlib
matplotlib.use('Agg') 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from mpl_toolkits.mplot3d import Axes3D

tnp = lambda x: x.detach().cpu().numpy()
torch.manual_seed(0)

# ###############
# Utilities    ##
#################


def get_2D_data_onaxes(n, theta, sigma=1, device='cpu'):
    """
    Generate 2D-Gaussian distributed data

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

    Returns
    -------
    x: torch array, shape (n samples, n features)
    y: torch array, shape (nsamples, n classes)
    """
    R = torch.tensor([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]], device=device)
    y = torch.zeros([n, 2], device=device)
    y[:n//2, 0] = 1
    y[n//2:, 1] = 1
    x = torch.randn(n, 2, device=device)*sigma
    x -= torch.mean(x, 0)
    x += y
    x = x.mm(R)
    return x, y


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
    S: int > 0
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
    beta: float
            weight of feature loss in the cost distance
    '''

    def __init__(self):
        self.sigma = .15
        self.num_classes = 2                                     # number of classes
        self.Ns_samples = 30      
        self.N_samples = 30   
        self.S = 4  
        self.theta_s = [-.6, -.2, .2,.6]
        self.theta = 0.0   # test angle
        self.embedding_dim = 2                                        # input dimension
        self.num_epochs = 1000                                       # maximum number of iterations
        self.lr = 0.01   
        self.lr_decay = 1.0
        self.l2_reg = 0.0                                        # learning rate
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.beta = 1




# ####################
#  Generate data    #
# ####################

config=Configuration()

# target
x, y = get_2D_data_onaxes(config.N_samples, config.theta, sigma=config.sigma)
yi = torch.argmax(y,1)

# sources
source_data = [get_2D_data_onaxes(config.Ns_samples, s, config.sigma) for s in config.theta_s]
source_yi = [torch.argmax(temp,1) for temp0, temp in source_data]
source_xy = WJDOT.get_xy_matrix(source_data)

# #################
# MSDA via WJDOT ##
# #################

net = WJDOT.ClassifierLayer(config)

epochs_alpha, epochs_loss = WJDOT.wjdot(net, config, source_xy, [config.Ns_samples]*config.S, x)
alpha = epochs_alpha[-1]

ypred = tnp(net(x))
ypred = ypred[:,1] - ypred[:,0]

w = net.l1.weight[:,0]
b = net.l1.bias[0]

y_05 = (.5*w[0])/w[1]
y_15 = (-1.5*w[0])/w[1]


# ##################
#  Plots          ##
# ##################


pl.figure(1, (16, 3.5))
pl.clf()

lst_mark=['^','s','o','d']

# Subplot 1
ax = pl.subplot(1,4,1, projection='3d')

for k, (xk,yk) in enumerate(source_data):
    ax.scatter(xk[:,0], 1.0*(k+1)*np.ones(config.Ns_samples), xk[:,1], c=source_yi[k],marker=lst_mark[k], 
               cmap='jet', alpha=0.5, label='Source ${}$'.format(k+1))
    

pl.legend()
ax.set_xlabel("$x_1$")
ax.set_ylabel("Source $s$")
ax.set_zlabel("$x_2$")
ax.set_yticks((1,2,3,4))
ax.view_init(24,-30)
pl.title('Rotation of Target distributions',position=(.5,1.08))
ax.zaxis._axinfo['juggled'] = (1,2,1)

# Subplot 2 

pl.subplot(1,4,2)
pl.scatter(x[:,0],x[:,1],c='k',marker='+', alpha=0.8,label='Target',s=50) 

for k, (xk,yk) in enumerate(source_data):
    pl.scatter(xk[:,0], xk[:,1], c=source_yi[k], marker=lst_mark[k],
               cmap='jet', alpha=0.5, label='Source ${}$'.format(k+1), s=20)


pl.xlim(-.5,1.5)
pl.ylim(-.5,1.5)
pl.legend()
pl.xlabel('$x_1$')
pl.ylabel('$x_2$')
pl.title('Sources and Target distributions')

# Subplot 3

pl.subplot(1,4,3)
pl.scatter(x[:,0],x[:,1],c='k',marker='+',alpha=0.8,label='Target',s=50) 

for k,(xk,yk) in enumerate(source_data):
    if alpha[k]>1e-3:
        pl.scatter(xk[:,0], xk[:,1], c=source_yi[k], marker=lst_mark[k], 
                   cmap='jet', alpha=0.5, label='Source ${}$'.format(k+1), s=alpha[k]*40)

pl.xlim(-.5,1.5)
pl.ylim(-.5,1.5)
pl.legend()
pl.xlabel('$x_1$')
pl.ylabel('$x_2$')
pl.title('Reweighted Sources and Target')

# Subplot 4

pl.subplot(1,4,4)
pl.scatter(x[:,0],x[:,1],c=ypred,marker='+',label='Target prediction', s=50,cmap='jet') 
pl.plot([-.5,1.5],[y_05,y_15],'g',label='Classifier')
pl.xlim(-.5,1.5)
pl.ylim(-.5,1.5)
pl.legend()
pl.xlabel('$x_1$')
pl.ylabel('$x_2$')
pl.title('Estimated Target labels and Classifier')

pl.savefig('2Dsimulated_data_domain_shift.pdf',bbox_inches='tight')
