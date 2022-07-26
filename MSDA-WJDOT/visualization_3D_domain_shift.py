import ot 
import sys
import os
import numpy as np
from WJDOT import *
from simulated_data import get_3D_data
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 15})

tonpy = lambda x: x.detach().cpu().numpy()


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
    S: int > 0
           number of sources
    T: int > 0
           number of targets
    theta_s: [float, ..., float]
           list of sources rotation angle
    theta: [float, ..., float]
           list of targets rotation angle
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
        self.num_classes = 3                            			 	   # number of classes
        self.Ns_samples = 60                          				      	   # number of sources samples
        self.N_samples = 60                          					   # number of target samples
        self.testing_samples = 1000                 					   # number of test target samples
        self.amax = 2 * np.pi / 3                					   # maximum angle allowed in training data
        self.S = 50                               			        	   # number of sources
        self.T = 50  									   # number of targets
        self.theta_s = torch.tensor(list(np.linspace(0, self.amax, self.S)))		   # source angles
        self.theta = sorted([float(np.random.rand(1) * self.amax) for t in range(self.T)]) # target angle
        # Network parameters
        self.embedding_dim = 3                 				    		   # embedding dimension (== input dimension in this case)
        self.num_epochs = 1000                 				        	   # maximum number of training epochs
        self.lr = 0.01                     				    	           # learning rate
        self.lr_decay = 1.0								   # learning rate decay
        self.l2_reg = 0.0                               				   # l2 regularization weight
        # DA parameters	
        self.bures_reg = 0.03						    		   # bures regularization
        self.beta = 1							  		   # beta parameter in WJDOT


# ################
# Generate  data #
##################

config = Configuration()

print('\n')
print('SIMULATED DATA')
print('Number of sources:', config.S)
print('Sources samples:', config.Ns_samples)
print('Target samples:', config.N_samples)
print('Feature dimension:', config.embedding_dim)
print('Number of classes:', config.num_classes)
print('\n')

sources_train_data = [get_3D_data(config.Ns_samples, angle, config.sigma, config.device) for angle in config.theta_s]
targets_train_data = [get_3D_data(config.N_samples, angle, config.sigma, config.device) for angle in config.theta]
Ns_list = [config.Ns_samples] * config.S
source_xy = get_xy_matrix(sources_train_data, config.beta)  # all training data

targets_test = [get_3D_data(config.testing_samples, angle, config.sigma, config.device) for angle in config.theta]

# ################
# MSDA via WJDOT #
# ################


alphas, accuracy = [[], []], [[], []]
for t in range(config.T):
  target_x = targets_train_data[t][0]
  target_test_x, target_test_y = targets_test[t]
  i = 0
  for wasserstein_cost in ['emd', 'bures']:
     net = ClassifierLayer(config).to(config.device)
     epochs_alphas, epochs_cost = wjdot(net, config, source_xy, Ns_list, target_x, wasserstein_cost)
     target_accuracy = inference(net, target_test_x,  torch.max(target_test_y, 1)[1])
     alphas[i].append(epochs_alphas[-1])
     accuracy[i].append(target_accuracy)
     i += 1

# ################
# Plot           #
# ################

max_theta_s = max(config.theta_s)
max_theta_t = max(config.theta)

fig = plt.figure(figsize=(20, 5))
plt.suptitle('3D simulated data')
ax = fig.add_subplot(1, 3, 1)
ax.set_ylim([np.min(np.array(accuracy)) - 0.001, np.max(np.array(accuracy)) + 0.001])
ax.boxplot(accuracy, patch_artist=True, showfliers=False, labels=['WJDOT(E)', 'WJDOT(B)'])
ax.set_ylabel('Accuracy')


plt.subplot(1, 3, 2)
plt.imshow(np.array(alphas[0]), extent=[0, max_theta_s, max_theta_t, 0])
plt.colorbar()
plt.title(r'$\alpha$ (Exact OT)')

plt.subplot(1, 3, 3)
plt.imshow(np.array(alphas[1]), extent=[0, max_theta_s, max_theta_t, 0])
plt.colorbar()
plt.title(r'$\alpha$ (Bures)')


pl.savefig('3Dsimulated_data_domain_shift.pdf',bbox_inches='tight')

