
import ot
import numpy as np
from WJDOT import *
import matplotlib
from matplotlib import pyplot as pl
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
tonpy = lambda x: x.detach().cpu().numpy()
torch.manual_seed(0)
from ot.datasets import make_data_classif

# ################
# Utilities    ##
# ###############

def plot_ax(dec, name=None, da=1.5):
    pl.plot([dec[0], dec[0]], [dec[1] - da, dec[1] + da], 'k', alpha=0.5)
    pl.plot([dec[0] - da, dec[0] + da], [dec[1], dec[1]], 'k', alpha=0.5)
    if name:
        pl.text(dec[0] - .75, dec[1] + 1.8, name)

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
        self.sigma = 0.5
        self.Ns_samples = 50
        self.N_samples = 50
        self.testing_samples = 1000
        self.S = 2
        self.p_s = [0.2, 0.9]
        self.pt = 0.8
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
        self.criterion = nn.CrossEntropyLoss()


dect = [2, 0]

config = Configuration()

# GENERATE DATA
source_dataset, source_np_data = [], []

for i in range(2):
    xs, ys = make_data_classif('2gauss_prop', config.Ns_samples+ 1, nz=config.sigma, p=config.p_s[i], bias=dect)
    source_dataset.append((torch.tensor(xs).float(), get_onehot_label(ys, config.num_classes)))
    source_np_data.append((xs, ys))


xnp, ynp = make_data_classif('2gauss_prop', config.N_samples , nz=config.sigma, p=config.pt, bias=dect)
x = torch.tensor(xnp).float()
y = torch.tensor(ynp)

xy_all = get_xy_matrix(source_dataset, 1)

# MSDA-WJDOT
net = LinearLayer(config)

epochs_alpha, epochs_loss = wjdot(net, config, xy_all, [d[0].shape[0] for d in source_np_data], x)
acc = inference(net, x, y)
alpha = epochs_alpha[-1]

ypred = net(x)
ypred = np.argmax(tonpy(ypred), 1)

w1 = net.l1.weight[:, 0]
b1 = net.l1.bias[0]
w2 = net.l1.weight[:, 1]
b2 = net.l1.bias[1]

wx = w1[0] - w1[1]
wy = w2[0] - w2[1]
b = b1 - b2

x_min, x_max = -1, 5

y_min = - (b + wx * x_min) / wy
y_max = - (b + wx * x_max) / wy

##

pl.figure(1, (20, 6))
pl.clf()
lst_mark = ['^', 's']
dec_s = [[0, 3], [4, 3]]

pl.subplot(1, 3, 1)
plot_ax(dec_s[0], 'Source 1')
plot_ax(dec_s[1], 'Source 2')
plot_ax(dect, 'Target')

pl.scatter(source_np_data[0][0][:, 0]-2, source_np_data[0][0][:, 1]+3, c=source_np_data[0][1], s=35, marker=lst_mark[0], cmap='Set1', vmax=9,
           label='Source {} ({:1.2f}, {:1.2f})'.format(1, 1 - config.p_s[0], config.p_s[0]))

pl.scatter(source_np_data[1][0][:, 0]+2, source_np_data[1][0][:, 1]+3, c=source_np_data[1][1], s=35, marker=lst_mark[1], cmap='Set1', vmax=9,
           label='Source {} ({:1.2f}, {:1.2f})'.format(2, 1 - config.p_s[1], config.p_s[1]))

pl.scatter(x[:, 0], x[:, 1], c=y, s=35, marker='o', cmap='Set1', vmax=9,
           label='Target ({:1.2f}, {:1.2f})'.format(1 - config.pt, config.pt))
pl.title('Data', fontsize=20)
pl.xticks([])
pl.yticks([])
pl.legend(prop={'size': 12}, fancybox=True, shadow=True, loc='upper center', bbox_to_anchor=(0.5, 0.27))
pl.axis('equal')
#pl.axis('off')

pl.subplot(1, 3, 2)
for i in range(config.S):
    xs, ys = source_np_data[i]
    pl.scatter(xs[:, 0], xs[:, 1], c=ys, s=alpha[i] * 40, marker=lst_mark[i], cmap='Set1', vmax=9,
           label=r'Source {} ($\alpha_{}={:.2f}$)'.format(1, 1, alpha[i]))


pl.axis('equal')
pl.xticks([])
pl.yticks([])
pl.legend(prop={'size': 12}, fancybox=True, shadow=True, loc='upper center', bbox_to_anchor=(0.5, 0.23))
pl.title('Weighted Sources and Classifier', fontsize=20)
pl.plot([x_min, x_max],[y_min, y_max],'g',label='Classifier')
pl.subplots_adjust(wspace=.5)


pl.subplot(1, 3, 3)
pl.imshow(alpha[:, None],  aspect='auto')

pl.xlabel(r'$P_{T}^{2}=$' + str(config.pt), fontsize=12)
pl.xticks([])
pl.yticks([0, 1], config.p_s)
pl.ylabel(r'Source proportions $P_{j}^{2}$',  fontsize=12)
pl.colorbar()
pl.title(r'$\bf \alpha$ coefficients', fontsize=20)

pl.savefig('2Dsimulated_data_target_shift.pdf',bbox_inches='tight')
pl.subplots_adjust(wspace=.5)



