import ot
import os
import torch
import sys
from copy import deepcopy as dcopy
from WJDOT import *
from multi_task_learning import *

# ################
# Utilities      #
# ################

def load_data(datatype, data_dir):
    """
    Returns source or target data.

    Arguments
    ---------
    datatype: str
          it has to be 'sources' or 'targets'
    data_dir: str
          it is the path of the folder containing the data. 

    Returns
    -------
    names = [str, .., str]
          list of noises corresponding to the different noisy datasets
    train_sets = list of torch array
          list of training sets
    val_sets = list of torch array
          list of validation sets
    test_sets = list of torch array
          list of testing sets
    """
    if datatype == 'sources':
        names = torch.load(data_dir + 'sources_datasets_names.pt')
        train_sets = torch.load(data_dir + 'sources_train_datasets.pt')
        val_sets = torch.load(data_dir + 'sources_val_datasets.pt')
        test_sets = torch.load(data_dir + 'sources_test_datasets.pt')
        
    elif datatype == 'targets':
        names = torch.load(data_dir + 'targets_datasets_names.pt')
        train_sets = torch.load(data_dir + 'targets_train_datasets.pt')
        val_sets = torch.load(data_dir + 'targets_val_datasets.pt')
        test_sets = torch.load(data_dir + 'targets_test_datasets.pt')
    else:
        print('Choice not allowed. Datatype must be: sources or targets.')
        sys.exit()
    return names, train_sets, val_sets, test_sets


class Configuration(object):
    """
    This class contains the network and wjdot pararmeters.

    Arguments
    ---------
    S: int
            number of sources 
    hid_dim: int
            number of memory blocks in each LSTM layer
    embedding_dim: int
            dimension of the embedding space 
    num_classes: int
            number of classes
    num_epochs: int
            maximum number of training epochs
    maxerror: int
            maximum number of errors allowed before to apply early stopping
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
    """
    def __init__(self):
        self.S = 11
        self.hid_dim = 50
        self.embedding_dim = self.hid_dim * 2
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 2
        self.num_epochs = 1000
        self.maxerror = 100
        self.lr = 0.01
        self.lr_decay = 1.0
        self.l2_reg = 0.0
        self.beta = 1.0

# ###########
#    Main   #
# ###########

data_dir = str(sys.argv[1])  
ndomain = int(sys.argv[2]) # target domain - it has to be in [0, 1, 2, 3]
early_stopping = str(sys.argv[3]) # it has to be 'acc' or 'sse'

config = Configuration()
totorch = lambda x: torch.from_numpy(x)


# Load data
sources_names, sources_train_sets, sources_val_sets, _ = load_data('labelled', data_dir)
Ns_samples = [source.shape[0] for source in sources_train_sets[0]]
targets_names, targets_train_sets, targets_dev_sets, targets_test_sets = load_data('unlabelled', data_dir)

# Run MTL
mtl_config = MTL_Configuration()
mtl_net = run_mtl(mtl_config, sources_train_sets, sources_val_sets)
print('MTL training done.')

# MTL EMBEDDINGS:

# extract features from sources
sources_embeddings = [mtl_embedding(mtl_net, source.to(mtl_config.device)) for source in sources_train_sets[0]]
sources_embedding_array = torch.cat(sources_embeddings, 0).to(config.device)
sources_output_array = get_onehot_label(torch.cat(sources_train_sets[1], 0) .int(), config.num_classes).to(config.device)
sources_ey = torch.cat([sources_embedding_array, sources_output_array], 1)

# extract features from targets
targets_embedding_train = mtl_embedding(mtl_net, targets_train_sets[0][ndomain].to(mtl_config.device))
targets_embedding_test = mtl_embedding(mtl_net, targets_test_sets[0][ndomain].to(mtl_config.device)) 

# MSDA-WJDOT
print('\nMSDA-WJDOT')
net = ClassifierLayer(config).to(config.device)
if early_stopping == 'acc':
    sources_val_data = [mtl_embedding(mtl_net, source.to(mtl_config.device)).to(config.device) for source in sources_val_sets[0]],  [y.to(config.device) for y in sources_val_sets[1]]
    wjdot_alphas, wjdot_loss, wjdot_val_measure =  wjdot_acc(net, config, sources_ey, Ns_samples, sources_val_data, targets_embedding_train)
elif early_stopping == 'sse':
    targets_embedding_val = mtl_embedding(mtl_net, targets_dev_sets[0][ndomain].to(mtl_config.device))
    wjdot_alphas, wjdot_loss, wjdot_val_measure =  wjdot_sse(net, config, sources_ey, Ns_samples, targets_embedding_train, targets_embedding_val)

target_accuracy = inference(net, targets_embedding_test, targets_test_sets[1][ndomain].to(config.device))
print('\nTarget domain N.', ndomain)
print('Testing Accuracy: {:.4f}'.format(target_accuracy))


