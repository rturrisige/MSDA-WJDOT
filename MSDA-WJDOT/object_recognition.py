import ot
import glob
import random
from copy import deepcopy as dcopy
import os
from WJDOT import *

##


class Configuration(object):
    """"
    This class contains the network and wjdot pararmeters.

    Arguments
    ---------
    device: 'cpu' or 'cuda'
            device
    ndatasets: int > 0
            number of datasets (sources + targets)
    S: int
            number of sources 
    embedding_dim: int
            dimension of the embedding space 
    num_classes: int
            number of classes
    num_epochs: int
            maximum number of training epochs
    maxerror: int
            maximum number of errors allowed before to apply early stopping
    batch_size: int
            data batch size
    lr: float
            learning rate
    lr_decay: float
            learning rate decay
    l2_reg: float
            weigt of the l2 regularization term
    beta: float
            weight of feature loss in the cost distance
    """
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.ndatasets = 4        		
        self.S = self.ndatasets - 1    		
        self.embedding_dim = 4096		
        self.num_classes = 10			
        self.num_epochs = 1000		
        self.maxerror = 100			
        self.batch_size = 100			
        self.lr = 0.01  			
        self.lr_decay = 1.0			
        self.l2_reg = 0.0			
        self.beta = 0.3		         	




config = Configuration()
tonpy = lambda x: x.detach().cpu().numpy()
totorch = lambda x: torch.from_numpy(x)
todevice= lambda x: torch.from_numpy(x).to(config.device)


##

def wjdot_object_recognition(config, training_datasets, validation_datasets, testing_datasets, early_stopping='sse', saver_dir=None):
   '''
   MSDA-WJDOT for each target domain
   
   Parameters
   ----------
   
   config: class 
           network and wjdot class parameters
   training_datasets: lists of torch array, [ [n samples, embedding dimension + 1], ... ]
           list of training sets
   validation_datasets: lists of torch array, [ [n samples, embedding dimension + 1], ... ]
           list of training sets
   testing_datasets: lists of torch array, [ [n samples, embedding dimension + 1], ... ]
           list of training sets
   early_stopping: 'sse' or 'acc'
           early stopping validation strategy
   saver_dir: str
           path to the folder in which the results will be saved
   '''
   if early_stopping not in ['acc', 'sse']:
      print('Early stopping has to be equal to "acc" or "sse"')
   if saver_dir:
       wjdot_file = open(saver_dir + 'msda_wjdot_results.txt', 'a')
       wjdot_file.write('Configuration:\n')
       for n, v in vars(config).items():
          wjdot_file.write(n + str(' = ') + str(v) + '\n')
       wjdot_file.write('\n')
       wjdot_file.flush()
       wjdot_file.write('\n Early stopping validation: {}\n'.format(early_stopping))
   for leave_out in range(config.ndatasets):
       results_dict = {}
       net = ClassifierLayer(config).to(config.device)
       # sources data
       sources_train_data = [totorch(training_datasets[i][:, :-1]) for i in range(config.ndatasets) if i != leave_out],  [totorch(training_datasets[i][:, -1]) for i in range(config.ndatasets) if i != leave_out]
       Ns_samples = [training_datasets[i].shape[0] for i in range(config.ndatasets) if i != leave_out]
       sources_xy = torch.cat( (torch.cat(sources_train_data[0], 0), get_onehot_label(torch.cat(sources_train_data[1], 0).int(), config.num_classes)), 1).to(config.device)
       # target data
       train_target_x = todevice(training_datasets[leave_out][:, :-1])
       test_target_x, test_target_y = todevice(testing_datasets[leave_out][:, :-1]), todevice(testing_datasets[leave_out][:, -1])                                                                           
       # WJDOT
       if early_stopping == 'acc':
            sources_val_data = [todevice(validation_datasets[i][:, :-1]) for i in range(config.ndatasets) if i != leave_out],  [todevice(validation_datasets[i][:, -1]) for i in range(config.ndatasets) if i != leave_out]
            wjdot_alphas, wjdot_loss, wjdot_val_measure =  wjdot_acc(net, config, sources_xy, Ns_samples, sources_val_data, train_target_x)
       elif early_stopping == 'sse':
            val_target_x = todevice(validation_datasets[leave_out][:, :-1])
            wjdot_alphas, wjdot_loss, wjdot_val_measure =  wjdot_sse(net, config, sources_xy, Ns_samples, train_target_x, val_target_x)    
       target_accuracy = inference(net, test_target_x, test_target_y)
       print('Domain: {} - Test accuracy: {:.4f}'.format(leave_out, target_accuracy))
       if saver_dir:
            results_dict['Epochs_alpha'] = wjdot_alphas
            results_dict['Epochs_loss'] = wjdot_loss
            results_dict['Early_stopping_measure'] = wjdot_val_measure
            results_dict['Target_testing_accuracy'] = target_accuracy 
            wjdot_file.write('Domain: {} \t Test accuracy: {:.4f}\n'.format(leave_out, target_accuracy))
            np.savez(saver_dir + 'MSDA_WJDOT_' + early_stopping + '_results_Domain' + str(leave_out) + '_.npz', **results_dict)
            wjdot_file.flush()

	    

