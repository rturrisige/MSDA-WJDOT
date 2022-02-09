"""
# rosanna.turrisi@edu.unige.it
This code performs training and evaluation of a 3D-CNN model. It requires the following external arguments:
- data_path: path to the folder "dataset" in which file in numpy format are saved (str)
- nexp in Configuration class: number of experiment (int)
- data_augmentation in DataParameter class(str):
  "0" corresponds to non-augmented data,
  "1" corresponds to zoom,
  "2" corresponds to shift,
  "3" corresponds to rotation,
  "123" applies separately zoom, shift, and rotation,
  "4" applied all transformations (zoom, shift,rotation) simultaneously
- ndataset in DataParameter class sets how many augmented datasets have to be used (int)
- out_channels in NetParameter class sets the number of filters in each convolutional layer (list)

It takes numpy data in data_path and saves results in saver_path + "Exp" + str(nexp) + "/".
Specifically, the following files are saved:
- "logfile.txt" reports Network and Training parameters, the Data pre-processing description, the number of k-folds and
  the number of training/validation/testing samples for each fold. Further, it shows the Validation and Testing accuracy
  for each k-fold and on average.
- "logfile_fold" + str(k) + ".txt", with k number of fold, displays training information (learning rate, loss,
   train accuracy - if required- , validation accuracy) at each epoch and information about early stopping.
- "Fold" + str(k) + "_training_info.npy" is a dictionary where the key (e.g., Loss) provides information of the value
   content (in the numpy format) which represents the key value at each epoch.
- "fold" + str(k) "_cnn_best_weights.pt", with k number of fold, saves the model weights at the best epoch
- "cnn_weights_init.pt" saves the initialized model weights
- "Best_val_and_test_accuracy.npy" is a dictionary containing the validation and testing accuracy at the best epoch.

"""
##
import ast
import sys
from cnn_utilities import *
data_path = str(sys.argv[1])


##
# ######################
#    CONFIGURATION    ##
# ######################


# TRAINING PARAMETERS:
class Configuration(object):
    def __init__(self):
        self.nexp = int(sys.argv[2])
        self.n_classes = 2  # number of classes
        self.batch_size = 50  # bach size
        self.num_epochs = 200  # maximum number of iterations
        self.maxerror = 20  # maximum allowed error before applying early stopping
        self.lr = 0.001  # learning rate
        self.lr_decay = 1.0  # learning rate decay
        self.l2_reg = 0.01  # weight of the l2 penalty loss
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()  # loss function
        self.return_train_acc = False
        self.return_train_prob = False


# DATA PARAMETERS
class DataParameter(object):
    def __init__(self):
        self.classes = ['CN', 'AD']
        self.dataset_type = ''
        self.data_augmentation = str(sys.argv[3])
        self.data_augmentation_dic = {'0': 'no_augmentation', '1': 'zoom', '2': 'shift', '3': 'rotation', '4': 'all'}
        if self.data_augmentation == '0':
            self.data_augmentation = False
            self.dataset_type += 'no_augmentation'
            print('No Data Augmentation')
        else:
            self.ndatasets = int(sys.argv[4])
            self.data_augmentation_type = []
            self.dataset_type += 'augmentation'
            print('Data Augmentation is performed with:')
            for s in self.data_augmentation:
                self.data_augmentation_type.append(self.data_augmentation_dic[s])
                self.dataset_type += '_' + self.data_augmentation_dic[s]
                print(self.data_augmentation_dic[s])
            self.dataset_type += '/' + str(self.ndatasets) + 'datasets'


# MODEL PARAMETERS
class NetParameter(object):
    def __init__(self):
        self.input_dim = [73, 96, 96]
        self.n_conv = int(sys.argv[5])
        self.kernels = [(3, 3, 3)] * self.n_conv
        if self.n_conv == 4:
            self.out_channels = [8, 16, 32, 64]
            self.pooling = [(4, 4, 4), (3, 3, 3), (2, 2, 2), (2, 2, 2)]
        elif self.n_conv == 6:
            self.out_channels = [8, 8, 16, 16, 32, 64]
            self.pooling = [(4, 4, 4), (0, 0, 0), (3, 3, 3), (2, 2, 2), (2, 2, 2), (0, 0, 0)]
        elif self.n_conv == 8:
            self.out_channels = [8, 8, 16, 16, 32, 32, 64, 64]
            self.pooling = [(4, 4, 4), (0, 0, 0), (3, 3, 3), (0, 0, 0), (2, 2, 2),
                            (0, 0, 0), (2, 2, 2), (0, 0, 0)]
        elif self.n_conv == 10:
            self.out_channels = [8, 8, 8, 16, 16, 16, 32, 32, 64, 64]
            self.pooling = [(4, 4, 4), (0, 0, 0), (0, 0, 0), (3, 3, 3), (0, 0, 0), (0, 0, 0), (2, 2, 2),
                            (0, 0, 0),  (2, 2, 2), (0, 0, 0), ]
        elif self.n_conv == 12:
            self.out_channels = [8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64]
            self.pooling = [(4, 4, 4), (0, 0, 0), (0, 0, 0), (3, 3, 3), (0, 0, 0), (0, 0, 0), (2, 2, 2),
                            (0, 0, 0), (0, 0, 0), (2, 2, 2), (0, 0, 0), (0, 0, 0)]
        self.padding = 'kernel_size-1 /2'  # or 'kernel_size/2 + 1'
        self.in_channels = [1]
        for nch in self.out_channels[:-1]:
            self.in_channels.append(nch)
        for i in range(self.n_conv):
            for d in range(3):
                if self.pooling[i][d] != 0:
                    self.input_dim[d] = compute_output_size(self.input_dim[d], self.pooling[i][d], 0, self.pooling[i][d])
        out = self.input_dim[0] * self.input_dim[1] * self.input_dim[2]
        self.fweights = [self.out_channels[-1] * out, 2]
        self.dropout = 0.0


param = NetParameter()
config = Configuration()
data_info = DataParameter()
tonpy = lambda x: x.detach().cpu().numpy()
totorch = lambda x: torch.from_numpy(x).to(config.device)
torchtensor = lambda x: torch.from_numpy(x)

saver_path = str(sys.argv[6]) + 'Exp' + str(config.nexp) + '/'

if not os.path.exists(saver_path):
    os.makedirs(saver_path)

logfile = open(saver_path + 'logfile.txt', 'w')

logfile.write('ADNI DATASET\n')
logfile.write('Binary Classification: CN vs AD\n')

logfile.write('Exp {}\n'.format(config.nexp))


logfile.write('\nNetwork parameters\n')
for (k, v) in param.__dict__.items():
    if 'input_dim' not in k:
        logfile.write(k + ': ' + str(v) + '\n')


logfile.write('\nTraining parameters\n')
logfile.write('LR = {}\nDecay = {}\n'.format(config.lr, config.lr_decay))
logfile.write('L2 reg. = {}\n'.format(config.l2_reg))
logfile.write('Max Epochs = {}\nMax Errors = {}\n'.format(config.num_epochs, config.maxerror))
logfile.write('Batch size = {}\n\n'.format(config.batch_size))
logfile.flush()

# ##########################
#  DATA AND MODEL LOADING  #
# ##########################

logfile.write('\nData processing:\n')
logfile.write('Rescaling (0.5) the and resize the first two dimensions: 96 x 96\n')
logfile.write('Rescaling (0.5) of the last dimension and dimensionality reduction to 73\n')
logfile.write('Intensity normalization of the resulting image\n\n')

shuffle_data = False
logfile.write('\nShuffle data:' + str(shuffle_data) + '\n')

folds = create_folds(data_path, augmentation_list=data_info.data_augmentation_type, shuffle_data=shuffle_data,
                     ndatasets=data_info.ndatasets)
k = len(folds)
ntr, nv, nts = [len(folds[0][i]) for i in range(len(folds[0]))]
lntr, lnv, lnts = [len(folds[-1][i]) for i in range(len(folds[-1]))]
lnts_cn = sum([1 for f in folds[-1][-1] if '/0_ADNI' in f])
lnts_ad = sum([1 for f in folds[-1][-1] if '/2_ADNI' in f])

print('')
print('N folders:', k)
print('The first', k - 1, 'contains:')
print('{} Train samples - {} Val samples - {} Test samples'.format(ntr, nv, nts))
print('Testing set is balanced')
print('The last fold (Fold ', k - 1, ') contains:')
print('{} Train samples - {} Val samples - {} Test samples'.format(lntr, lnv, lnts))
print('Testing set is unbalanced: {} CN and {} AD'.format(lnts_cn, lnts_ad))


logfile.write('\nNumber of folders: {}\n'.format(k))
logfile.write('The first {} contains:\n'.format(k-1))
logfile.write('{} Train samples - {} Val samples - {} Test samples\n'.format(ntr, nv, nts))
logfile.write('Testing set is balanced\n')
logfile.write('The last fold (Fold {}) contains:\n'.format(k-1))
logfile.write('{} Train samples - {} Val samples - {} Test samples\n'.format(lntr, lnv, lnts))
logfile.write('Testing set is unbalanced: {} CN and {} AD\n'.format(lnts_cn, lnts_ad))
logfile.flush()

net = CNN(param).to(config.device)
w_init = dcopy(net.state_dict())
torch.save(w_init, saver_path + 'cnn_weights_init.pt')

##
best_val_acc, test_acc = [], []

for i in range(len(folds)):
    print('')
    print('Fold', i, ': Training phase')
    config.saver_path = saver_path + 'fold' + str(i) + '_'
    fold_logfile = open(saver_path + 'logfile_fold' + str(i) + '.txt', 'w')
    fold_logfile.write('Fold ' + str(i) + '\n\n')

    # #####################
    #   DATA SPLIT        #
    # #####################

    train, val, test = folds[i]
    train_data = loader(train, transform=torch_norm)
    val_data = loader(val, transform=torch_norm)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size,  shuffle=False, pin_memory=True)

    # #####################
    #   MODEL TRAINING    #
    # #####################

    net.load_state_dict(w_init)   # for each split, the model keeps the same weight initialization
    if config.return_train_acc and config.return_train_prob:
        loss, dev_acc, train_acc, train_prob, best_epoch = train_model(net, config, train_loader, val_loader,
                                                                       train_acc=True, return_prob=True,
                                                                       logfile=fold_logfile)
        np.save(saver_path + 'Fold' + str(i) + '_training_info.npy', {'Loss': loss, 'Val_accuracy': dev_acc,
                                                                      'Train_accuracy': train_acc, 'Train_prob': train_prob})
    else:
        loss, dev_acc, best_epoch = train_model(net, config, train_loader, val_loader,
                                                train_acc=False, return_prob=False, logfile=fold_logfile)
        np.save(saver_path + 'Fold' + str(i) + '_training_info.npy', {'Loss': loss, 'Val_accuracy': dev_acc})
    best_val_acc.append(max(dev_acc))
    fold_logfile.write('\nBest validation accuracy at epoch {}\n\n'.format(best_epoch))
    fold_logfile.flush()

    # #################
    #   TESTING       #
    # #################

    w = torch.load(saver_path + 'fold' + str(i) + '_cnn_best_weights.pt')
    net.load_state_dict(w)
    test_data = loader(test, transform=torch_norm)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=False, pin_memory=True)
    test_accuracy = test_model(net, config, test_loader).item()
    test_acc.append(test_accuracy)


    print('Fold', i, '- Test accuracy', test_accuracy)
    fold_logfile.write('\n\nTesting accuracy {:.4f}'.format(test_accuracy))
    fold_logfile.flush()
    fold_logfile.close()
    logfile.write('\nFold {}\n'.format(i))
    logfile.write('Best Validation accuracy: {}\n'.format(best_val_acc[i]))
    logfile.write('Testing accuracy: {}\n'.format(test_acc[i]))
    logfile.flush()

n_test = [nts]*(k-1) + [lnts]
w_test_acc = [test_acc[i]*n_test[i] for i in range(len(test_acc))]
average_test_acc = sum(w_test_acc)/sum(n_test)
logfile.write('\nAverage values:\n')
logfile.write('Best Validation accuracy: {}\n'.format(np.mean(best_val_acc)))
logfile.write('Weighted Testing accuracy: {}\n'.format(average_test_acc))
logfile.flush()
logfile.close()

print('Results on average:')
print('Validation accuracy:', np.mean(best_val_acc))
print('Weighted Testing accuracy:', average_test_acc)

np.save(saver_path + 'Best_val_and_test_accuracy.npy', {'Val_accuracy': best_val_acc, 'Test_accuracy': test_acc})