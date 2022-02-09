from torchvision import transforms
import numpy as np
from glob import glob as gg
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from copy import deepcopy as dcopy
from math import sqrt
import torch.nn.functional as F
from torch.utils.data import Dataset
tonpy = lambda x: x.detach().cpu().numpy()
import time
from sklearn.utils import shuffle
import os

##
# ######################
#     DATA LOADING    ##
# ######################


def normalize_intensity(img_tensor, normalization="mean"):
   """
   Accepts an image tensor and normalizes it
   :param normalization: choices = "max", "mean" , type=str
   For mean normalization we use the non zero voxels only.
   """
   if normalization == "mean":
       mask = img_tensor.ne(0.0)
       desired = img_tensor[mask]
       mean_val, std_val = desired.mean(), desired.std()
       img_tensor = (img_tensor - mean_val) / std_val
   elif normalization == "max":
     MAX, MIN = img_tensor.max(), img_tensor.min()
     img_tensor = (img_tensor - MIN) / (MAX - MIN)
   return img_tensor


def torch_norm(input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    input_tensor = normalize_intensity(input_tensor)
    return input_tensor.unsqueeze_(0)


class loader(Dataset):
    def __init__(self, dataset, nclasses=2, transform=None):
        """
        dataset : list of all filenames
        nclasses : number of classes
        transform (callable) : a function/transform that acts on the images (e.g., normalization).
        """
        self.dataset = dataset
        self.transform = transform
        self.num_classes = nclasses

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x_batch, y_batch = np.load(self.dataset[index], allow_pickle=True)
        if self.transform:
            x_batch = self.transform(x_batch)
        if self.num_classes == 2:
            y_batch = np.where(y_batch == 2, 1, y_batch)
        return x_batch, y_batch


##
# ###################
# K FOLDS FUNCTIONS #
# ###################


def get_data(data_path, shuffle_data=True):
    """
    :param
    data_path: path to the folder "processed_data/" containing numpy files, type=str
    shuffle_data: if True the dataset is randomly shuffled, type=Boolean
    """
    if shuffle_data:
        all_cn = shuffle(gg(data_path + 'processed_data/0*.npy'))
        all_ad = shuffle(gg(data_path + 'processed_data/2*.npy'))
    else:
        all_cn = gg(data_path + 'processed_data/0*.npy')
        all_ad = gg(data_path + 'processed_data/2*.npy')
    return all_cn, all_ad


def add_augmentation(data_path, train_files, augm_list, ndatasets=1):
    """
    It returns a list of augmented-data files corresponding to the training samples
    :param
    data_path: path to folder "augmentation/" containing augmentation-data files, type=str
    train files: list of files name in the training dataset, type=list of str
    augm_list: list of transformations to use for data augmentation, type=list of str
    ndataset: number of augmented dataset, type=int
     """
    augmented_data = []
    for augmentation in augm_list:
        augmented_data += [data_path + 'augmentation/' + augmentation + '/' + os.path.basename(f) for f in train_files]
        if ndatasets > 1:
            for n in range(2, ndatasets + 1):
                augmented_data += [data_path + 'augmentation/' + augmentation + str(n) + '/' + os.path.basename(f) for f in
                                    train_files]
    return augmented_data


def create_folds(data_path, ntest=80, nval=80, augmentation_list=[], ndatasets=1, shuffle_data=False):
    """
    It returns a list of k folds. Each fold is a list containing training, validation and testing data.
    :param
    data_path: is the path folder containing numpy files, type=str
    ntest: is the number of samples in the testing set, type=int, default=80
    nval: is the number of samples in the validation test, type=int, default=80
    augmentation_list: if not empty, it contains the names of the affine transformation used to augment data, type=list,
    default=[] (no-augmented data)
    ndataset: number of augmented dataset, type=int, default=1
    shuffle_data: if True the dataset is randomly shuffled, type=Boolean, default=False
    """

    all_data = get_data(data_path, shuffle_data=shuffle_data)
    nt, nv = int(ntest/2), int(nval/2)
    k = int(min([len(d) for d in all_data])/nt)
    folds = []
    for i in range(k):
        test, val, train = [], [], []
        for all_files in all_data:
            test += all_files[nt*i: nt*(i+1)]
            if i == k-1:
                val += all_files[:nv]
            else:
                val += all_files[nt*(i+1): nt*(i+1) + nv]
            train += [f for f in all_files if f not in test+val]
        if augmentation_list:
            train += add_augmentation(data_path, train, augmentation_list, ndatasets)
        folds.append([train, val, test])
    test = all_data[0][k*nt:] + all_data[1][k*nt:]
    val = all_data[0][nv:nv*2] + all_data[1][nv:nv*2]
    train = [f for f in all_data[0] if f not in test + val] + [f for f in all_data[1] if f not in test + val]
    if augmentation_list:
        train += add_augmentation(data_path, train, augmentation_list, ndatasets)
    folds.append([train, val, test])
    return folds


##
# ###################
#     TRAIN & EVAL  #
# ###################


def test_model(net, config, data_loader):
    """
    It returns the average accuracy
    :param
    net: model network
    config (class of parameters) with config.device referring to the device (cpu/gpu) that must be used
    data_loader: loader of the dataset
     """
    net = net.eval()
    tot_acc = 0.0
    N = 0
    for _, (x, y) in enumerate(data_loader):
        x, y = x.to(config.device), y.to(config.device)
        acc = torch.sum(torch.max(net(x), 1)[1] == y.long()).to(dtype=torch.float)
        tot_acc += acc
        N += x.shape[0]
    return tot_acc / N


def train_model(net, config, train_loader, val_loader, train_acc=False, return_prob=False, logfile=False):
    """
    It returns a list containing the loss function, the validation accuracy, the train accuracy (if train_acc=True),
    the probability of the training otuput (if return_prob=True), and the epoch in which early stopping is performed.
    :param
    net: model network
    config: class of parameters
    training_loader: loader of the training dataset
    val_loader: loader of the validation dataset
    train_acc: True or False
    return_prob: Ture or False
    logfile: (default, False) is a txt file where training information are reported at each epoch.
     """
    # PARAMETER DEFINITION:
    epochs_loss, epochs_dev_acc, epochs_train_acc = [], [], []
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.l2_reg)
    updated_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.lr_decay)
    nerrors, best_accuracy, all_probs = 0, 0.0, []
    total_step = len(train_loader)
    print('Number of total step per epoch:', total_step)
    # STARTING TRAINING:
    for epoch_counter in range(config.num_epochs):
        start = time.time()
        epoch_loss = 0.0
        epoch_probs = np.array([])
        # EPOCH TRAINING:
        for step, (batch_x, batch_y) in enumerate(train_loader):
            net = net.train()
            batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)
            prob = net(batch_x)
            loss = config.criterion(prob, batch_y.long())
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            if step == 0 and epoch_counter != 0:
                if current_lr > 0.00001:
                    updated_lr.step()
            optimizer.step()
            if step % 10 == 0:
                print('Epoch {} step {} - Loss {:.4f}'.format(epoch_counter, step, loss.item()))
            if return_prob:
                prob = tonpy(prob)
                if epoch_probs.size:
                    epoch_probs = np.concatenate((epoch_probs, prob), 0)
                else:
                    epoch_probs = prob
        end_training = time.time()
        if return_prob:
            all_probs.append(epoch_probs)
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        mean_accuracy = test_model(net, config, val_loader).item()
        val_time = time.time()
        # COMPUTE TRAINING ACCURACY:
        if train_acc:
            train_accuracy = test_model(net, config, train_loader).item()
            train_time = time.time()
            epochs_train_acc.append(train_accuracy)
            print('Epoch {} LR {:.8f} - Loss {:.4f} - Train Acc {:.4f} '
                  '- Val Acc {:.4f}'.format(epoch_counter, current_lr, epoch_loss / total_step,
                                            train_accuracy, mean_accuracy))
            if logfile:
                logfile.write('Epoch {} LR {:.7f} - Loss {} - Train Acc {:.4f}'
                              '- Val Acc {}\n'.format(epoch_counter, config.lr, epoch_loss / total_step,
                                                      train_accuracy, mean_accuracy))
                logfile.flush()
        else:
            print('Epoch {} LR {:.8f} - Loss {:.4f} - Val Acc {:.4f}'.format(epoch_counter, current_lr,
                                                                             epoch_loss / total_step,
                                                                             mean_accuracy))
            if logfile:
                logfile.write('Epoch {} LR {:.8f} - Loss {} - Val Acc {}\n'.format(epoch_counter, config.lr,
                                                                                   epoch_loss / total_step,
                                                                                   mean_accuracy))
                logfile.flush()
        print('')
        epochs_loss.append(epoch_loss / total_step)
        epochs_dev_acc.append(mean_accuracy)
        # COMPUTE VAL ACCURACY AND APPLY EARLY STOPPING
        if (mean_accuracy < best_accuracy) or (mean_accuracy == best_accuracy):
            nerrors += 1
            if nerrors > config.maxerror:
                print('Early stopping applied at iteration', epoch_counter, '. Dev accuracy=', best_accuracy)
                if logfile:
                    logfile.write('Early stopping applied at epoch {}. Val. accuracy {:.4f}'.format(epoch_counter,
                                                                                                    best_accuracy))
                    logfile.flush()
                net.load_state_dict(net_weights)
                break
        else:
            print('Saved weights at epoch', epoch_counter)
            torch.save(dcopy(net.state_dict()), config.saver_path + 'cnn_best_weights.pt')
            print('At ', config.saver_path + 'cnn_best_weights.pt')
            print('f1 bias',net.state_dict()['f.0.bias'])
            net_weights = dcopy(net.state_dict())
            best_epoch = dcopy(epoch_counter)
            best_accuracy = mean_accuracy
            nerrors = 0
        end_epoch = time.time()
        if epoch_counter == 0:
            print('Training time:', end_training - start)
            print('Validation accuracy time:', val_time - end_training)
            if train_acc:
                print('Training accuracy time:', train_time - val_time)
            print('Total epoch time:', end_epoch - start)
    net.load_state_dict(net_weights)
    to_return = [epochs_loss, epochs_dev_acc]
    if train_acc:
        to_return.append(epochs_train_acc)
    if return_prob:
        to_return.append(all_probs)
    to_return.append(best_epoch)
    return to_return



##
# ##################
#    CNN MODEL     #
# ##################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_output_size(i, K, P, S):
    output_size = ((i - K + 2*P)/S) + 1
    return int(output_size)


class CNN(nn.Module):
    def __init__(self, param):
        super(CNN, self).__init__()
        self.embedding = nn.ModuleList()
        for i in range(param.n_conv):
            if param.padding == 'kernel_size-1 /2':
                pad = tuple([int((k-1)/2) for k in param.kernels[i]])
            elif param.padding == 'kernel_size/2 + 1':
                pad = tuple([int(k/2 + 1) for k in param.kernels[i]])
            if param.pooling[i] != (0, 0, 0):
                self.embedding.append(nn.Sequential(
                        nn.Conv3d(in_channels=param.in_channels[i], out_channels=param.out_channels[i],
                                  kernel_size=param.kernels[i], stride=(1, 1, 1), padding=pad, bias=False),
                        nn.BatchNorm3d(param.out_channels[i]),
                        nn.ReLU(inplace=True),
                        nn.MaxPool3d(param.pooling[i], stride=param.pooling[i])))
            else:
                self.embedding.append(nn.Sequential(
                    nn.Conv3d(in_channels=param.in_channels[i], out_channels=param.out_channels[i],
                              kernel_size=param.kernels[i], stride=(1, 1, 1), padding=pad, bias=False),
                    nn.BatchNorm3d(param.out_channels[i]),
                    nn.ReLU(inplace=True)))
            self.embedding = nn.ModuleList(self.embedding)
        self.ReLU = nn.ReLU(inplace=True)
        self.Dropout = nn.Dropout(p=param.dropout)
        self.f = nn.ModuleList()
        for i in range(len(param.fweights)-1):
            self.f.append(nn.Linear(param.fweights[i], param.fweights[i+1]))
    def forward(self, x, return_conv=False):
        out = self.embedding[0](x)
        if return_conv:
            all_layers = [out]
        for i in range(1, len(self.embedding)):
            out = self.embedding[i](out)
            if return_conv:
                all_layers.append(out)
        out = out.view(out.size(0), -1)
        # out = self.Dropout(out)
        for fc in self.f[:-1]:
            out = fc(out)
            out = self.ReLU(out)
            out = self.Dropout(out)
        out = self.f[-1](out)
        if return_conv:
            return F.softmax(out, dim=1), all_layers
        else:
            return F.softmax(out, dim=1)
