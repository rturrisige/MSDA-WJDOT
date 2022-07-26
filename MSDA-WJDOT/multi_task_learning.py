import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy as dcopy
from torch.utils.data import Dataset

# #####################
# Multi-task learning #
#######################


class MTL_Configuration(object):
    """
    This class contains the mtl network pararmeters.

    Arguments
    ---------
    S: int
            number of sources 
    n_layers: int > 0
            number of network layers
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
    criterion: torch function
            loss function to minimize
    l2_reg: float
            weigt of the l2 regularization term
    nT: int > 0
            number of tasks
    batch_size: int > 0
            batch size of each task dataset. At each step, the total batch size is batch_size*nT
    """
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.n_layers = 2
        self.hid_dim = 50
        self.embedding_dim = self.hid_dim * 2
        self.input_dim = 13
        self.num_classes = 2
        self.num_epochs = 500
        self.maxerror = 50
        self.lr = 0.01
        self.l2_reg = 0.0 
        self.lr_decay = 1.0
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 5
        self.nT = 11


class MTL_loader(Dataset):
    def __init__(self, dataset_list):
        """ Given a list of datasets, it returns a list of batches """
        self.dataset_list = dataset_list
    def __len__(self):
        return self.dataset_list[0][0].shape[0]
    def __getitem__(self, index):
        x_batch, y_batch = [], []
        nsets = len(self.dataset_list[0])
        for t in range(nsets):
            task_x_batch = self.dataset_list[0][t][index]
            task_y_batch = self.dataset_list[1][t][index]
            x_batch.append(task_x_batch)
            y_batch.append(task_y_batch)
        return x_batch, y_batch



class MTL_BiRNN(nn.Module):
    def __init__(self, config):
        super(MTL_BiRNN, self).__init__()
        self.hidden_size = config.hid_dim
        self.device = config.device
        self.num_layers = config.n_layers
        self.lstm = nn.LSTM(config.input_dim, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.ft = nn.ModuleList()
        for t in range(config.nT):
            self.ft.append(nn.Linear(self.hidden_size * 2, config.num_classes))
    def forward(self, x, num_task):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        c = out[:, -1, :]
        # Decode the hidden state of the last time step
        out = self.ft[num_task](c)
        return torch.sigmoid(out)


def mtl_embedding(mtl_net, x):
    """ Extracts the embedding features """
    return mtl_net.lstm(x)[0][:, -1, :].detach()

def eval_mtl_model(net, n_task, x, y): 
    """ Returns the accuracy """
    prediction = torch.max(net(x, num_task=n_task), 1)[1]
    accuracy = torch.sum(prediction == y.long()).to(dtype=torch.float) * 1.0 / y.shape[0]
    return accuracy.item()


def train_mtl_model(net, config, train_loader, val_datasets):
    """
    It trains the MTL network and returns the loss and the accuracy.

    Arguments
    ---------
    net: torch class
          network to train
    config: class
          it contains the network parameters
    train_loader: DataLoader of torch
          loader of source training sets
    val_datasets: list of torch array
          it is a list of source validation sets

    Returns
    -------
    epochs_loss: [float, ..., float]
          list of epoch losses
    epochs_val_acc: [float, ..., float]
          list of epoch validation accuracy
    """  
    epochs_loss, epochs_val_acc = [], []
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    updated_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.lr_decay)
    nerrors = 0
    start_accuracy = 0.0
    total_step = len(train_loader)
    print('Start MTL training')
    for epoch_counter in range(config.num_epochs):
        epoch_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            loss = 0.0
            acc = 0.0
            for t in range(len(val_datasets[0])):
                task_batch_x, task_batch_y = batch_x[t].to(config.device), batch_y[t].to(config.device)
                output = net(task_batch_x, num_task=t)
                task_loss = config.criterion(output, task_batch_y.long())
                loss += task_loss
            optimizer.zero_grad()
            loss.backward()
            if step == 0 and epoch_counter != 0:
                if current_lr > 0.00001:
                    updated_lr.step()
            optimizer.step()
            epoch_loss += loss.item() / config.nT
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        tasks_accuracy = []
        for t in range(len(val_datasets[0])):
            DevData, DevLabels = val_datasets[0][t].to(config.device), val_datasets[1][t].to(config.device)
            ACC_dev = eval_mtl_model(net, t, DevData, DevLabels)
            tasks_accuracy.append(ACC_dev)
        mean_accuracy = np.mean(np.array(tasks_accuracy))
        print('Epoch {} LR {} - Loss {}  - Validation Accuracy {}'.format(epoch_counter, current_lr,
                                                                             epoch_loss / total_step,
                                                                             mean_accuracy))
        epochs_loss.append(epoch_loss / total_step)
        epochs_val_acc.append(mean_accuracy)
        # Early stopping
        if mean_accuracy < start_accuracy:
            nerrors += 1
            if nerrors > config.maxerror:
                print('Early stopping applied at iteration', epoch_counter, '. Dev accuracy=', start_accuracy)
                net.load_state_dict(net_weights)
                break
        else:
            net_weights = dcopy(net.state_dict())
            start_accuracy = mean_accuracy
            nerrors = 0
    net.load_state_dict(net_weights)
    return epochs_loss, epochs_val_acc



def run_mtl(config, sources_train_sets, sources_val_sets):
    """
    Returns the trained network

    Arguments
    ---------
    config: class
          it contains the network parameters
    sources_train_setes: list of torch array
          it is a list of source training datasets
    sources_val_sets: list of torch array
          it is a list of source validation datasets

    Returns
    -------
    mtl_net: trained network
    """  
    mtl_net = MTL_BiRNN(config).to(config.device)
    mtl_train_set = MTL_loader(sources_train_sets)
    mtl_train_loader = torch.utils.data.DataLoader(mtl_train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    mtl_loss, mtl_val_acc = train_mtl_model(mtl_net, config, mtl_train_loader, sources_val_sets)
    return mtl_net

