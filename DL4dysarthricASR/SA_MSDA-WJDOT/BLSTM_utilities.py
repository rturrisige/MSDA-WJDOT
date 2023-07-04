import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
td = lambda x: x.to('device')


#################################
#                              ##
#         Loader               ##
#                              ##
#################################

def batch_sorting(inputs, labels, input_len, label_len):
    sorted_idx = sorted(range(len(input_len)), key=lambda k: input_len[k], reverse=True)
    input_len = input_len[sorted_idx]
    label_len = label_len[sorted_idx]
    labels = labels[sorted_idx, :]
    inputs = inputs[sorted_idx, :, :]
    return inputs, labels, input_len, label_len


def my_collate(batch):
    input_features = [batch[i][0] for i in range(len(batch))]
    target = [batch[i][1] for i in range(len(batch))]
    seq_len = [batch[i][2] for i in range(len(batch))]
    label_len = [batch[i][3] for i in range(len(batch))]
    max_seq_len, max_lab_len = max(seq_len), max(label_len)
    for i, seq in enumerate(input_features):
       input_features[i] = torch.from_numpy(input_features[i])
       target[i] = torch.from_numpy(target[i])
       if len(batch)!=1:
            input_features[i] = F.pad(input_features[i], pad=(0, 0, 0, max_seq_len - seq_len[i]), mode='constant', value=0)
            target[i] = F.pad(target[i], pad=(0, max_lab_len - label_len[i]), mode='constant', value=0)
    batch = [torch.stack(input_features, 0), torch.stack(target,0), torch.from_numpy(np.asarray(seq_len)),
            torch.from_numpy(np.asarray(label_len))]
    return batch


##################
# Inference     ##
# ################


def compute_ERR(output, labels):
    prediction = torch.max(output, 1)[1]
    acc_test = torch.sum(prediction == labels.long()).to(dtype=torch.float) * 1.0 / labels.shape[0]
    return 1 - acc_test.item()

def compute_acc(output, labels):
    """ Returns accuracy

        Parameters
        ----------
        output: torch tensor
        labels: torch tensor
    """
    test_output = torch.max(output, 1)[1]
    acc_test = torch.sum(test_output == labels.long()).to(dtype=torch.float)
    return acc_test.item()


def inference(model, data_loader):
    """ Returns accuracy

        Parameters
        ----------
        model: class
            trained model
        data_loader: class
            data generator (input, label, input_length, label_length)
    """
    model = model.eval()
    acc = 0.0
    n = 0
    for _, (inputs, labels, input_len, label_len) in enumerate(data_loader):
        x, y, x_len, y_len = batch_sorting(inputs, labels, input_len, label_len)
        x, y, x_len = td(x.permute(1, 0, 2)), td(y), td(x_len)
        outputs = model(x, x_len)
        acc += torch.sum(torch.max(outputs, 1)[1] == y.long()).to(dtype=torch.float)
        n += y.shape[0]
    return acc / n


def mtl_inference(model, data_loader):
    """ Returns the list of accuracy computed for each task

        Parameters
        ----------
        model: class
            trained model
        data_loader: list
            list of data loader
    """
    model = model.eval()
    t = 0
    acc = []
    for task_loader in data_loader:
        task_acc, n = 0.0, 0
        for _, (inputs, labels, input_len, label_len) in enumerate(task_loader):
            x, y, x_len, y_len = batch_sorting(inputs, labels, input_len, label_len)
            x, y, x_len = td(x.permute(1, 0, 2)), td(y), td(x_len)
            outputs = model(x, x_len, t)
            task_acc += torch.sum(torch.max(outputs, 1)[1] == y.long()).to(dtype=torch.float)
            n += y.shape[0]
        t += 1
        acc.append(task_acc / n)
    return acc


def compute_embedding(model, config, data_loader):
    """ Returns the list of accuracy computed for each task

        Parameters
        ----------
        model: class
            trained model
        data_loader: class
            data generator (input, label, input_length, label_length)

        Returns
        ----------
        embeddings: torch tensor
            extracted embeddings
        labels: torch tensor
            corresponding labels
    """
    model.eval()
    all_embeddings, all_labels = [], []
    for i, (inputs, labels, input_len, label_len) in enumerate(data_loader):
        inputs, labels, input_len, label_len = batch_sorting(inputs, labels, input_len, label_len)
        all_labels.append(labels.to(config.device))
        inputs = inputs.permute(1, 0, 2).float().to(config.device)
        batch_size = inputs.shape[1]
        nhidden = model.init_hidden(batch_size)
        input_seq = torch.nn.utils.rnn.pack_padded_sequence(inputs, input_len)
        lstm_output, nhidden = model.bi_lstm(input_seq.float(), nhidden)
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output)  #
        last_output = torch.cat([lstm_output[input_len[j]-1, j, :].view(-1, config.n_hidden * 2) for j in range(batch_size)], 0)
        all_embeddings.append(last_output)
    embeddings = torch.cat(all_embeddings, 0)
    labels = torch.cat(all_labels, 0)
    return embeddings, labels


# ################
#  Training     ##
# ################

def train(model, data_loader, opt, lr_decay):
    """ Performs training

        Parameters
        ----------
        model: class
            trained model
        data_loader: class
            data generator (input, label, input_length, label_length)
        opt:
            optimizer
        lr_decay:
            learning rate decay

        Returns
        ----------
        epoch_loss: float
            epoch loss
        epoch_acc: float
            epoch accuracy
        step: int
            number of training steps
    """
    epoch_loss, epoch_acc = 0.0, 0.0
    n = 0
    for step, (inputs, labels, input_len, label_len) in enumerate(data_loader):
        n += inputs.shape[0]
        x, y, x_len, y_len = batch_sorting(inputs, labels, input_len, label_len)
        x, y, x_len = td(x.permute(1, 0, 2)), td(y), td(x_len)
        model = model.train()
        outputs = model(x, x_len) 
        batch_loss = model.criterion(outputs, y)
        epoch_loss += batch_loss.item()
        opt.zero_grad()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        batch_loss.backward()
        opt.step()
        lr_decay.step()
        model = model.eval()
        epoch_acc += compute_acc(model(x, x_len), y)
    epoch_loss /= (step + 1)
    epoch_acc /= n
    return epoch_loss, epoch_acc, step


def mtl_train(model, data_loader, opt, lr_decay):
    """ Performs multi-task learning

        Parameters
        ----------
        model: class
            trained model
        data_loader: class
            data generator (input, label, input_length, label_length)
        opt:
            optimizer
        lr_decay:
            learning rate decay

        Returns
        ----------
        epoch_loss: float
            average of the epoch losses (over tasks)
        epoch_acc: float
            epoch accuracy of the epoch losses (over tasks)
        step: int
            number of training steps
    """
    epoch_loss = []
    epoch_acc = []
    t = 0
    for task_loader in data_loader:
        task_loss, task_acc = 0.0, 0.0
        n = 0
        for task_step, (inputs, labels, input_len, label_len) in enumerate(task_loader):
            n += inputs.shape[0]
            x, y, x_len, y_len = batch_sorting(inputs, labels, input_len, label_len)
            x, y, x_len = td(x.permute(1, 0, 2)), td(y), td(x_len)
            model = model.train()
            outputs = model(x, x_len, t) 
            batch_loss = model.criterion(outputs, y)
            task_loss += batch_loss.item()
            opt.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            batch_loss.backward()
            opt.step()
            lr_decay.step()
            model = model.eval()
            task_acc += compute_acc(model(x, x_len, t), y)
        task_loss /= (task_step + 1)
        task_acc /= n
        epoch_loss.append(task_loss)
        epoch_acc.append(task_acc)
        t += 1
    return np.mean(epoch_loss), np.mean(epoch_acc)


#################################
#                              ##
#       Make a plot            ##
#                              ##
#################################

def plotting(graphs, saver_name, labels=['Loss', 'Val CER']):
    colors = ['g', 'r', 'b', 'c', 'm', 'y']
    plt.figure()
    plt.suptitle('Network training')
    for i in range(len(graphs)):
        plt.subplot(1, len(graphs), i + 1)
        f = np.array(graphs[i])
        plt.plot(f, color=colors[i], label=labels[i]) 
        plt.xlabel('Epochs')
        plt.legend()
    plt.savefig(saver_name)
    plt.tight_layout()
    print('Figure saved')
   
   
