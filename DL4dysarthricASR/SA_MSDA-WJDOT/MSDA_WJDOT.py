import ot
from copy import deepcopy as dcopy
import torch
import torch.nn as nn
import numpy as np
from ot_torch import cost_emd_jdot, cost_bures_jdot, proj_simplex

tonpy = lambda x: x.detach().cpu().numpy()


# ################
# Utilities     ##
# ################


class ClassifierLayer(nn.Module):
    """ One layer network """

    def __init__(self, config):
        super(ClassifierLayer, self).__init__()
        self.l1 = nn.Linear(config.embedding_dim, config.num_classes)

    def forward(self, x):
        y_pred = torch.sigmoid(self.l1(x))
        return y_pred


def get_xy_matrix(data, target_reg=1, xy_list=False):
    """ It returns the concatenation of input and labels """
    if xy_list:
        xy_all = []
        for d in data:
            xy_all.append(torch.cat((d[0], d[1]), 1))
    else:
        x_all = torch.cat([d[0] for d in data], 0)  # all training input
        y_all = torch.cat([d[1] for d in data], 0)  # all training output
        xy_all = torch.cat((x_all, target_reg * y_all), 1)  # all training data
    return xy_all


def get_sources_histogram(N_samples, N_sources, device='cpu'):
    """ It returns the sources histogram """
    histogram = torch.zeros((sum(N_samples), N_sources), device=device)
    N = 0
    for s in range(N_sources):
        histogram[N: N + N_samples[s], s] = 1 / N_samples[s]
        N += N_samples[s]
    return histogram


def get_onehot_label(y, num_classes):
    """ It converts the label vector in one hot vector """
    onehot = torch.zeros([y.shape[0], num_classes])
    j = 0
    for row in y:
        onehot[j, row] = 1
        j += 1
    return onehot.float()


def compute_sse(net, config, xtest):
    """
    It compute the sum of the squared errors between
    the estimated outputs $f(X)$ and their estimated cluster centroids.
    """
    _, predtest = torch.max(net(xtest), 1)
    d = 0
    for i in range(config.num_classes):
        index = torch.where(predtest == i)[0]
        cluster = xtest[index]
        centroid = torch.mean(cluster, 0)
        for nrow in range(cluster.shape[0]):
            d += torch.dist(cluster[nrow], centroid)
    return d.item()


def inference(net, x, y):
    """ It computes the accuracy """
    prediction = torch.max(net(x), 1)[1]
    acc = torch.sum(prediction == y.long()).to(dtype=torch.float) * 1.0 / y.shape[0]
    return acc.item()


# ##################
# Wjdot algorithms #
# ##################


def wjdot(net, config, source_xy, Ns_list, x, cost='emd'):
    """"
    WJDOT algorithm

    Parameters
    ----------
    net: class
        neural network
    source_xy: torch array, [sources samples, embedding dimension + n classes]
        concatenated source datasets
    Ns_list: list of float
        list of number of source samples
    x: torch array, [target samples, embedding dimension]
        target input
    cost: 'emd' or 'bures'
        wasserstein cost function

    Returns
    -------
    epochs_alphas: list of array with lenght
        alpha weights at each epoch
    epochs_cost: list of float
        loss function at each epoch
    """

    alpha = (torch.ones(config.S, device=config.device) / config.S).requires_grad_()  # alpha initialization
    Tn = get_sources_histogram(Ns_list, config.S, config.device)  # sources histogram
    wt = torch.ones(x.shape[0], device=config.device) * 1.0 / x.shape[0]  # target histogram

    epochs_cost, epochs_alphas = [], []

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.l2_reg)
    updated_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.lr_decay)

    for epoch_counter in range(1, config.num_epochs + 1):
        output = net(x)
        if 'bures' in str(cost):
            loss = cost_bures_jdot(alpha, source_xy, x, output, Tn, wt, target_reg=config.beta,
                                   bures_reg=config.bures_reg)
        else:
            loss = cost_emd_jdot(alpha, source_xy, x, output, Tn, wt, target_reg=config.beta)
        optimizer.zero_grad()
        loss.backward()
        if epoch_counter != 1:
            if current_lr > 0.0001:
                updated_lr.step()

        optimizer.step()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        dalpha = alpha.grad
        with torch.no_grad():
            dalpha -= dalpha.sum(0)
            alpha -= current_lr * dalpha
            alpha[:] = proj_simplex(alpha)
        alpha.grad.zero_()

        epochs_alphas.append(tonpy(alpha))
        epochs_cost.append(loss.item())

        print('Epoch {} - LR {} - Loss {:.6f}'.format(epoch_counter, current_lr, epochs_cost[-1]))
    return epochs_alphas, epochs_cost


def wjdot_acc(net, config, source_xy, Ns_list, sources_val_data, x, cost='emd'):
    """"
    WJDOT algorithm with early stopping based on the weighted accuracy of the source datasets

    Parameters
    ----------
    net: class
        neural network
    source_xy: torch array, [sources samples, embedding dimension + n classes]
        concatenated source datasets
    Ns_list: list of float
        list of number of source samples
    sources_val_data: list of torch array
        list of source validation sets
    x: torch array, [target samples, embedding dimension]
        target input
    cost: 'emd' or 'bures'
        wasserstein cost function

    Returns
    -------
    epochs_alphas: list of array with lenght
        recovered alpha weights at each epoch
    epochs_cost: list of float
        loss function at each epoch
   epochs_sources_accuracy: list of float
        weighted accuracy of the source datasets at each epoch
    """
    net_weights = dcopy(net.state_dict())
    current_lr = config.lr
    sources_x_val, sources_y_val = sources_val_data

    alpha = (torch.ones(config.S, device=config.device) / config.S).requires_grad_()  # alpha initialization
    Tn = get_sources_histogram(Ns_list, config.S, config.device)  # sources histogram
    wt = torch.ones(x.shape[0], device=config.device) * 1.0 / x.shape[0]  # target histogram

    epochs_cost, epochs_sources_accuracy, epochs_alphas = [], [], []
    nerrors = 0
    labelled_acc = 0

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.l2_reg)
    updated_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.lr_decay)

    for epoch_counter in range(1, config.num_epochs + 1):
        output = net(x)
        if 'bures' in str(cost):
            loss = cost_bures_jdot(alpha, source_xy, x, output, Tn, wt, target_reg=config.beta,
                                   bures_reg=config.bures_reg)
        else:
            loss = cost_emd_jdot(alpha, source_xy, x, output, Tn, wt, target_reg=config.beta)
        optimizer.zero_grad()
        loss.backward()
        if epoch_counter != 1:
            if current_lr > 0.0001:
                updated_lr.step()

        optimizer.step()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        dalpha = alpha.grad
        with torch.no_grad():
            dalpha -= dalpha.sum(0)
            alpha -= current_lr * dalpha
            alpha[:] = proj_simplex(alpha)
        alpha.grad.zero_()

        epochs_alphas.append(tonpy(alpha))
        epochs_cost.append(loss.item())

        # Early Stopping based on the Weighted Sources Accuracy:
        ACC_source_val_data = 0
        for s in range(config.S):
            _, dev_prediction = torch.max(net(sources_x_val[s]), 1)
            task_acc = torch.sum(dev_prediction == sources_y_val[s].long()).to(dtype=torch.float) * 1.0 / \
                       sources_y_val[s].shape[0]
            ACC_source_val_data += float(alpha[s]) * task_acc.item()

        epochs_sources_accuracy.append(ACC_source_val_data)
        print('Epoch {} - LR {} - Loss {:.6f} - Weighted Source Acc {:.4f}'.format(epoch_counter, current_lr,
                                                                                   epochs_cost[-1],
                                                                                   ACC_source_val_data))
        if ACC_source_val_data < labelled_acc:
            print('ERROR: new acc=', ACC_source_val_data, 'best acc=', labelled_acc, 'Nerror=', nerrors)
            nerrors += 1
            if nerrors > config.maxerror:
                print('Early stopping applied at iteration', epoch_counter,
                      '. Dev lab ACC={:.3f}'.format(ACC_source_val_data))
                net.load_state_dict(net_weights)
                break
        else:
            net_weights = dcopy(net.state_dict())
            labelled_acc = ACC_source_val_data
            nerrors = 0

    return epochs_alphas, epochs_cost, epochs_sources_accuracy


def wjdot_sse(net, config, source_xy, Ns_list, x, val_x, cost='emd'):
    """"
    WJDOT algorithm with early stopping based on the sum of the squared errors (SSE)
    between the estimated outputs and their estimated cluster centroids

    Parameters
    ----------
    net: class
        neural network
    source_xy: torch array, [sources training samples, embedding dimension + n classes]
        concatenated source datasets
    Ns_list: list of float
        list of number of source samples
    x: torch array, [target training samples, embedding dimension]
        target training input
    val_x: torch array,  [target training samples, embedding dimension]
        target validation input
    cost: 'emd' or 'bures'
        wasserstein cost function

    Returns
    -------
    epochs_alphas: list of array with lenght
        alpha weights at each epoch
    epochs_cost: list of float
        loss function at each epoch
    epochs_sse: list of float
        sse at each epoch
    """

    sse = 10000
    current_lr = config.lr
    net_weights = dcopy(net.state_dict())

    alpha = (torch.ones(config.S, device=config.device) / config.S).requires_grad_()  # alpha initialization
    Tn = get_sources_histogram(Ns_list, config.S, config.device)  # sources histogram
    wt = torch.ones(x.shape[0], device=config.device) * 1.0 / x.shape[0]  # target histogram

    epochs_cost, epochs_sse, epochs_alphas = [], [], []
    nerrors = 0

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.l2_reg)
    updated_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.lr_decay)

    for epoch_counter in range(config.num_epochs):
        output = net(x)
        if 'bures' in str(cost):
            loss = cost_bures_jdot(alpha, source_xy, x, output, Tn, wt, target_reg=config.beta,
                                   bures_reg=config.bures_reg)
        else:
            loss = cost_emd_jdot(alpha, source_xy, x, output, Tn, wt, target_reg=config.beta)
        optimizer.zero_grad()
        loss.backward()
        if epoch_counter != 0:
            if current_lr > 0.0001:
                updated_lr.step()
        optimizer.step()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        dalpha = alpha.grad
        with torch.no_grad():
            dalpha -= dalpha.sum(0)
            alpha -= current_lr * dalpha
            alpha[:] = proj_simplex(alpha)
        alpha.grad.zero_()
        epochs_alphas.append(tonpy(alpha))
        epochs_cost.append(loss.item())

        # Early Stopping based on the Sum of Squared Errors (SSE):
        SSE_val = compute_sse(net, config, val_x)

        print('Epoch {} - LR {} - Loss {:.6f} - SSE {:.4f}'.format(epoch_counter, current_lr,
                                                                   epochs_cost[-1], SSE_val))
        if SSE_val - 0.001 > sse:
            nerrors += 1
            if nerrors > config.maxerror:
                print('Early stopping applied at iteration', epoch_counter, '. Dev SSE={:.3f}'.format(SSE_val))
                net.load_state_dict(net_weights)
                break
        else:
            net_weights = dcopy(net.state_dict())
            sse = SSE_val
            nerrors = 0
        epochs_sse.append(SSE_val)
    return epochs_alphas, epochs_cost, epochs_sse
