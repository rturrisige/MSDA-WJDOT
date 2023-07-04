import torch
import torch.nn as nn

#################################
#                              ##
#            Network           ##
#                              ##
#################################


class Net(nn.Module):
    """
    BLSTM model

    Parameters
    ----------
    config: class with parameters
    - n_hidden: number of cells
    - n_layers: number of hidden layers
    - feature_size: input size
    - n_classes: number of classes
    - drop_prob: dropout
    - device: gpu/cpu device information
    """
    def __init__(self, config):
        super(Net, self).__init__()
        self.n_hidden = config.n_hidden
        self.n_layers = config.n_layers
        self.n_input = config.feature_size
        self.n_classes = config.n_classes
        self.device = config.device
        self.dropout = config.drop_prob
        self.bi_lstm = nn.LSTM(input_size=self.n_input, hidden_size=self.n_hidden, num_layers=self.n_layers,
                               bidirectional=True,
                               dropout=self.dropout)
        self.f = nn.Linear(self.n_hidden * 2, self.n_classes)
        self.init_weights()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_seq, input_len):
        batch_size = input_seq.shape[1]
        self.hidden = self.init_hidden(batch_size)
        input_seq = torch.nn.utils.rnn.pack_padded_sequence(input_seq, input_len)
        lstm_output, self.hidden = self.bi_lstm(input_seq.float(), self.hidden)
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output)  # undo the packing operation
        last_output = torch.cat([lstm_output[input_len[j]-1, j, :].view(-1, self.n_hidden * 2) for j in range(batch_size)], 0)
        output = self.f(last_output)
        return torch.softmax(output, 1)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers * 2, batch_size, self.n_hidden).to(self.device),
                torch.randn(self.n_layers * 2, batch_size, self.n_hidden).to(self.device))

    def init_weights(self):
        initrange = 0.5
        self.f.bias.data.fill_(0.5)
        self.f.weight.data.uniform_(-initrange, initrange)


class MTL_Net(nn.Module):
    """
    BLSTM model based on Multi-task learning

    Parameters
    ----------
    config: class with parameters
    - n_hidden: number of cells
    - n_layers: number of hidden layers
    - feature_size: input size
    - n_classes: number of classes
    - drop_prob: dropout
    - device: gpu/cpu device information
    - nT: number of tasks
    """
    def __init__(self, config):
        super(MTL_Net, self).__init__()
        self.n_hidden = config.n_hidden
        self.n_layers = config.n_layers
        self.n_input = config.feature_size
        self.n_classes = config.n_classes
        self.nT = config.nT
        self.device = config.device
        self.dropout = config.dropout
        self.bi_lstm = nn.LSTM(input_size=self.n_input, hidden_size=self.n_hidden, num_layers=self.n_layers,
                               bidirectional=True,
                               dropout=self.dropout)
        self.ft = nn.ModuleList()
        for t in range(self.nT):
            self.ft.append(nn.Linear(self.n_hidden * 2, self.n_classes))
        self.init_weights()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_seq, input_len, num_task):
        batch_size = input_seq.shape[1]
        self.hidden = self.init_hidden(batch_size)
        input_seq = torch.nn.utils.rnn.pack_padded_sequence(input_seq, input_len)  #
        lstm_output, self.hidden = self.bi_lstm(input_seq.float(), self.hidden)
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output)
        last_output = torch.cat([lstm_output[input_len[j]-1, j, :].view(-1, self.n_hidden * 2) for j in range(batch_size)], 0)
        output = self.ft[num_task](last_output)
        return torch.softmax(output, 1)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers * 2, batch_size, self.n_hidden).to(self.device),
                torch.randn(self.n_layers * 2, batch_size, self.n_hidden).to(self.device))

    def init_weights(self):
        init_range = 0.5
        for nt in range(self.nT):
            self.ft[nt].bias.data.fill_(0.5)
            self.ft[nt].weight.data.uniform_(-init_range, init_range)

