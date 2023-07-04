import torch.optim as optim
from torch.utils.data import TensorDataset
import os
import sys
from copy import deepcopy as dcopy

sys.path.append(os.getcwd())
from BLSTM_models import *
from BLSTM_utilities import *

############  PATH AND PARAMETERS   ################


saver_dir = str(sys.argv[1])  # Path to save the model
data_dir = str(sys.argv[2])  # Path to data
test_speaker = str(sys.argv[3])  # Example: 'se'
ExpNum = 1


class Configuration(object):
    def __init__(self):
        self.n_hidden = 50
        self.feature_size = 72
        self.n_layers = 2
        self.device = 'cuda'
        self.n_classes = 25
        self.drop_prob = 0
        self.num_epochs = 100
        self.lr = 0.01
        self.decay_step = 5
        self.decay_rate = 1.0
        self.batch_size = 100
        self.val_batch_size = 23
        self.l2_reg = 0.0
        self.patience = 20
        self.nT = 11


config = Configuration()
checkpoints_dir = saver_dir + 'checkpoints/exp' + str(ExpNum) + '/'
plot_dir = saver_dir + 'plots/'
traininglog_dir = saver_dir + 'training_logs/'

if not os.path.exists(saver_dir):
    os.makedirs(saver_dir)
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
if not os.path.exists(traininglog_dir):
    os.makedirs(traininglog_dir)


trainingLogFile = open(traininglog_dir + 'TrainingExperiment' + str(ExpNum) + '.txt', 'w')

print('')
print('## Speaker: {}'.format(test_speaker))
trainingLogFile.write('## Speaker: {}\n'.format(test_speaker))
print('## EXPERIMENT NUMBER ', ExpNum)
trainingLogFile.write('# EXPERIMENT NUMBER {:d} \n'.format(ExpNum))
print('## number of hidden layers : ', config.n_layers)
trainingLogFile.write('## number of hidden layers : {:d} \n'.format(config.n_layers))
print('## number of hidden units : ', config.n_hidden)
trainingLogFile.write('## number of hidden units : {:d} \n'.format(config.n_hidden))
print('## learning rate : ', config.lr)
trainingLogFile.write('## learning rate : {:.6f} \n'.format(config.lr))
print('## learning rate update steps: ', config.decay_step)
trainingLogFile.write('## learning rate update steps: {:d} \n'.format(config.decay_step))
print('## learning rate decay : ', config.decay_rate)
trainingLogFile.write('## learning rate decay : {:.6f} \n'.format(config.decay_rate))
print('## batch size : ', config.batch_size)
trainingLogFile.write('## batch size : {:d} \n\n'.format(config.batch_size))
print('')

#####DATA LOADING######

train_data = MyData(data_dir, test_speaker, 'TRAIN')
train_loader = torch.utils.data.DataLoader(train_data, collate_fn=my_collate, batch_size=config.batch_size,
                                           shuffle=True)
print('## N. training samples: ', len(train_data))

val_data = MyData(data_dir, test_speaker, 'VAL')
val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.val_batch_size, collate_fn=my_collate,
                                         shuffle=False)
print('## N. validation samples: ', len(val_data))
print('')

test_data = MyData(data_dir, test_speaker, 'TEST')
test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.val_batch_size, collate_fn=my_collate,
                                          shuffle=False)
print('## N. testing samples: ', len(test_data))
print('')

model = Net(config).to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2_reg)
lr_expDecay = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_step, gamma=config.decay_rate)

print('## Number of parameters = %d' %(sum(p.numel() for p in model.parameters() if p.requires_grad)))
print('## Number of steps per epoch: ', int(len(train_data) / config.batch_size))
print('## Number of total steps: ', int(len(train_data) * config.num_epochs/ config.batch_size))
print('')

##TRAINING###

train_loss = []
val_err = []

step = 1

# early stopping parameters
start_ERR = 100000
bad_generalization = 0
eps = 0
best_ERR = 100000


# Starting training:
for epoch_counter in range(config.num_epochs):  # loop over the dataset multiple times
    model.train()
    epoch_LR = optimizer.param_groups[0]['lr']
    epoch_loss, epoch_cer, step = train(model, train_loader, optimizer, lr_expDecay)
    # print statistics
    train_loss.append(epoch_loss)
    print('Epoch [{}], Step [{}], LR [{}], Loss: [{:.4f} CER: [{:.4f}]'.format(epoch_counter+1, step + 1, epoch_LR, epoch_loss, epoch_cer))
    # Computing loss on validation set
    model.eval()
    val_CER = []
    for i, val_data in enumerate(val_loader):
        val_input, val_target, val_seqlen, val_labellen = val_data
        val_input, val_target, val_seqlen, val_labellen = batch_sorting(val_input, val_target, val_seqlen, val_labellen)
        val_target = val_target[:, 0]
        outputs = model(val_input.permute(1, 0, 2).to(config.device), val_seqlen.to(config.device))
        val_CER.append(compute_ERR(outputs, val_target.to(config.device)))
    print('Validation CER: ' + str(np.mean(val_CER)))
    trainingLogFile.write('Epoch {}. Loss = {:.4f} CER ={:.4f} Val CER ={:.4f}\n'.format(epoch_counter + 1, epoch_loss, epoch_cer, np.mean(val_CER)))
    val_err.append(np.mean(val_CER))
    ERR_val = np.mean(val_CER)
    if ERR_val > start_ERR - eps:
        bad_generalization += 1
        if bad_generalization > config.patience:
            print('Early stopping applied at iteration', epoch_counter, '. Dev ERR=', best_ERR)
            model.load_state_dict(net_weights)
            break
    else:
        bad_generalization = 0
        best_ERR = ERR_val
        start_ERR = ERR_val
        net_weights = dcopy(model.state_dict())


torch.save(model.state_dict(), checkpoints_dir + 'Experiment' + str(ExpNum) + 'final_weights.pt')
print('Training Finished.')

### TESTING ###

test_CER = []
for i, val_data in enumerate(test_loader):
    test_input, test_target, test_seq_len, test_target_len = val_data
    test_input, test_target, test_seq_len, _ = batch_sorting(test_input, test_target, test_seq_len, test_target_len)
    test_target = test_target[:, 0]
    outputs = model(test_input.permute(1, 0, 2).to(config.device), test_seq_len.to(config.device))
    test_CER.append(compute_ERR(outputs, test_target.to(config.device)))
print('Test CER: ' + str(np.mean(test_CER)))
trainingLogFile.write('\n Test CER ={:.4f}\n'.format(np.mean(test_CER)))