import os
import sys
sys.path.append(os.getcwd())

from MSDA_WJDOT import *
from AllSpeak_WJDOT_data_configuration import *


############  PATH AND PARAMETERS   ################

saver_dir = str(sys.argv[1])  # Path to save the model
data_dir = str(sys.argv[2])  # Path to data
test_speaker = str(sys.argv[3])  # Example: 'se'

ExpNum = 1
early_stopping = 'sse'  # Measure criterion for early stopping 'sse' or 'acc'


class Configuration(object):
    def __init__(self):
        self.n_hidden = 50
        self.embedding_dim = self.n_hidden * 2
        self.feature_size = 72
        self.num_classes = 25
        self.num_epochs = 1000
        self.lr = 0.1
        self.lr_decay = 1.0
        self.decay_step = 1.0
        self.batch_size = 100
        self.val_batch_size = 500
        self.l2_reg = 0.0
        self.patience = 100
        self.S = 28
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.beta = 0.4


config = Configuration()

tonpy = lambda x: x.detach().cpu().numpy()
totorch = lambda x: torch.from_numpy(x)
todevice= lambda x: torch.from_numpy(x).to(config.device)
td = lambda x: x.to(config.device)

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
print('## EXPERIMENT NUMBER ', ExpNum)

print('## number of hidden layers : ', config.n_layers)
print('## number of hidden units : ', config.n_hidden)
print('## learning rate : ', config.lr)
print('## batch size : ', config.batch_size)
print('')

trainingLogFile.write('# EXPERIMENT NUMBER {:d} \n'.format(ExpNum))
for n, v in vars(config).items():
    trainingLogFile.write(n + str(' = ') + str(v) + '\n')


trainingLogFile.flush()


source_ey, embedding_train, y_train, embedding_test, y_test, Ns_list = load_train_test_data(data_dir)

y_train, y_test = y_train[:, 0], y_test[:, 0]

net = ClassifierLayer(config).to(config.device)

if early_stopping == 'sse':
    embedding_val, y_val = load_target_val_data(data_dir)
    alphas, cost, es = wjdot_sse(net, config, td(source_ey), Ns_list, td(embedding_train), td(embedding_val))
elif early_stopping == 'acc':
    embedding_val, y_val = load_source_val_data(data_dir)
    y_val = [y[:, 0] for y in y_val]
    sources_val_data = [td(e) for e in embedding_val], [td(l) for l in y_val]
    alphas, cost, es = wjdot_acc(net, config, source_ey, Ns_list, sources_val_data, td(embedding_train))

target_accuracy = inference(net, td(embedding_test), td(y_test))
print('Testing CER: {:.4f}'.format(1 - target_accuracy))
trainingLogFile.write('Testing CER: {:.4f}'.format(1 - target_accuracy))
trainingLogFile.flush()
trainingLogFile.close()


