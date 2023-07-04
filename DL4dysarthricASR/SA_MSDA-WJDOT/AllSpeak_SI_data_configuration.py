from torch.utils.data import Dataset
from glob import glob
import os
import sys
import numpy as np


class MyData(Dataset):
    """
        Class for Data Loading

        Parameters
        ----------

        data_dir: str
            path to data
        test_speaker: str
            target speaker
        data: str
            'TRAIN', 'VAL', or 'TEST'

        Returns
        -------

        input_features: numpy array
            input sequence
        target: numpy array
            target sequence
        seq_len: int
            length of input sequence
        label_len: int
            length of label sequence
    """

    def __init__(self, data_dir, test_speaker, data='TRAIN'):
        self.data_dir = data_dir
        if data == 'TEST':
            # CREATE TESTING SET:
            self.paths = glob(
                self.data_dir + 'patients/' + test_speaker + '/*_1.dat')  # take one sample for each sentence
        elif data in ['TRAIN', 'VAL']:
            # Data from dysarthric speakers:
            dysarthric_speakers = os.listdir(self.data_dir + 'patients/')  # all dysarthric speakers
            dysarthric_speakers.remove(test_speaker)  # remove test speakers from learning set
            # Data from healthy speakers:
            healthy_speakers = glob(self.data_dir + 'controls/*.dat')  # all healthy speakers

            # CREATE TRAINING AND VALIDATION SETS:
            train_list, dev_list = [], []
            # Data from healthy speakers
            dev_data = glob(
                self.data_dir + 'controls/allcontrols/*_0.dat')  # take one sample for each sentence to create the validation set
            train_list += [i for i in healthy_speakers if i not in dev_data]  # remove validation set from training set
            dev_list += dev_data
            # Data from dysarhtric speakers
            for s in dysarthric_speakers:
                healthy_speakers = glob(self.data_dir + 'patients/' + s + '/*.dat')
                dev_data = glob(self.data_dir + 'patients/' + s + '/*_0.dat')
                train_list += [i for i in healthy_speakers if i not in dev_data]
            for s in dysarthric_speakers:
                dev_list += glob(self.data_dir + 'patients/' + s + '/*_0.dat')
            if data == 'TRAIN':
                self.paths = train_list
            elif data == 'VAL':
                self.paths = dev_list
        else:
            print('data type not found.')
            sys.exit()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        input_features = np.genfromtxt(self.paths[index], dtype=None)
        name = os.path.basename(self.paths[index]).replace('dat', 'npy')
        if os.path.basename(self.paths[index].split('/FTC')[0]) == 'controls':
            target = np.load(self.data_dir + 'controls/' + name)
        else:
            target = np.load(self.data_dir + 'patients/' + name)
        seq_len, label_len = len(input_features), len(target)
        return input_features, target, seq_len, label_len

