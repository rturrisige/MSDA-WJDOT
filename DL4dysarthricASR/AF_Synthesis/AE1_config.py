import os
import sys

def AE1_config(ExpNum):
  
  saver_dir = os.getcwd() + '/'

  checkpoints_dir = saver_dir + 'AE1_results/checkpoints/exp' + str(ExpNum) + '/'
  tensorboard_dir = saver_dir + 'AE1_results/tensorboard/exp' + str(ExpNum) + '/'
  traininglog_dir = saver_dir + 'AE1_results/training_logs/'
  
  return saver_dir, checkpoints_dir, tensorboard_dir, traininglog_dir

def AE1_train_and_val_filenames(data_dir, WindowSize,Number_of_Files):

  # TRAINING DATA: define filenames list 
  filenames_train = [data_dir + 'train_SpeakerNormalizedAudio_SFs_Ws{:d}_{:02d}.tfrecords'.format(WindowSize, i) for i in range(0, Number_of_Files)]
  
  # VALIDATION DATA:
  validation_audio = data_dir + 'val_SpeakerNormalizedAudio_Ws' + str(WindowSize) + '.npy' #Speaker Normalized MFCCs with context

  return filenames_train, validation_audio


def AE1_test_filenames(data_dir, WindowSize):
  
  SFs_test = data_dir +  'test_SFs.npy'  # statistical features
  VTVs_test = data_dir + 'test_AFs.npy'  # measured VTVs 
  audio_test = data_dir + 'test_SpeakerNormalizedAudio_Ws' + str(WindowSize) + '.npy'  # Speaker normalized MFCCs with context
  
  return SFs_test, VTVs_test, audio_test