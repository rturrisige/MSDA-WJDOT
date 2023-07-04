import os
import sys

def ResDNN_config(ExpNum):
  
  saver_dir = os.getcwd() + '/'
 

  checkpoints_dir = saver_dir + 'ResDNN_results/checkpoints/exp' + str(ExpNum) + '/'
  tensorboard_dir = saver_dir + 'ResDNN_results/tensorboard/exp' + str(ExpNum) + '/'
  traininglog_dir = saver_dir + 'ResDNN_results/training_logs/'
  
  return saver_dir, checkpoints_dir, tensorboard_dir, traininglog_dir


def ResDNN_train_and_val_filenames(data_dir, WindowSize, Number_of_Files):

  # TRAINING DATA: define filenames list 
  filenames_train = [data_dir + 'train_SFs_SpeakerNormalizedAudio_Ws{:d}_{:02d}.tfrecords'.format(WindowSize, i) for i in range(0, Number_of_Files)]
  # VALIDATION DATA:
  validation_audio = data_dir + 'val_SpeakerNormalizedaudio.npy'  # Speaker normalized MFCCs without context
  validation_SFs = data_dir + 'val_SFs_Ws' + str(WindowSize) +'.npy'  # statistical features with context
  
  return filenames_train, validation_audio, validation_SFs


def ResDNN_test_filenames(data_dir, WindowSize):
  
  test_audio = data_dir + 'test_SpeakerNormalizedaudio.npy' # Speaker normalized MFCCs without context
  test_VTVs = data_dir + 'test_AFs.npy' # measured VTVs
  test_SFs = data_dir + 'test_SFs.npy' # statistical features without context
  test_SFs_Ws = data_dir + 'test_SFs_Ws' + str(WindowSize) +'.npy'  # statistical features with context
  
  return test_audio, test_VTVs, test_SFs, test_SFs_Ws


  
  