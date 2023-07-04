import os
import sys

def AE2_config(ExpNum):
  
  saver_dir = os.getcwd() + '/'

  checkpoints_dir = saver_dir + 'AE2_results/checkpoints/exp' + str(ExpNum) + '/'
  tensorboard_dir = saver_dir + 'AE2_results/tensorboard/exp' + str(ExpNum) + '/'
  traininglog_dir = saver_dir + 'AE2_results/training_logs/'
  
  return saver_dir, checkpoints_dir, tensorboard_dir, traininglog_dir

def AE2_train_and_val_filenames(data_dir, WindowSize, Number_of_Files):
 
  # TRAINING DATA: define filenames list 
  filenames_train = [data_dir + 'train_SFs_SpeakerNormalizedAudio_Ws{:d}_{:02d}.tfrecords'.format(WindowSize, i) for i in range(0, Number_of_Files)]

  # VALIDATION DATA:
  validation_audio = data_dir + 'val_SpeakerNormalizedaudio.npy'  # speaker normalized MFCCs (without context)
  validation_SFs = data_dir + 'val_SFs_Ws' + str(WindowSize) +'.npy'  # statistical features with context

  return filenames_train, validation_audio, validation_SFs

def AE2_test_filenames(data_dir, WindowSize):
  
  VTVs_test = data_dir + 'test_AFs.npy'  # measured VTVs
  SFs_test_Ws = data_dir + 'test_SFs_Ws' + str(WindowSize) + '.npy'  # statistical features with context
  SFs_test = data_dir + 'test_SFs.npy'  # statistical features without context
  
  return VTVs_test, SFs_test_Ws, SFs_test