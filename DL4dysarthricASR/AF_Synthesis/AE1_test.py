##  About
    #
    # Authors: R. Turrisi (rosanna.turrisi@edu.unige.it), Raffaele Tavarone (raffaele.tavarone@iit.it), Leonardo Badino (leoardo.badino@iit.it)
    # Title: IMPROVING GENERALIZATION OF VOCAL TRACT FEATURE RECONSTRUCTION: FROM AUGMENTED ACOUSTIC INVERSION TO ARTICULATORY FEATURE RECONSTRUCTION WITHOUT ARTICULATORY DATA
    # Conference: IEEE SLT 2018, Workshop on Spoken Language Technology
    # Conference Date: 19-21/12/2018
    #
##  AE1 Description
    #
    # AE1 takes the audio as input and returns its reconstruction. This map goes through the encoding layer, 
    # which we would like to resemble an articulatory representation by adding an additional term to the standard autoencoder. 
    #
    #
##  Code details
    #
    # FLAGS OR CONSTANT VALUES. Here, the data and network parameters, the log file and directories are set up. Finally, the audio data for validation is loaded. 
    # AUXILIARY FUNCTIONS. Pearson's correlation and root mean squared error (RMSE) are here defined.
    # TEST. The train function trains the network. Early stopping on the audio validation data is applied. 
    #
##  Examples of Usage:
    #
    # >>$  python AE1_train_and_val.py 1 25
    #
    
    
from __future__ import division
from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
from datetime import datetime
import os
from math import sqrt
from sklearn.metrics import mean_squared_error
import math

import DNN_Input
from AE1_config import AE1_config  # This file contains the paths of the saving/loading folders, the train and validation file names
from AE1_config import AE1_test_filenames
from AE1_model import model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)

################################
#                             ##
# FLAGS OR CONSTANT VALUES    ##
#                             ##
################################

print('Your current installed version of tensorflow is = ', tf.__version__)

ExpNum = int(sys.argv[1])
epoch_to_load = int(sys.argv[2])

# --------------------------------set up parameters--------------------------------
WindowSize = 25  # frame window size
InputDimension = 39  # MFCCs dimension
motor_dimension = 6

reconstructed_VTVs_lambda = 2  # penalization term of the reconstructed_VTVs loss function
biases_init= 0.1
batch_size = 500
num_epochs = 2  # maximum number of training epochs
n_features = InputDimension*WindowSize
n_hidden_1 = 200
n_hidden_2 = 130
n_hidden_3 = 70
n_hidden_4 = 10  # n_hidden_4 must be equal to the number of extended VTVs


# -------------------------------------------------------Saver----------------------------------------------------

data_dir, checkpoints_dir, tensorboard_dir, traininglog_dir = AE1_config(ExpNum)
SFs_test, VTVs_test, audio_test = AE1_test_filenames(data_dir, WindowSize)

if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if not os.path.exists(traininglog_dir):
    os.makedirs(traininglog_dir) 

trainingLogFile = open(traininglog_dir + 'TrainingExperiment' + str(ExpNum) + '.txt', 'a')
Parameters_file = checkpoints_dir + 'model_epoch' + str(epoch_to_load) + '.ckpt'

# -----------------------------------------------------Testing data --------------------------------------------------

SFs = np.load(SFs_test)
SFs = np.float32(SFs[:, :motor_dimension])

VTVs = np.load(VTVs_test)
VTVs = np.float32(VTVs)

MFCCs = np.load(audio_test)
MFCCs = np.float32(MFCCs)


print('Test data loaded.')

#############################
#                          ##
# AUXILIARY FUNCTIONS      ##
#                          ##
#############################

def correlation(x, y):
    cm = np.corrcoef(x, y, rowvar=0)
    d = np.diagonal(cm, offset=x.shape[1])
    corr_mean = np.mean(d)
    return corr_mean


def rmse(x, y):
    mse = mean_squared_error(x, y)
    return sqrt(mse)

#############################
#                          ##
# TEST                     ##
#                          ##
#############################


def evaluation():

    with tf.Graph().as_default():

        with tf.device('/cpu:0'):
            with tf.variable_scope('test_variables'):
                audio_batch_test = tf.placeholder('float', [None, n_features])
            print('Placeholder initialized.')


        with tf.device('/gpu:0'):

            with tf.variable_scope('training_graph'):
                audio_reconstruction, encoding_layer = model(audio_batch_test, n_features, n_hidden_1, n_hidden_2,
          n_hidden_3, n_hidden_4, biases_init)
                
        # VARIABLES INITIALIZER
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # OPS to save and restore all the variables.
        saver = tf.train.Saver()

        # Merge summaries for tensorboard
        merged = tf.summary.merge_all()
         
        # Create a session for running operations in the Graph.
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

            sess.run(init_op)
            # Restore variables from disk.
            saver.restore(sess, Parameters_file)
            print("Model restored.")
            
            
            reconstructed_VTVs = np.zeros([VTVs.shape[0], motor_dimension])
            reconstructed_audio = np.zeros([MFCCs.shape[0], MFCCs.shape[1]])
            for i in range(0, VTVs.shape[0]):
                # Evaluate and print accuracy
                audio_reconstruction_i, encoding_i = sess.run([audio_reconstruction, encoding_layer], feed_dict={audio_batch_test: np.expand_dims(MFCCs[i, :], 0)})
                reconstructed_VTVs[i, :] = encoding_i[:, :motor_dimension]
                reconstructed_audio[i, :] = audio_reconstruction_i
           
            np.save(checkpoints_dir + 'ReconstructedAFs.npy', reconstructed_VTVs)
            err_audio = rmse(reconstructed_audio, MFCCs)
            corr_audio = correlation(reconstructed_audio, MFCCs)
            err_test = rmse(reconstructed_VTVs,VTVs)
            corr_test = correlation(reconstructed_VTVs, VTVs)
            corr_no_DNN = correlation(SFs, VTVs)
            err_no_DNN = rmse(SFs, VTVs)
            
            print('Audio RMSE: ', err_audio)
            print('Audio corr: ', corr_audio)
            print('\n')
            print('RMSE without training: ', err_no_DNN)
            print('RMSE after training (epoch '+ str(epoch_to_load) + '): ' + str(err_test))
            print('\n')
            print('Correlation without training: ', corr_no_DNN)
            print('Correlation after training (epoch '+ str(epoch_to_load) + '): ' + str(corr_test))
            
            trainingLogFile.write("\n")
            trainingLogFile.write("\nEvaluation on Testing data")
            trainingLogFile.write("\n\nRMSE on audio: " + str(err_audio))
            trainingLogFile.write("\nCorr on audio: " + str(corr_audio))
            trainingLogFile.write("\n\nMotor evaluation : ")
            trainingLogFile.write('\nRMSE without training: ' + str(err_no_DNN))
            trainingLogFile.write('\nRMSE after training (epoch '+ str(epoch_to_load)+ '): ' + str(err_test))
            trainingLogFile.write('\nCorrelation without training: ' + str(corr_no_DNN))
            trainingLogFile.write('\nCorrelation after training (epoch '+ str(epoch_to_load) + '): ' + str(corr_test))


        trainingLogFile.close()


def main(argv=None):
    evaluation()


if __name__ == '__main__':
    tf.app.run()
