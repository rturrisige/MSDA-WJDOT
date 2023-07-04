##  About
    #
    # Authors: R. Turrisi (rosanna.turrisi@edu.unige.it), Raffaele Tavarone (raffaele.tavarone@iit.it), Leonardo Badino (leoardo.badino@iit.it)
    # Title: IMPROVING GENERALIZATION OF VOCAL TRACT FEATURE RECONSTRUCTION: FROM AUGMENTED ACOUSTIC INVERSION TO ARTICULATORY FEATURE RECONSTRUCTION WITHOUT ARTICULATORY DATA
    # Conference: IEEE SLT 2018, Workshop on Spoken Language Technology
    # Conference Date: 19-21/12/2018
    #
##  AE2 Description
    #
    # Here, the statistical features (SFs) are the input of the AE which provide the articulatory reconstruction. 
    # We force th encoding layer to match the acoustic latent representation. 
    #
    #
##  Code details
    #
    # FLAGS OR CONSTANT VALUES. Here, the data and network parameters, the log file and directories are set up. Finally, the validation acoustic data and SFs are loaded. 
    # AUXILIARY FUNCTIONS. Pearson's correlation and root mean squared error (RMSE) are here defined.
    # TEST. The evaluation function tests the trained network on the testing set.
    #
##  Examples of Usage:
    #
    # >>$  python AE2_test.py 1 25 
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
from AE2_config import AE2_config
from AE2_config import AE2_test_filenames
from AE2_model import model

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
InputDimension = 10  # number of extended descrete VTVs
WindowSize = 25
motor_dimension = 6

biases_init= 0.1
n_features = InputDimension*WindowSize
n_hidden_1 = 200
n_hidden_2 = 130
n_hidden_3 = 70
n_hidden_4 = 39  # it must be equal to the number of audio features

# -------------------------------------------------------Saver----------------------------------------------------

data_dir, checkpoints_dir, tensorboard_dir, traininglog_dir = AE2_config(ExpNum)
VTVs_test, SFs_test_Ws, SFs_test = AE2_test_filenames(data_dir, WindowSize)

if not os.path.exists(checkpoints_dir):
    print('checkpoints directory not found')
    sys.exit()
if not os.path.exists(traininglog_dir):
    print('traininglog directory not found')
    sys.exit()

trainingLogFile = open(traininglog_dir + 'TrainingExperiment' + str(ExpNum) + '.txt', 'a')
Parameters_file = checkpoints_dir + 'model_epoch' + str(epoch_to_load) + '.ckpt'

# -----------------------------------------------------Testing data --------------------------------------------------

SFs = np.load(SFs_test)
SFs = np.float32(SFs[:, :motor_dimension])

VTVs = np.load(VTVs_test)
VTVs = np.float32(VTVs)

SFs_with_context = np.load(SFs_test_Ws)
SFs_with_context = np.float32(SFs_with_context)


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
                SFs_batch_test = tf.placeholder('float', [None, n_features])
            print('Placeholder initialized.')


        with tf.device('/gpu:0'):

            with tf.variable_scope('training_graph'):
                reconstruction, encoding_layer = model(SFs_batch_test, n_features, n_hidden_1, n_hidden_2,
          n_hidden_3, n_hidden_4, biases_init)
                
        # VARIABLES INITIALIZER
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # OPS to save and restore all the variables.
        saver = tf.train.Saver()

        # Merge summaries for tensorboard
        merged = tf.summary.merge_all()
        
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        # Create a session for running operations in the Graph.
        with tf.Session(config=config) as sess:

            sess.run(init_op)
            # Restore variables from disk.
            saver.restore(sess, Parameters_file)
            print("Model restored.")
            
            
            reconstructed_VTVs = np.zeros([VTVs.shape[0], motor_dimension])
            for i in range(0, VTVs.shape[0]):
                # Evaluate and print accuracy
                reconstruction_i, encoding_i = sess.run([reconstruction, encoding_layer], feed_dict={SFs_batch_test: np.expand_dims(SFs_with_context[i, :], 0)})
                rec_i = reconstruction_i[:, int(InputDimension*((WindowSize-1)/2)):int(InputDimension*((WindowSize-1)/2)) + InputDimension] #take 6 VTVs from the extended VTVs
                reconstructed_VTVs[i, :] = rec_i[:, :motor_dimension]
            
           
            np.save(checkpoints_dir + 'ReconstructedAFs.npy', reconstructed_VTVs)
            err_test = rmse(reconstructed_VTVs,VTVs)
            corr_test = correlation(reconstructed_VTVs, VTVs)
            corr_no_DNN = correlation(SFs, VTVs)
            err_no_DNN = rmse(SFs, VTVs)
            
            #print('Audio RMSE: ', err_audio)
            #print('Audio corr: ', corr_audio)
            print('\n')
            print('RMSE without training: ', err_no_DNN)
            print('RMSE after training (epoch '+ str(epoch_to_load) + '): ' + str(err_test))
            print('\n')
            print('Correlation without training: ', corr_no_DNN)
            print('Correlation after training (epoch '+ str(epoch_to_load) + '): ' + str(corr_test))
            
            trainingLogFile.write("\n")
            trainingLogFile.write("\nEvaluation on Testing data")
            #trainingLogFile.write("\n\nRMSE on audio: " + str(err_audio))
            #trainingLogFile.write("\nCorr on audio: " + str(corr_audio))
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
