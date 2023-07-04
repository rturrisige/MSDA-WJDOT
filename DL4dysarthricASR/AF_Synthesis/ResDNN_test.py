##  About
    #
    # Authors: R. Turrisi (rosanna.turrisi@edu.unige.it), Raffaele Tavarone (raffaele.tavarone@iit.it), Leonardo Badino (leoardo.badino@iit.it)
    # Title: IMPROVING GENERALIZATION OF VOCAL TRACT FEATURE RECONSTRUCTION: FROM AUGMENTED ACOUSTIC INVERSION TO ARTICULATORY FEATURE RECONSTRUCTION WITHOUT ARTICULATORY DATA
    # Conference: IEEE SLT 2018, Workshop on Spoken Language Technology
    # Conference Date: 19-21/12/2018
    #
##  ResDNN Description
    #
    # The deep neaural network consists of a residual layer that takes the statistical features SFs_batch_train
    # as input and targets acoustic features. The residual layer modulates the input with its left and right context
    # weighted by a learned parameter, thus returning a coarticulation-modulated version of the SFs.
    # Finally, from the residual layer output the acoustic feature is reconstructed. 
    #
    #
##  Code details
    #
    # FLAGS OR CONSTANT VALUES. Here, the data and network parameters, the log file and directories are set up. Finally, the validation acoustic data and SFs are loaded. 
    # AUXILIARY FUNCTIONS. Pearson's correlation and root mean squared error (RMSE) are here defined.
    # TEST. The evaluation function tests the trained network and saves the reconstructed VTVs. 
    #
##  Example of Usage:
    #
    # >>$  python ResDNN_test.py 1 25 
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
from ResDNN_config import ResDNN_config
from ResDNN_config import ResDNN_test_filenames
from ResDNN_model import residual_layer
from ResDNN_model import inference


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

# ----------------------------Data parameters------------------------------------------------

WindowSize = 7
output_ResLayer_dim = 6  # number of VTVs
InputDimension = 10  # number of Extended discrete  VTVs
n_targets = 39
ExamplePlusTargetsDimension = InputDimension * WindowSize + n_targets

# ----------------------------Network parameters-----------------------------------------------

learning_rate_starter = 0.01
n_nodes = 500
l2 = 0.01
activation = tf.nn.relu
res_biases_initialization = 0.0
biases_initialization = 0.1
residual_initialization = 'uniform between -0.01 and 0.01'


# -------------------------------------------------------Saver----------------------------------------------------

data_dir, checkpoints_dir, tensorboard_dir, traininglog_dir = ResDNN_config(ExpNum)
test_audio, test_VTVs, test_SFs, test_SFs_Ws = ResDNN_test_filenames(data_dir, WindowSize)

if not os.path.exists(checkpoints_dir):
    print('checkpoints directory not found.')
    sys.exit()
if not os.path.exists(traininglog_dir):
    print('traininglog directory not found.')
    sys.exit()
    
trainingLogFile = open(traininglog_dir + 'TrainingExperiment' + str(ExpNum) + '.txt', 'a')
Parameters_file = checkpoints_dir + 'model_epoch' + str(epoch_to_load) + '.ckpt'

# -----------------------------------------------------Testing data --------------------------------------------------

SFs_with_context = np.load(test_SFs_Ws)
SFs_with_context = np.float32(SFs_with_context)

SFs = np.load(test_SFs)
SFs = np.float32(SFs[:,:6])

VTVs = np.load(test_VTVs)
VTVs = np.float32(VTVs)

MFCCs = np.load(test_audio)
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
    return corr_mean, d


def rmse(x, y):
    rmse_sep = np.sqrt(np.mean(np.square(x-y), 0))
    rmse = sqrt(mean_squared_error(x, y))
    return rmse, rmse_sep

#############################
#                          ##
# TEST                     ##
#                          ##
#############################


def evaluation():

    with tf.Graph().as_default():

        with tf.device('/cpu:0'):
            with tf.variable_scope('test_variables'):
                SFs_batch_test = tf.placeholder('float', [None, InputDimension * WindowSize])
            print('Placeholder initialized.')


        with tf.device('/gpu:0'):

            with tf.variable_scope('training_graph'):
                res_output_test, _ = residual_layer(SFs_batch_test, InputDimension, WindowSize, output_ResLayer_dim, residual_initialization, res_biases_initialization)
                acoustic_output = inference(res_output_test, InputDimension, n_nodes, n_targets, activation, biases_initialization)
                
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
            #saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            print("Model restored.")
            reconstructed_audio = np.zeros([MFCCs.shape[0], MFCCs.shape[1]])
            reconstructed_VTVs = np.zeros([VTVs.shape[0], output_ResLayer_dim])
            
            for i in range(0, VTVs.shape[0]):
                # Evaluate and print accuracy
                reconstructed_VTVs_i, reconstructed_audio_i = sess.run([res_output_test, acoustic_output], feed_dict={SFs_batch_test: np.expand_dims(SFs_with_context[i, :], 0)})
                reconstructed_VTVs[i, :] = reconstructed_VTVs_i[:, :output_ResLayer_dim]
                reconstructed_audio[i, :] = reconstructed_audio_i
            
            # Saving reconstructed VTVs:
            np.save(checkpoints_dir + 'ReconstructedAFs.npy', reconstructed_VTVs)

                        
            err_audio, _ = rmse(reconstructed_audio, MFCCs)
            corr_audio, _ = correlation(reconstructed_audio, MFCCs)
            err_test, err_sep_test = rmse(reconstructed_VTVs,VTVs)
            corr_test, corr_sep_test = correlation(reconstructed_VTVs, VTVs)
            corr_no_DNN, _ = correlation(SFs, VTVs)
            err_no_DNN, _ = rmse(SFs, VTVs)
            
            print('Audio RMSE: ', err_audio)
            print('Audio corr: ', corr_audio)
            print('\n')
            print('RMSE without training: ', err_no_DNN)
            print('RMSE after training: ' + str(err_test))
            print('RMSE of each feature: ' + str(list(err_sep_test)))
            print('\n')
            print('Correlation without training: ', corr_no_DNN)
            print('Correlation after training ' + str(corr_test))
            print('Correlation of each feature: '+ str(list(corr_sep_test)))
            
            trainingLogFile.write("\n")
            trainingLogFile.write('\nLast epoch loaded.')
            trainingLogFile.write("\nEvaluation on Validation data")
            trainingLogFile.write("\nRMSE on audio: " + str(err_audio))
            trainingLogFile.write("\nCorr on audio: " + str(corr_audio))
            trainingLogFile.write('\nRMSE without training: ' + str(err_no_DNN))
            trainingLogFile.write('\nRMSE after training : ' + str(err_test))
            trainingLogFile.write('\nRMSE of each feature: ' +str(list(err_sep_test)))
            trainingLogFile.write('\nCorrelation without training: ' + str(corr_no_DNN))
            trainingLogFile.write('\nCorrelation after training: ' + str(corr_test))
            trainingLogFile.write('\nCorrelation of each feature: ' +str(list(corr_sep_test)))


        trainingLogFile.close()


def main(argv=None):
    evaluation()


if __name__ == '__main__':
    tf.app.run()
