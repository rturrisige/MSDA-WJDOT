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
    # BUILT THE GRAPH AND RUN IT. The train function trains the network. Early stopping on the audio validation data is applied. 
    #
##  Example of Usage:
    #
    # >>$  python ResDNN_train_and_val.py 1 
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
import scipy.stats as stats
import math 

import DNN_Input
from ResDNN_config import ResDNN_config
from ResDNN_config import ResDNN_train_and_val_filenames
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


# ----------------------------Data parameters-------------------------------------------------------------------------------------

Number_of_TrainFiles = 9  # number of training files
num_examples_per_epoch = 1787126  # total number of example frames
WindowSize = 7 #number of frame window size
output_ResLayer_dim = 6  # number of VTVs
InputDimension = 10  # number of Extended discrete  VTVs
n_targets = 39  # number of phonemes
ExamplePlusTargetsDimension = InputDimension * WindowSize + n_targets
step_val = 420

# ---------------------------Network parameters------------------------------------------------------------------------------------

optimizer_name = 'GD'

if optimizer_name not in ['adam', 'momentum', 'GD']:
    print('Optimizer not found.')
    print('Allowed choices : adam, momentum, GD.')
    sys.exit()

learning_rate_starter = 0.01
batch_size = 100
num_epochs = 50
n_nodes = 500
l2 = 0.01
activation = tf.nn.relu
res_biases_initialization = 0.0
biases_initialization = 0.1
residual_initialization = 'uniform between -0.01 and 0.01'

# Queue parameters
read_threads = 3
max_read_up_to = 200000
capacity = num_examples_per_epoch
min_after_dequeue = 2000

#-----------------------------Directories----------------------------------------------------------------------------------------

data_dir, checkpoints_dir, tensorboard_dir, traininglog_dir = ResDNN_config(ExpNum)
filenames_train, validation_audio, validation_SFs = ResDNN_train_and_val_filenames(data_dir, WindowSize, Number_of_TrainFiles)

if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if not os.path.exists(traininglog_dir):
    os.makedirs(traininglog_dir)

trainingLogFile = open(traininglog_dir + 'TrainingExperiment' + str(ExpNum) + '.txt', 'w')


# -----------------------------------Validation data loading---------------------------------------------------------------

# statistical features (with context) loading:
SFs_val = np.load(validation_SFs)
SFs_val = np.float32(SFs_val)

# (speaker-normalized) audio loading:
audio_val = np.load(validation_audio)
audio_val = np.float32(audio_val)

print('Validation data loaded.')  

#############################
#                          ##
# Auxiliary functions      ##
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
  



####################
#                 ##
#  Trainining     ##
#                 ##
####################


def training_runner(): 
    start_time = datetime.now()

    for f in filenames_train:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.Graph().as_default():

        with tf.device('/cpu:0'):
            with tf.variable_scope('validation_variable'):
                SFs_batch_val = tf.placeholder(tf.float32, [None, InputDimension * WindowSize])
            print('Placeholder initialized.')

            with tf.variable_scope('training_batch'):
                SFs_batch_train, audio_batch_train = \
                    DNN_Input.input_pipeline(filenames_train,
                                             batch_size=batch_size,
                                             num_epochs=num_epochs,
                                             read_threads=read_threads,
                                             ExamplePlusLabelsDimension=ExamplePlusTargetsDimension,
                                             n_classes=n_targets,
                                             max_read_up_to=max_read_up_to,
                                             capacity=capacity,
                                             min_after_dequeue=min_after_dequeue)

        with tf.device('/gpu:0'):
            global_step = tf.Variable(0, trainable=False)
            with tf.variable_scope('training_graph') as train_scope:
                res_output_train, res_w = residual_layer(SFs_batch_train, InputDimension,
                                                                         WindowSize, output_ResLayer_dim, residual_initialization, res_biases_initialization)
                reconstructed_audio_train = inference(res_output_train, InputDimension, n_nodes, n_targets,
                                                          activation, biases_initialization)

               # COST FUNCTION:
                with tf.variable_scope('cost'):
                    cost = tf.reduce_mean(tf.losses.mean_squared_error(audio_batch_train,
                                                                       reconstructed_audio_train)) + l2 * tf.nn.l2_loss(res_w)
                    tf.summary.scalar('cost', cost)

                with tf.variable_scope('optimizer'):
                    learning_rate = tf.train.exponential_decay(learning_rate_starter, global_step, 10000, 0.96, staircase=True)
                    # METHOD OF MINIMIZATION:
                    if optimizer_name == 'adam':
                        optimizer = tf.train.AdamOptimizer().minimize(cost, global_step)
                    elif optimizer_name == 'momentum':
                        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost, global_step)
                    elif optimizer_name == 'GD':
                        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step)
            with tf.variable_scope('Evaluation_session'):
                with tf.variable_scope(train_scope, reuse=True):
                    
                    res_output_val, _ = residual_layer(SFs_batch_val, InputDimension, WindowSize, output_ResLayer_dim, residual_initialization, res_biases_initialization)
                    reconstructed_audio_val = inference(res_output_val, InputDimension, n_nodes, n_targets,
                                                         activation, biases_initialization)

        # VARIABLES INITIALIZER
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # OPS to save and restore all the variables.
        saver = tf.train.Saver()

        # Merge summaries for tensorboard
        merged = tf.summary.merge_all()

        # Create a session for running operations in the Graph.
        configuration = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        configuration.gpu_options.allow_growth = True
        
        with tf.Session(config=configuration) as sess:

            train_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

            # Initialize all the variables (including the epoch counter, hidden somewhere).
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print('')
            print('## EXPERIMENT NUMBER ', ExpNum)
            trainingLogFile.write('# EXPERIMENT NUMBER {:d} \n'.format(ExpNum))
            print('## Method: Residual neural network')
            trainingLogFile.write('# Method: Residual neural network\n')
            print('## Residual weight initialization ' + residual_initialization)
            trainingLogFile.write('## Residual weight initialization ' + residual_initialization + '\n')
            print('## number of hidden layers : ', 5)
            trainingLogFile.write('## number of hidden layers : {:d} \n'.format(5))
            print('## number of neurons : ', n_nodes)
            trainingLogFile.write('## number of neurons : {:d} \n'.format(n_nodes))
            print('## window size : ', WindowSize)
            trainingLogFile.write('## Ws {:d} \n'.format(WindowSize))
            print('## optimizer : ' + optimizer_name)
            trainingLogFile.write('## optimizer : ' + optimizer_name + '\n')
            print('## learning rate starter: ' + str(learning_rate_starter))
            trainingLogFile.write('## learning rate : ' + str(learning_rate) + '\n')
            print('## number of epochs : ', num_epochs)
            trainingLogFile.write('## number of epochs : {:d} \n'.format(num_epochs))
            print('## batch size : ', batch_size)
            trainingLogFile.write('## batch size : {:d} \n'.format(batch_size))

            try:
                # variables for cross-validation:
                starting_error = 100.0
                counter4earlystopping = 0

                step = 1
                epoch_counter = 1
                epoch_cost = 0.0
                start_training_time = datetime.now()
                start_time = datetime.now()
                while not coord.should_stop():

                    # Run training step
                    _, c = sess.run([optimizer, cost])
                    epoch_cost += c

                    if np.isinf(c):
                        print('GOT INSTABILITY: cost is inf. Leaving.')
                        sys.exit()

                
                    if (step % int(num_examples_per_epoch / batch_size) == 0) and (step != 1):
                        epoch_cost /= (num_examples_per_epoch / batch_size)
                        end_time = datetime.now()
                        sec = (end_time - start_time).total_seconds()
                        LR = sess.run(learning_rate)
                        print("time[{:6.2f}] step[{:7d}] speed[{:6d} LR[{:4f}] cost[{:2.5f}] "
                              .format(sec, step, int((500 * batch_size) / sec), LR, c))
                        start_time = end_time
                        if (epoch_counter % 1) == 0:
                            #summary = sess.run(merged)
                            #train_writer.add_summary(summary, step)
                            print("epoch " + str(epoch_counter) + " RMSE = " +
                                  str(sqrt(epoch_cost)))
                            trainingLogFile.write('\n{:d}\t{:.8f}'.
                                                  format(epoch_counter, epoch_cost) + ' ' + str(sqrt(epoch_cost)))
                            audio_corr_val = 0.0
                            audio_err_val = 0.0
                            
                            for i in range(0, SFs_val.shape[0] - 1, step_val):
                                audio_prediction_i = sess.run(reconstructed_audio_val, feed_dict={SFs_batch_val: SFs_val[i:i + step_val, :]})
             
                                # Evaluation on audio
                                audio_err_i, _ = rmse(audio_prediction_i, audio_val[i: i + step_val, :])
                                audio_err_val += audio_err_i
                                audio_corr_i, _ = correlation(audio_prediction_i, audio_val[i:i + step_val, :])
                                audio_corr_val += audio_corr_i

                            audio_err_val /= ((audio_val.shape[0] - 1) / step_val)
                            audio_corr_val /= ((audio_val.shape[0] - 1) / step_val)
                            if audio_err_val > starting_error:
                               counter4earlystopping += 1
                            else:
                               counter4earlystopping = 0
                            starting_error = audio_err_val
                            print("\n")
                            print("Epochs: ", epoch_counter, ". Evaluation on validation dataset:")
                            print("Audio: RMSE = ", audio_err_val)
                            print("Audio: correlation = ", audio_corr_val)
                            print("\n")
                            trainingLogFile.write("\nValidation. \t Audio RMSE = " + str(audio_err_val) +
                                                  "\t. Audio correlation = " + str(audio_corr_val))
                            trainingLogFile.write("\n")
                            trainingLogFile.flush()
                            save_path = saver.save(sess, checkpoints_dir + 'model_epoch' + str(epoch_counter) + '.ckpt')
                            print('Model saved in file: %s' % save_path)

                            if counter4earlystopping > 2:
                               print('Done training (Early stopping)')
                               end_time = datetime.now()
                               training_time = end_time - start_training_time
                               total_time = end_time - start_time
                               print('Training started at ', start_training_time)
                               trainingLogFile.write('\nTraining started at ' + str(start_training_time))
                               print('Training ended   at ', end_time)
                               trainingLogFile.write('\nTraining ended   at ' + str(end_time))
                               print('Total training      ', training_time)
                               trainingLogFile.write('\nTotal training      ' + str(training_time))
                               print('Total time          ', total_time)
                               trainingLogFile.write('\nTotal time          ' + str(total_time))
                               trainingLogFile.close()
                               sys.exit()

                        print('\n')
                        epoch_counter += 1
                        epoch_cost = 0.0

                    step += 1
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')

            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            coord.join(threads)
            save_path = saver.save(sess, checkpoints_dir + 'model_end.ckpt')
            print("model saved in file: %s" % save_path)

            # Print useful info and a log file
            end_time = datetime.now()

            training_time = end_time - start_training_time
            total_time = end_time - start_time

            print('Training started at ', start_training_time)
            trainingLogFile.write('\nTraining started at ' + str(start_training_time))
            print('Training ended   at ', end_time)
            trainingLogFile.write('\nTraining ended   at ' + str(end_time))
            print('Total training      ', training_time)
            trainingLogFile.write('\nTotal training      ' + str(training_time))
            print('Total time          ', total_time)
            trainingLogFile.write('\nTotal time          ' + str(total_time))

            training_log_string = '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(WindowSize, batch_size,
                                                                        read_threads, max_read_up_to,
                                                                        min_after_dequeue, num_epochs, training_time)
            trainingLogFile.write(training_log_string)
            trainingLogFile.close()



def main(argv=None):
   training_runner()
if __name__ == '__main__':
    tf.app.run()
 
