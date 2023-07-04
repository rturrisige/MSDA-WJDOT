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
    # BUILT THE GRAPH AND RUN IT. The train function trains the network. Early stopping on the audio validation data is applied. 
    #
##  Examples of Usage:
    #
    # >>$  python AE2_train_and_val.py 1 
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
from AE2_config import AE2_train_and_val_filenames
from AE2_model import model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

###############################
#                            ##
# FLAGS OR CONSTANT VALUES   ##
#                            ##
###############################

print('Your current installed version of tensorflow is = ', tf.__version__)

ExpNum = int(sys.argv[1])

# --------------------------------Data parameters---------------------------------------

Number_of_TrainFiles = 9  # total number of training files
num_examples_per_epoch = 1787126 # total number of example frames
WindowSize = 25  #  frame window context size
InputDimension = 10  # number of extended descrete VTVs
step_val = 420

# -------------------------------Network parameters-------------------------------------
biases_init= 0.1
batch_size = 500
num_epochs = 50  # maximum number of training epochs 
n_features = InputDimension*WindowSize
n_hidden_1 = 200
n_hidden_2 = 130
n_hidden_3 = 70
n_hidden_4 = 39  # it must be equal to the number of audio features
audio_lambda = 0.5 # penalization term of the audio loss function

optimizer_name = 'GD'
if optimizer_name not in ['adam', 'momentum', 'GD']:
    print('Optimizer not found.')
    print('Allowed choices : adam, momentum, GD.')
    sys.exit()
    
learning_rate_starter = 0.01

# Queue parameters
read_threads = 3
max_read_up_to = 200000
capacity = num_examples_per_epoch
min_after_dequeue = 2000

#-----------------------------Directories----------------------------------------------------------------------------------------

data_dir, checkpoints_dir, tensorboard_dir, traininglog_dir = AE2_config(ExpNum)
filenames_train, validation_audio, validation_SFs = AE2_train_and_val_filenames(data_dir, WindowSize, Number_of_TrainFiles)

if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if not os.path.exists(traininglog_dir):
    os.makedirs(traininglog_dir)

# --------------------------------data loading------------------------------------------

# Statistical features with context 
SFs_val = np.load(validation_SFs)
SFs_val = np.float32(SFs_val)

# (Speaker normalized) audio data 
audio_val = np.load(validation_audio)
audio_val = np.float32(audio_val)
print('Validation data loaded!')

#----------------------------- Training LogFile ---------------------------------------

trainingLogFile = open(traininglog_dir + 'TrainingExperiment' + str(ExpNum) + '.txt', 'w')
trainingLogFile.write('Experiment number ' + str(ExpNum))
trainingLogFile.write('N. max of epochs = ' + str(num_epochs) + '\n')
trainingLogFile.write('batch size = ' + str(batch_size) + '\n')
trainingLogFile.write('num hidden layers = 4\n')
trainingLogFile.write('N. neurons (layer 1) = ' + str(n_hidden_1) + '\n')
trainingLogFile.write('N. neurons (layer 2) = ' + str(n_hidden_2) + '\n')
trainingLogFile.write('N. neurons (layer 3) = ' + str(n_hidden_3) + '\n')
trainingLogFile.write('N. neurons (encoding layer) = ' + str(n_hidden_4) + '\n')
trainingLogFile.write('activation function =  hard tangent \n')
trainingLogFile.write('learining rate starter = ' + str(learning_rate_starter) + '\n')
trainingLogFile.write('optimizer  = ' + optimizer_name + '\n')
trainingLogFile.write('audio lambda = ' + str(audio_lambda) + '\n')
trainingLogFile.write('\n')
print('\n')
print('Experiment number ' + str(ExpNum))
print('N. max of epochs = ' + str(num_epochs))
print('batch size = ' + str(batch_size))
print('num hidden layers = 4')
print('N. neurons (layer 1) = ' + str(n_hidden_1))
print('N. neurons (layer 2) = ' + str(n_hidden_2))
print('N. neurons (layer 3) = ' + str(n_hidden_3))
print('N. neurons (encoding layer) = ' + str(n_hidden_4))
print('activation function =  hard tangent')
print('learining rate starter = ' + str(learning_rate_starter))
print('optimizer  = ' + optimizer_name)
print('audio lambda = ' + str(audio_lambda) + '\n')
print('\n')

###############################
#                            ##
# AUXILIARY FUNCTIONS        ##
#                            ##
###############################


def correlation(x, y):
    cm = np.corrcoef(x, y, rowvar=0)
    d = np.diagonal(cm, offset=x.shape[1])
    corr_mean = np.mean(d)
    return corr_mean


def rmse(x, y):
    mse = mean_squared_error(x, y)
    return sqrt(mse)



###############################
#                            ##
# BUILT THE GRAPH AND RUN IT ##
#                            ##
###############################



def train():
  
    start_time = datetime.now()

    for f in filenames_train:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
      
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            with tf.variable_scope('input'):
                motor_batch_val = tf.placeholder("float", [None, n_features])
            print('Placeholder initialized.')

            with tf.variable_scope('training_batch'):
                motor_batch_train, audio_batch_train = \
                    DNN_Input.input_pipeline(filenames_train,
                                             batch_size=batch_size,
                                             num_epochs=num_epochs,
                                             read_threads=read_threads,
                                             ExamplePlusLabelsDimension=n_features+n_hidden_4,
                                             n_classes=n_hidden_4,
                                             max_read_up_to=max_read_up_to,
                                             capacity=capacity,
                                             min_after_dequeue=min_after_dequeue)
                    
        with tf.device('/gpu:0'):
            
            global_step = tf.Variable(0, trainable=False)
            
            with tf.variable_scope('training_graph') as train_scope:
                
                motor_prediction_train, reconstructed_audio_train = model(motor_batch_train, n_features, n_hidden_1, n_hidden_2,
          n_hidden_3, n_hidden_4, biases_init)
                
                with tf.variable_scope('cost'):
                    audio_cost = tf.reduce_mean(tf.pow(reconstructed_audio_train - audio_batch_train, 2))
                    motor_cost = tf.reduce_mean(tf.pow(motor_prediction_train - motor_batch_train, 2))
                    cost = audio_lambda * audio_cost + motor_cost
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

            with tf.name_scope('training_evaluation'):
                with tf.variable_scope(train_scope, reuse=True):
                     motor_prediction_val, reconstructed_audio_val = model(motor_batch_val, n_features, n_hidden_1, n_hidden_2,
          n_hidden_3, n_hidden_4, biases_init)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # OPS to save and restore all the variables.
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='training_graph'))

        # Merge summaries for tensorboard
        merged = tf.summary.merge_all()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        #config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            # tensorboard writer
            train_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                                                
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


                    if (step % int(num_examples_per_epoch / batch_size) == 0) and (step is not 1):
                        epoch_cost /= (num_examples_per_epoch / batch_size)
                        end_time = datetime.now()
                        sec = (end_time - start_time).total_seconds()
                        LR = sess.run(learning_rate)
                        print("time[{:6.2f}] step[{:7d}] speed[{:6d} LR[{:4f}] cost[{:2.5f}] "
                              .format(sec, step, int((500 * batch_size) / sec), LR, c))
                        start_time = end_time
                        if (epoch_counter % 1) == 0:
                            #train_writer.add_summary(epoch_cost, step)
                            print("epoch " + str(epoch_counter) + " RMSE = " +
                                  str(sqrt(epoch_cost)))
                            trainingLogFile.write('\n{:d}\t{:.8f}'.
                                                  format(epoch_counter, epoch_cost) + ' ' + str(sqrt(epoch_cost)))
                            audio_corr_val = 0.0
                            audio_err_val = 0.0
                                                        
                            for i in range(0, SFs_val.shape[0] - 1, step_val):
                                audio_prediction_i = sess.run(reconstructed_audio_val, feed_dict={motor_batch_val: SFs_val[i:i + step_val, :]})
                                # Evaluation on audio
                                audio_err_i = rmse(audio_prediction_i, audio_val[i: i + step_val, :])
                                audio_err_val += audio_err_i
                                audio_corr_i = correlation(audio_prediction_i, audio_val[i:i + step_val, :])
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
                            trainingLogFile.write("\nValidation. \t Audio RMSE = " + str(audio_err_val) +
                                                  "\t. Audio correlation = " + str(audio_corr_val))
                            trainingLogFile.write("\n")
                            trainingLogFile.flush()
                            save_path = saver.save(sess, checkpoints_dir + 'model_epoch' + str(epoch_counter) + '.ckpt')
                            print('Model saved in file: %s' % save_path)

                            if counter4earlystopping > 2:
                               print('\nCross-validation- Done training')
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




def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.app.run()
 
