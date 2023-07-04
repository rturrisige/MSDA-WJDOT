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
    # AUTOENCODER ARCHITECTURE. Th residual_layer function describes the residual layer, the first layer of the nerual network architecture. The following layers are defined in the inference function.
    # These functions are recalled in ResDNN_train_and_val.py and ResDNN_test.py
    #

import numpy as np
import tensorflow as tf

##############################
#                           ##
# define the dnn structure  ##
#                           ##
##############################



def residual_layer(motor_input, InputDimension, WindowSize, output_ResLayer_dim, residual_initialization, res_biases_initialization):
    if 'zero' in residual_initialization:
        res_weight = tf.get_variable('residual_weight', dtype=tf.float32, initializer=tf.zeros([InputDimension*WindowSize, InputDimension]))
    elif 'uniform' in residual_initialization:
        res_weight = tf.get_variable('residual_weight', dtype=tf.float32, initializer=tf.random_uniform([InputDimension*WindowSize, InputDimension], -0.01, 0.01))
    else:
        print('Residual weight initialization not found.')
        sys.exit()
    tf.summary.histogram('residual_weight', res_weight)
    res_biases = tf.get_variable('res_biases', initializer=tf.constant(res_biases_initialization, shape=[InputDimension]))
    resh2 = tf.add(tf.matmul(motor_input, res_weight), res_biases)
    cs = (WindowSize // 2) * InputDimension 
    central_motor_input = tf.slice(motor_input, [0, cs], [-1, InputDimension])
    res_output = tf.add(central_motor_input,resh2)
    return res_output, res_weight



def inference(examples, input_dim, nodes, classes, act, biases_init):
    with tf.variable_scope('layer_1'):
        layer_1_weights = tf.get_variable('weights', [input_dim, nodes],
                                          initializer=tf.contrib.layers.xavier_initializer())
        layer_1_biases = tf.get_variable('biases', initializer=tf.constant(biases_init, shape=[nodes]))
        layer_1 = act(tf.add(tf.matmul(examples, layer_1_weights), layer_1_biases))
        tf.summary.histogram('weights', layer_1_weights)
        tf.summary.histogram('biases', layer_1_biases)
        tf.summary.histogram('activation', layer_1)
    with tf.variable_scope('layer_2'):
        layer_2_weights = tf.get_variable('weights', [nodes, nodes], initializer=tf.contrib.layers.xavier_initializer())
        layer_2_biases = tf.get_variable('biases', initializer=tf.constant(biases_init, shape=[nodes]))
        layer_2 = act(tf.add(tf.matmul(layer_1, layer_2_weights), layer_2_biases))
        tf.summary.histogram('weights', layer_2_weights)
        tf.summary.histogram('biases', layer_2_biases)
        tf.summary.histogram('activation', layer_2)
    with tf.variable_scope('layer_3'):
        layer_3_weights = tf.get_variable('weights', [nodes, nodes], initializer=tf.contrib.layers.xavier_initializer())
        layer_3_biases = tf.get_variable('biases', initializer=tf.constant(biases_init, shape=[nodes]))
        layer_3 = act(tf.add(tf.matmul(layer_2, layer_3_weights), layer_3_biases))
        tf.summary.histogram('weights', layer_3_weights)
        tf.summary.histogram('biases', layer_3_biases)
        tf.summary.histogram('activation', layer_3)
    with tf.variable_scope('layer_4'):
        layer_4_weights = tf.get_variable('weights', [nodes, nodes], initializer=tf.contrib.layers.xavier_initializer())
        layer_4_biases = tf.get_variable('biases', initializer=tf.constant(biases_init, shape=[nodes]))
        layer_4 = act(tf.add(tf.matmul(layer_3, layer_4_weights), layer_4_biases))
        tf.summary.histogram('weights', layer_4_weights)
        tf.summary.histogram('biases', layer_4_biases)
        tf.summary.histogram('activation', layer_4)
    with tf.variable_scope('output'):
        output_weights = tf.get_variable('weights', [nodes, classes],
                                         initializer=tf.contrib.layers.xavier_initializer())
        output_biases = tf.get_variable('biases', initializer=tf.constant(biases_init, shape=[classes]))
        output = tf.add(tf.matmul(layer_2, output_weights), output_biases)
        tf.summary.histogram('weights', output_weights)
        tf.summary.histogram('biases', output_biases)
        tf.summary.histogram('activation', output)
    return output 
