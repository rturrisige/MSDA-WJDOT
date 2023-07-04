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
    # AUTOENCODER ARCHITECTURE. The model function describes the AE1 architecture. 
    #
    
import numpy as np
import tensorflow as tf

###############################
#                            ##
# AUTOENCODER ARCHITECTURE   ##
#                            ##
###############################


def model(data_input, n_features, n_hidden_1, n_hidden_2,
          n_hidden_3, n_hidden_4, biases_init):

    with tf.variable_scope('encoder_h1'):

        encoder_h1_weights = tf.get_variable('weights', [n_features, n_hidden_1], initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        encoder_h1_biases = tf.get_variable('biases', initializer=tf.constant(biases_init, shape=[n_hidden_1],dtype=tf.float32))

        encoder_layer_1 = tf.nn.relu(tf.add(tf.matmul(data_input, encoder_h1_weights), encoder_h1_biases))

        tf.summary.histogram('weights', encoder_h1_weights)
        tf.summary.histogram('biases', encoder_h1_biases)
        tf.summary.histogram('activation', encoder_layer_1)

    with tf.variable_scope('encoder_h2'):
        encoder_h2_weights = tf.get_variable('weights', [n_hidden_1, n_hidden_2], initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        encoder_h2_biases = tf.get_variable('biases', initializer=tf.constant(biases_init, shape=[n_hidden_2]),dtype=tf.float32)

        encoder_layer_2 = tf.nn.relu(tf.add(tf.matmul(encoder_layer_1, encoder_h2_weights), encoder_h2_biases))

        tf.summary.histogram('weights', encoder_h2_weights)
        tf.summary.histogram('biases', encoder_h2_biases)
        tf.summary.histogram('activation', encoder_layer_2)

    with tf.variable_scope('encoder_h3'):

        encoder_h3_weights = tf.get_variable('weights', [n_hidden_2, n_hidden_3],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        encoder_h3_biases = tf.get_variable('biases', initializer=tf.constant(biases_init, shape=[n_hidden_3]))

        encoder_layer_3 = tf.nn.relu(tf.add(tf.matmul(encoder_layer_2, encoder_h3_weights), encoder_h3_biases))

        tf.summary.histogram('weights', encoder_h3_weights)
        tf.summary.histogram('biases', encoder_h3_biases)
        tf.summary.histogram('activation', encoder_layer_3)

    with tf.variable_scope('encoder_h4'):

        encoder_h4_weights = tf.get_variable('weights', [n_hidden_3, n_hidden_4],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        encoder_h4_biases = tf.get_variable('biases', initializer=tf.constant(biases_init, shape=[n_hidden_4]))

        encoding_output = tf.add(tf.matmul(encoder_layer_3, encoder_h4_weights), encoder_h4_biases)
        encoding_layer_activation = tf.nn.relu(encoding_output)

        tf.summary.histogram('weights', encoder_h4_weights)
        tf.summary.histogram('biases', encoder_h4_biases)
        tf.summary.histogram('output', encoding_output)
        tf.summary.histogram('activation', encoding_layer_activation)

    with tf.variable_scope('decoder_h1'):
        decoder_h1_weights = tf.get_variable('weights', [n_hidden_4, n_hidden_3],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        decoder_h1_biases = tf.get_variable('biases', initializer=tf.constant(biases_init, shape=[n_hidden_3]))

        decoder_layer_1 = tf.nn.relu(tf.add(tf.matmul(encoding_layer_activation, decoder_h1_weights), decoder_h1_biases))

        tf.summary.histogram('weights', decoder_h1_weights)
        tf.summary.histogram('biases', decoder_h1_biases)
        tf.summary.histogram('activation', decoder_layer_1)

    with tf.variable_scope('decoder_h2'):

        decoder_h2_weights = tf.get_variable('weights', [n_hidden_3, n_hidden_2],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        decoder_h2_biases = tf.get_variable('biases', initializer=tf.constant(biases_init, shape=[n_hidden_2]),dtype=tf.float32)

        decoder_layer_2 = tf.nn.relu(tf.add(tf.matmul(decoder_layer_1, decoder_h2_weights), decoder_h2_biases))

        tf.summary.histogram('weights', decoder_h1_weights)
        tf.summary.histogram('biases', decoder_h1_biases)
        tf.summary.histogram('activation', decoder_layer_1)



    with tf.variable_scope('decoder_h3'):

        decoder_h3_weights = tf.get_variable('weights', [n_hidden_2, n_hidden_1],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        decoder_h3_biases = tf.get_variable('biases', initializer=tf.constant(biases_init, shape=[n_hidden_1]),dtype=tf.float32)

        decoder_layer_3 = tf.nn.relu(tf.add(tf.matmul(decoder_layer_2, decoder_h3_weights), decoder_h3_biases))

        tf.summary.histogram('weights', decoder_h3_weights)
        tf.summary.histogram('biases', decoder_h3_biases)
        tf.summary.histogram('activation', decoder_layer_3)


    with tf.variable_scope('decoder_h4'):

        decoder_h4_weights= tf.get_variable('weights', [n_hidden_1, n_features],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        decoder_h4_biases = tf.get_variable('biases', initializer=tf.constant(biases_init, shape=[n_features]),dtype=tf.float32)

        decoder_layer_4 = tf.add(tf.matmul(decoder_layer_3, decoder_h4_weights), decoder_h4_biases)

        tf.summary.histogram('weights', decoder_h4_weights)
        tf.summary.histogram('biases', decoder_h4_biases)
        tf.summary.histogram('output', decoder_layer_4)

    return decoder_layer_4 , encoding_output

 
