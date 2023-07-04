##  About
    #
    # Authors: R. Turrisi (rosanna.turrisi@edu.unige.it), Raffaele Tavarone (raffaele.tavarone@iit.it), Leonardo Badino (leoardo.badino@iit.it)
    # Title: IMPROVING GENERALIZATION OF VOCAL TRACT FEATURE RECONSTRUCTION: FROM AUGMENTED ACOUSTIC INVERSION TO ARTICULATORY FEATURE RECONSTRUCTION WITHOUT ARTICULATORY DATA
    # Conference: IEEE SLT 2018, Workshop on Spoken Language Technology
    # Conference Date: 19-21/12/2018
    #
##  Description
    #
    # DNN_Input is a tf.record file loader. Data must be a tensor with dimension NumExamples x (InputDimension + OutputDimension), given by the concatenation of input and output files.
    # More info here: https://www.tensorflow.org/api_guides/python/reading_data#file_formats
    #
    


import tensorflow as tf 


#########################
#                      ##
# Data Reader          ##                  
#                      ##
#########################

def read_my_file_format(filename_queue, ExamplePlusLabelsDimension, n_classes, max_read_up_to):

    # Read multiple lines at every call
    reader = tf.TFRecordReader()

    key, serialized_example = reader.read_up_to(filename_queue,max_read_up_to)

    examples = tf.parse_example(serialized_example,
                                features={'features': tf.FixedLenFeature([ExamplePlusLabelsDimension-n_classes], tf.float32),
                                          'label': tf.FixedLenFeature([n_classes], tf.float32)})

    return examples['features'], examples['label']


#########################
#                      ##
# Input Pipeline       ##
#                      ##
#########################

def input_pipeline(filenames,batch_size, num_epochs, read_threads, ExamplePlusLabelsDimension, n_classes, max_read_up_to, capacity, min_after_dequeue):

    # Initialize files queue
    # If shuffle=True the files in the queue are shuffled,
    # i.e. the queue is not a FIFO queue anymore.

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=False)

    example,labels = read_my_file_format(filename_queue, ExamplePlusLabelsDimension, n_classes, max_read_up_to)

    examples_batch, labels_batch = tf.train.shuffle_batch([example, labels],
        batch_size=batch_size,
        num_threads=read_threads,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=True)

    return examples_batch, labels_batch
