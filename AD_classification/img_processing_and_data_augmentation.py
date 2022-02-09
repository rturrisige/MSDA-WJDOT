import os
import numpy as np
import sys
from glob import glob as gg
from torchvision import transforms
import torch
from img_processing_functions import *
data_path = str(sys.argv[1])

##
# ###################
#     UTILITIES     #
# ###################


def img_processing(data, param, zoom=False, shift=False, rotation=False):
    """
    :param data: file name
    :param param: class function containing the parameters of the affine transformations
    :param zoom: set it to True to perform image zoom
    :param shift: set it to True to perform image shift
    :param rotation: set it to True to perform image rotation
    :return: transformed image and the associated label
    """
    Ndim = len(param.smallest_img_size)
    final_dim = [int(d / int(1/param.scaling)) for d in param.smallest_img_size]
    input_image, label = np.load(data, allow_pickle=True)
    if zoom:
        input_image = random_zoom(input_image, min_percentage=param.z_min, max_percentage=param.z_max)
    if shift:
        input_image = random_shift(input_image)
    if rotation:
        input_image = random_3Drotation(input_image, min_angle=param.min_angle, max_angle=param.max_angle)
    input_image = resize_data_volume_by_scale(input_image, scale=param.scaling)
    new_scaling = [final_dim[i]/input_image.shape[i] for i in range(Ndim)]
    final_image = resize_data_volume_by_scale(input_image, scale=new_scaling)
    return final_image, label


##
# ###################
#  Set Parameters   #
# ###################

class Parameter(object):
    def __init__(self):
        self.smallest_img_size = [192, 192, 146]  # dimension of the smallest image
        self.scaling = 0.5  # scaling to apply to all images
        # Zoom parameters:
        self.z_min = 0.8  # minimum percentage
        self.z_max = 1.2  # maximum percentage
        # Rotation parameters:
        self.min_angle = -5  # minimum angle rotation
        self.max_angle = 5   # maximum angle rotation


param = Parameter()

# Define saving folders:
for folder in ['processed_data/', 'augmentation/']:
    if not os.path.exists(data_path + folder):
        os.makedirs(data_path + folder)

for subfolder in ['zoom/', 'shift/', 'rotation/', 'all', 'all2', 'all3']:
    if not os.path.exists(data_path + 'augmentation/' + subfolder):
        os.makedirs(data_path + 'augmentation/' + subfolder)

# Load npy imaging data: 0 and 2 corresponds to control and AD subjects
files = gg(data_path + 'dataset/0*.npy') + gg(data_path + 'dataset/2*.npy')

# Start pre-processing and data augmentation:
for f in files:
    name = os.path.basename(f).split('.npy')[0]
    # Processing
    processed_data = img_processing(f, param)
    np.save(data_path + 'processed_data/' + name + '.npy', processed_data)
    # Data Augmentation: transformation are applied separately
    processed_data = img_processing(f, param, zoom=True)
    np.save(data_path + 'augmentation/zoom/' + name + '.npy', processed_data)
    processed_data = img_processing(f, param, shift=True)
    np.save(data_path + 'augmentation/shift/' + name + '.npy', processed_data)
    processed_data = img_processing(f, param, rotation=True)
    np.save(data_path + 'augmentation/rotation/' + name + '.npy', processed_data)
    # Data Augmentation: transformation are applied simultaneously
    processed_data = img_processing(f, param, zoom=True, shift=True, rotation=True)
    np.save(data_path + 'augmentation/all/' + name + '.npy', processed_data)
    processed_data = img_processing(f, param, zoom=True, shift=True, rotation=True)
    np.save(data_path + 'augmentation/all2/' + name + '.npy', processed_data)
    processed_data = img_processing(f, param, zoom=True, shift=True, rotation=True)
    np.save(data_path + 'augmentation/all3/' + name + '.npy', processed_data)


