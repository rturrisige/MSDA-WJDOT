from scipy import ndimage
import numpy as np

##
# ZOOM TRANSFORMATION


def random_zoom(matrix,min_percentage=0.8, max_percentage=1.2):
    """
    Zoom (in or out) generating a random value between min and max percentage
    :param min_percentage: minimum percentage of zoom, type=float
    :param max_percentage: maximum percentage of zoom, type=float
    """
    z = np.random.sample() * (max_percentage-min_percentage) + min_percentage
    zoom_matrix = np.array([[z, 0, 0, 0],
                            [0, z, 0, 0],
                            [0, 0, z, 0],
                            [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(matrix, zoom_matrix)

##
# SCALING FUNCTION


def resize_data_volume_by_scale(data, scale):
    """
    Resize the data based on the provided scale
    :param scale: float between 0 and 1
    """
    if isinstance(scale, float):
        scale_list = [scale, scale, scale]
    else:
        scale_list = scale
    return ndimage.interpolation.zoom(data, scale_list, order=0)


##
# SHIFT TRANSFORMATION


def transform_matrix_offset_center_3d(matrix, x, y, z):
    offset_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(matrix, offset_matrix)


def random_shift(img_numpy, max_percentage=0.4):
     """
     Shift data with a randomly shifting parameter
     :param img_numpy: 3D numpy array
     :param max_percentage: maximum shifting parameter in percentage
     """
     dim1, dim2, dim3 = img_numpy.shape
     m1, m2, m3 = int(dim1*max_percentage/2), int(dim1*max_percentage/2), int(dim1*max_percentage/2)
     d1 = np.random.randint(-m1, m1)
     d2 = np.random.randint(-m2, m2)
     d3 = np.random.randint(-m3, m3)
     return transform_matrix_offset_center_3d(img_numpy, d1, d2, d3)


##
# ROTATION TRANSFORMATION


def random_3Drotation(img, min_angle, max_angle):
   """
   Returns a random rotated array in the same shape
   :param img_numpy: 3D numpy array
   :param min_angle: in degrees
   :param max_angle: in degrees
   """
   assert img.ndim == 3, "provide a 3d numpy array"
   assert min_angle < max_angle, "min should be less than max val"
   assert min_angle > -360 or max_angle < 360
   all_axes = [(1, 0), (1, 2), (0, 2)]
   angle = np.random.randint(low=min_angle, high=max_angle+1)
   axes_random_id = np.random.randint(low=0, high=len(all_axes))
   axes = all_axes[axes_random_id]
   return ndimage.rotate(img, angle, axes=axes)

##

