import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from .corruptions import *

corruption_tuple = (gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                    glass_blur, motion_blur, zoom_blur, snow, frost, fog,
                    brightness, contrast, elastic_transform, pixelate, jpeg_compression,
                    speckle_noise, gaussian_blur, spatter, saturate, occlusion, affine, rotation)

corruption_dict = {corr_func.__name__: corr_func for corr_func in corruption_tuple}


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def corrupt(x, severity=2, corruption_name=None, corruption_number=-1):
    """
    :param x: image to corrupt; a 224x224x3 numpy array in [0, 255]
    :param severity: strength with which to corrupt x; an integer in [0, 5]
    :param corruption_name: specifies which corruption function to call;
    must be one of 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
                    the last four are validation functions
    :param corruption_number: the position of the corruption_name in the above list;
    an integer in [0, 18]; useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
    :return: the image x corrupted by a corruption function at the given severity; same shape as input
    """
    if len(x[0,0]) == 1:
        x_rgb = np.concatenate((x, x, x), axis=-1)
    else:
        x_rgb = x

    float_input = 0
    if "uint8" in str(np.dtype(x_rgb[0,0,0])):
        x_rgb = x_rgb
    elif "float32" in str(np.dtype(x_rgb[0,0,0])):
        float_input = 1
        x_rgb = np.uint8(x_rgb * 255.0)
    else:
        raise ValueError("Input should be a h*w*3 numpy array in uint8 or float32")

    if corruption_name:
        x_corrupted = corruption_dict[corruption_name](Image.fromarray(x_rgb), severity)
    elif corruption_number != -1:
        x_corrupted = corruption_tuple[corruption_number](Image.fromarray(x_rgb), severity)
    else:
        raise ValueError("Either corruption_name or corruption_number must be passed")

    if float_input == 1:
        x_rgb = np.float32(x_corrupted / 255.0)
    else:
        x_rgb = np.uint8(x_corrupted)

    if len(x[0,0]) == 1:
        output = rgb2gray(x_rgb)
        output =output[:, :, np.newaxis]
    else:
        output = x_rgb


    return output
