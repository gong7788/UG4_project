import numpy as np
import webcolors
from skimage.color import rgb2hsv


def namedtuple_to_rgb(rgb):
    return np.array(rgb, dtype=np.float32)/255

def get_colour(colour_name):
    colour_tuple = webcolors.name_to_rgb(colour_name)
    return np.array(colour_tuple, dtype=np.float32)/255



def name_to_rgb(name):
    rgb = webcolors.name_to_rgb(name)
    return np.array(rgb, dtype=np.float32) / 255

def name_to_hsv(name):
    rgb = name_to_rgb(name)
    hsv = rgb2hsv([[rgb]])[0][0]
    return hsv
