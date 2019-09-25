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


def generate_colour_generator(mean=0, std=0):
    """ Creates a colour generator function which generates random HSV colours based on the provided mean and std

    the SV channels will always use the same mean and std, ensuring that colours are not too dark or too light

    :param mean:
    :param std:
    :return: colour_generator function
    """
    def colour_generator():
        return np.array((((np.random.randn() * std + mean) % 360) / 3.6 , 100 - np.abs(np.random.randn() * 10), 100 - np.abs(np.random.rand() * 20)))/100
    return colour_generator

# These mean and std values seem to generate good values for each colour which all look sensibly like the specified colour
colour_values = {"red": (0, 5),
                 "orange": (30, 5),
                 "yellow": (58, 2),
                 "green": (120, 9),
                 "blue": (220, 13),
                 "purple": (270, 9),
                 "pink": (315, 9)}
# This dict maps a colour to its colour generator
# So to generate red use colour_generators['red']()
colour_generators = {colour: generate_colour_generator(*values) for colour, values in colour_values.items()}
