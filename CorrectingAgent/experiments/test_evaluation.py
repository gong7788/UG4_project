

import pytest
from evaluation import *
from prob_model import ColourModel
import numpy as np


def test_name_to_rgb_red():
    assert(name_to_rgb('red') == pytest.approx(np.array([1., 0., 0.])))

def test_name_to_rgb_blue():
    assert (name_to_rgb('blue') == pytest.approx(np.array([0., 0., 1.])))

def test_extract_experiment_parameters():
    filename = "random_bijection_0.7.pickle"
    assert(extract_experiment_parameters(filename) == ['random', 'bijection', '0.7'])

def test_failure_extract_params():
    filename = 'marronblue.png'
    with pytest.raises(ValueError):
        extract_experiment_parameters(filename)

# def test_extract_file():
#     filename = 'results/colours/bijection_blue_0.7.pickle'
#     assert(isinstance(extract_file(filename), ColourModel))
#
# def test_failure_extract_file():
#     filename = 'results/colours/bijectionblue.png'
#     with pytest.raises(TypeError):
#        extract_file(filename)

def test_colour_probs():
    cm = ColourModel('red')
    colour_dict = {'red':['red'], 'blue':['blue']}
    result = {'red':{'red':0.5}, 'blue':{'blue':0.5}}
    assert(colour_probs(cm, colour_dict) == pytest.approx(result))

def test_colour_confusion():
    input_dict = {'red': {'red': 0.9}, 'blue': {'blue': 0.7}, 'green': {'green':0.8}}
    result = {'tp':1, 'fn':0, 'fp':2, 'tn':0}
    assert(colour_confusion('red', input_dict) == pytest.approx(result))


def test_colour_confusion2():
    input_dict = {'red': {'red': 0.9}, 'blue': {'blue': 0.3}, 'green': {'green':0.1}}
    result = {'tp':1, 'fn':0, 'fp':0, 'tn':2}
    assert(colour_confusion('red', input_dict) == pytest.approx(result))

def test_colour_confusion3():
    input_dict = {'red': {'red': 0.5}, 'blue': {'blue': 0.5}, 'green': {'green':0.1}}
    result = {'tp':0, 'fn':1, 'fp':0, 'tn':2}
    assert(colour_confusion('red', input_dict) == pytest.approx(result))

def test_colour_confusion4():
    input_dict = {'red': {'red': 0.9, 'maroon':0.4}, 'blue': {'blue': 0.3}, 'green': {'green':0.1}}
    result = {'tp':1, 'fn':1, 'fp':0, 'tn':2}
    assert(colour_confusion('red', input_dict) == pytest.approx(result))