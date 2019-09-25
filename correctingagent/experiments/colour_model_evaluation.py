
from ..world.colours import colour_generators, get_colour, name_to_hsv, name_to_rgb
from ..models import prob_model
from skimage.color import hsv2rgb, rgb2hsv
from ..util.colour_dict import colour_dict
from ..util.util import get_config
from collections import defaultdict
import pickle


def train_cms_new_colours(use_negative=False, use_hsv=False, iterations=1, **kwargs):
    """ Trains colour models for all colours included in colour_generators

    :param use_negative: whether to use negative samples to train the model
    :param use_hsv: whether to use rgb or hsv
    :param iterations: number of training examples per class
    :param kwargs: additional argumets passed to the KDE colour model
    :return: dictionary of colour:colour_model
    """
    cms = {}

    for colour_category in colour_generators.keys():
        cm = prob_model.KDEColourModel(colour_category, **kwargs)
        cms[colour_category] = cm
        for second_colour, generator in colour_generators.items():
            for i in range(iterations):
                colour_data = generator()
                if not use_hsv:
                    colour_data = hsv2rgb([[colour_data]])[0][0]
                if colour_category == second_colour:
                    cm.update(colour_data, 1)
                elif use_negative:
                    cm.update_negative(colour_data, 1)
    return cms


def train_cms(use_negative=False, use_hsv=False, iterations=1, **kwargs):
    """ Trains colour models for all primary colours in colour_dict using the colours definedi n colour dict

    :param use_negative: whether to update for negative examples
    :param use_hsv: whether to use hsv or rgb
    :param iterations: whether to repeat the sequence of colours
    :param kwargs: args passed to KDE colour model
    :return: dictionary of colour:colour_model
    """
    cms = {}

    for colour_category in colour_dict.keys():
        cm = prob_model.KDEColourModel(colour_category, **kwargs)
        cms[colour_category] = cm
        for second_colour in colour_dict.keys():
            for i in range(iterations):
                for specific_colour in colour_dict[second_colour]:

                    colour_data = get_colour(specific_colour)
                    if use_hsv:
                        colour_data = rgb2hsv([[colour_data]])[0][0]
                    if colour_category == second_colour:
                        cm.update(colour_data, 1)

                    elif use_negative:
                        cm.update_negative(colour_data, 1)

    return cms


def colour_probs(colour_model, prior=0.5, use_hsv=False):
    """ Caluclates the output probability of colour_model for 10 random samples of each colour from colour_generators

    :param colour_model: ColourModel to be evaluated
    :param use_hsv: whether to use HSV or RGB
    :return: dict of dicts containing output of classifiers
    """
    results_dict = defaultdict(dict)
    for c, generator in colour_generators.items():
        for i in range(10):
            data = generator()
            if not use_hsv:
                data = hsv2rgb([[data]])[0][0]

            p_c = colour_model.p(1, data)
            results_dict[c][i] = p_c

    return results_dict


def evaluate_colour_model(colour_model, prior=0.5, threshold=0.5, use_hsv=False):
    """print the confusion matrix for the specified colour model on 10 random samples from each colour generator

    :param colour_model: ColourModel
    :param prior: P(C=1)
    :param threshold: decision threshold
    :param use_hsv: whether HSV is used
    :return: None
    """
    results_dict = colour_probs(colour_model, prior=prior, use_hsv=use_hsv)
    print_confusion(colour_confusion(colour_model.name, results_dict, threshold=threshold), colour_model.name[0])


def evaluate_cms(threshold=0.5, prior=0.5, use_hsv=False, use_new_colours=False, **kwargs):
    """ Train and evaluate colour models

    :param threshold: decision boundary
    :param prior: prior for p(c=1)
    :param use_hsv: whether to use hsv or rgb
    :param use_new_colours: whether to use randomly generated colours or colour_dict colours
    :param kwargs: args for train_cms_new_colours (mainly KDE)
    :return: trained colour models
    """
    if use_new_colours:
        cms = train_cms_new_colours(use_hsv=use_hsv, **kwargs)
    else:
        cms = train_cms(use_hsv=use_hsv, **kwargs)
    for colour in colour_dict.keys():
        results_dict = colour_probs(cms[colour], prior=prior, use_hsv=use_hsv)
        print(colour)
        print_confusion(colour_confusion(colour, results_dict, threshold=threshold), colour[0])
    return cms


def load_agent(dataset, threshold=0.7, file_modifiers=''):
    locations = get_config()
    with open('{}/agents/correcting_{}_{}{}.pickle'.format(locations['results_location'], dataset, threshold, file_modifiers), 'rb') as f:
        agent = pickle.load(f)
    return agent


def colour_probs_w_colourdict(colour_model, colour_dict=colour_dict, prior=0.5, use_hsv=False):
    results_dict = {c:{c_i:-1 for c_i in cs} for c, cs in colour_dict.items()}
    for c, cs in colour_dict.items():
        for c_i in cs:
            if use_hsv:
                p_c = colour_model.p(1, name_to_hsv(c_i))
            else:
                p_c = colour_model.p(1, name_to_rgb(c_i))
            results_dict[c][c_i] = p_c
    return results_dict


def colour_confusion(colour, results_dict, threshold=0.5):
    output = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    for c, cs in results_dict.items():
        for p in cs.values():
            if c == colour and p > threshold:
                output['tp'] += 1
            elif c == colour:
                output['fn'] += 1
            elif p > threshold:
                output['fp'] += 1
            else:
                output['tn'] += 1
    return output


def test_colour_model_w_colourdict(colour_model, colour_dict=colour_dict, colour_thresh=0.5, pretty_printing=True):
    probs = colour_probs(colour_model, colour_dict)
    confusion = colour_confusion(colour_model.name, probs, colour_thresh)
    colour_initial = colour_model.name[0].upper()
    if pretty_printing:
        print_confusion(confusion, colour_initial)
    return confusion


def print_confusion(confusion_dict, colour_initial):
    print('True Label  {ci}=1 {ci}=0'.format(ci=colour_initial))
    print('Predict {ci}=1| {tp} | {fp} |'.format(ci=colour_initial, **confusion_dict))
    print('        {ci}=0| {fn} | {tn} |'.format(ci=colour_initial, **confusion_dict))


def plot_colours(dataset, threshold=0.7, file_modifiers='', colour_dict=colour_dict, colour_thresh=0.5):
    agent = load_agent(dataset, threshold=threshold, file_modifiers=file_modifiers)
    for cm in agent.colour_models.values():
        probs = colour_probs(cm, colour_dict, prior=0.5)
        confusion = colour_confusion(cm.name, probs, colour_thresh)
        print(cm.name, confusion)
        cm.draw(save_location_basename=dataset)
