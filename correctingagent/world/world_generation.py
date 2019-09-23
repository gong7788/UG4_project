from pathlib import Path

from . import world
import random
import os
from ..pddl import problem_def
from ..util.colour_dict import colour_dict
from collections import defaultdict
import numpy as np
import json


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


def sample_colour(colour_dict):
    category = random.choice(list(colour_dict.keys()))
    return random.choice(colour_dict[category])


def sample_colours(n, colour_dict=colour_dict):
    return [sample_colour(colour_dict) for i in range(n)]


def sample_from_category(category):
    try:
        choices = colour_dict[category]
        return random.choice(choices)
    except KeyError:
        return category


def generate_dataset(N, rules, directory, colour_dict=colour_dict):
    os.makedirs(directory, exist_ok=True)
    dataset = []
    while len(dataset) < N:
        colours = sample_colours(10, colour_dict=colour_dict)
        with open('tmp/generation.pddl', 'w') as f:
            f.write(problem_def.create_problem(colours, rules).asPDDL())
        w = world.PDDLWorld('blocks-domain.pddl', 'tmp/generation.pddl')
        if not w.test_failure():
            file_name = '{}/problem{}.pddl'.format(directory,len(dataset)+1)
            os.rename('tmp/generation.pddl', file_name)
            dataset.append(file_name)


class Constraint(object):

    def __init__(self, rule):
        c1 = rule.first_obj[0]
        c2 = rule.second_obj[0]


        if rule.constrained_obj == 'first':
            self.str = "{} <= {}".format(c1, c2)
            self.smaller = c1
            self.bigger = c2
        else:
            self.str = "{} <= {}".format(c2, c1)
            self.smaller = c2
            self.bigger = c1

    def test_constraint(self, colour_counts):
        return colour_counts[self.smaller] <= colour_counts[self.bigger]


def generate_colour_counts(colour_dict=colour_dict):
    colour_counts = {category:0 for category in colour_dict.keys()}
    for i in range(10):
        colour_counts[random.choice(list(colour_counts.keys()))] += 1
    return colour_counts


def generate_biased_sample(rules, colour_dict=colour_dict):
    important_colours = set()
    colour_count = {category:0 for category in colour_dict.keys()}
    for rule in rules:
        c1 = rule.first_obj[0]
        c2 = rule.second_obj[0]
        important_colours.add(c1)
        important_colours.add(c2)
    important_colours = list(important_colours)

    r = random.random()
    if r < 0.2:
        c = random.choice(list(colour_dict.keys()))
        return c
    else:
        return random.choice(important_colours)


def generate_full_sample(N, rules):
    colour_count = defaultdict(int)
    #colour_count = {category:0 for category in colour_dict.keys()}
    for i in range(N):
        c = generate_biased_sample(rules)
        colour_count[c] += 1
    return colour_count


def generate_biased_colour_counts(rules, colour_dict=colour_dict):
    colour_count = generate_full_sample(10, rules)
    while not verify_colour_counts(rules, colour_count):
        colour_count = generate_full_sample(10, rules)
    return colour_count


def verify_colour_counts(rules, colour_counts):
    constraints = [Constraint(rule) for rule in rules]

    return all([constraint.test_constraint(colour_counts) for constraint in constraints])


def generate_from_colour_count(rules, colour_count):
    colours = []
    for colour, count in colour_count.items():
        for i in range(count):
            if colour in colour_dict.keys():
                colours.append(sample_from_category(colour))
            else:
                colours.append(colour)
    random.shuffle(colours)
    return colours

def generate_random_colour_from_colour_count(colour_count):
    colours = []
    for colour, count in colour_count.items():
        for i in range(count):
            colours.append((colour, colour_generators[colour]()))

    random.shuffle(colours)
    return colours

def generate_colour(p_primary_colour=0.8):
    if np.random.random() < p_primary_colour:
        return np.random.choice(list(colour_dict.keys()))
    else:
        return sample_colour(colour_dict)


def generate_rule(p_primary_colour=0.8):
    c1 = generate_colour(p_primary_colour)
    c2 = generate_colour(p_primary_colour)
    while True:
        try:
            bad = c1 in colour_dict[c2]
        except KeyError:
            bad = False
        try:
            bad = c2 in colour_dict[c1] or bad
        except KeyError:
            pass

        if c1 == c2 or bad:
            c2 = generate_colour(p_primary_colour)
        else:
            break
    direction = 'first' if np.random.random() < 0.5 else 'second'

    return problem_def.Ruledef([c1], [c2], direction)


def generate_biased_dataset(N, rules, directory, colour_dict=colour_dict, use_random_colours=False):
    directory = Path(directory)

    os.makedirs(directory, exist_ok=True)
    dataset = []

    while len(dataset) < N:
        if use_random_colours:
            cc = generate_biased_colour_counts(rules)
            colours = generate_random_colour_from_colour_count(cc)
            colour_object_dict = {f"b{i}": tuple(hsv) for i, (colour, hsv) in enumerate(colours)}
            colours = [colour for colour, hsv in colours]
        else:
            cc = generate_biased_colour_counts(rules)
            colours = generate_from_colour_count(rules, cc)

        with open('../data/tmp/generation.pddl', 'w') as f:
            f.write(problem_def.create_problem(colours, rules).asPDDL())
        w = world.PDDLWorld('blocks-domain.pddl', problem_file='../data/tmp/generation.pddl')
        if not w.test_failure():
            file_name = directory / f"problem{len(dataset)+1}.pddl"
            json_name = directory / f"colours{len(dataset)+1}.json"
            os.rename('../data/tmp/generation.pddl', file_name)
            if use_random_colours:
                with open(json_name, 'w') as f:
                    json.dump(colour_object_dict, f)
            dataset.append(file_name)




def generate_dataset_set(N_datasets, N_data, num_rules, dataset_name, colour_dict=colour_dict, p_primary_colour=0.8):
    data_path = '/home/mappelgren/Desktop/correcting-agent/data'
    top_path = os.path.join(data_path, dataset_name)
    rules = [generate_rule() for i in range(num_rules)]
    for i in range(N_datasets):
        dataset_path = os.path.join(top_path, '{}{}'.format(dataset_name, i))
        os.makedirs(dataset_path, exist_ok=True)

        generate_biased_dataset(N_data, rules, dataset_path, colour_dict=colour_dict)

