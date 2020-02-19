from pathlib import Path

from correctingagent.pddl.ff import FailedParseError
from correctingagent.world.rules import ColourCountRule, RedOnBlueRule
from . import world
import random
import os
from correctingagent.world import problem_def
from ..util.colour_dict import colour_dict
from collections import defaultdict
import numpy as np
import json
from .colours import colour_generators


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
            problem = problem_def.BlocksWorldProblem.generate_problem(colours, rules)
            f.write(problem.asPDDL())

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
    if r < 0.4:
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


def generate_biased_dataset(N, rules, directory, colour_dict=colour_dict, use_random_colours=False,
                            bijection=False, domain_name="blocksdomain"):
    directory = Path(directory)

    os.makedirs(directory, exist_ok=True)

    scenarios = []
    while len(scenarios) < N:
        if use_random_colours:
            cc = generate_biased_colour_counts(rules)
            colours = generate_random_colour_from_colour_count(cc)
            colour_object_dict = {f"b{i}": tuple(hsv) for i, (colour, hsv) in enumerate(colours)}
            colours = [colour for colour, hsv in colours]
        else:
            cc = generate_biased_colour_counts(rules)
            colours = generate_from_colour_count(rules, cc)
        print(colours)
        with open('../data/tmp/generation.pddl', 'w') as f:
            problem = problem_def.BlocksWorldProblem.generate_problem(colours, rules, domainname=domain_name).asPDDL()
            f.write(problem)
        domain_file = 'blocks-domain-unstackable.pddl' if domain_name == 'blocksdomain-unstack' else 'blocks-domain.pddl'
        w = world.PDDLWorld(domain_file, problem_file='../data/tmp/generation.pddl')
        try:
            if not w.test_failure():
                file_name = directory / f"problem{len(scenarios)+1}.pddl"
                json_name = directory / f"colours{len(scenarios)+1}.json"
                os.rename('../data/tmp/generation.pddl', file_name)
                if use_random_colours:
                    with open(json_name, 'w') as f:
                        json.dump(colour_object_dict, f)
                scenarios.append(file_name)
                print("added scenario!")
        except FailedParseError as e:
            print(e)

def rules_consistent(rules):
    constrained_colours = [rule.first_obj for rule in rules if rule.constrained_obj == 'first']
    constrained_colours += [rule.second_obj for rule in rules if rule.constrained_obj == 'second']

    other_colours = [rule.second_obj for rule in rules if rule.constrained_obj == 'first']
    other_colours += [rule.first_obj for rule in rules if rule.constrained_obj == 'second']
    for colour in constrained_colours:
        if constrained_colours.count(colour) > 1:
            return False
    for colour in other_colours:
        if colour in constrained_colours:
            # this ensure that bijections are allowed but it is horrible, so sorry
            a = [rule.second_obj for rule in rules if rule.first_obj == colour]
            b = [rule.first_obj for rule in rules if rule.second_obj == colour]
            if len(a) > 0 and len(b) == 0:
                q = [c for c in a if c != a[0]]
                return len(q) == 0
            elif len(b) > 0 and len(a) == 0:
                q = [c for c in b if c != b[0]]
                return len(q) == 0
            else:
                return False
    return True

def generate_bijection(p_primary_colour=1.0):
    c1 = generate_colour(p_primary_colour)
    c2 = generate_colour(p_primary_colour)
    return [problem_def.Ruledef([c1], [c2], 'first'), problem_def.Ruledef([c1], [c2], 'second')]


def generate_dataset_set(N_datasets, N_data, num_rules, dataset_name, colour_dict=colour_dict,
                         p_primary_colour=0.8, use_random_colours=True, create_bijection=False,
                         domain_name="blocksdomain"):
    data_path = '/home/mappelgren/Desktop/correcting-agent/data'
    top_path = os.path.join(data_path, dataset_name)

    try:
        num_datasets = len(os.listdir(top_path))
    except FileNotFoundError:
        num_datasets = 0
    while num_datasets < N_datasets:
        if create_bijection:
            bijections = generate_bijection(p_primary_colour)
        else:
            bijections = []
        rules = bijections + [generate_rule(p_primary_colour=p_primary_colour) for i in range(num_rules - len(bijections))]
        while not rules_consistent(rules):
            rules = bijections + [generate_rule(p_primary_colour=p_primary_colour) for i in range(num_rules - len(bijections))]
        print(rules)
        dataset_path = os.path.join(top_path, f'{dataset_name}{num_datasets}')
        os.makedirs(dataset_path, exist_ok=True)

        generate_biased_dataset(N_data, rules, dataset_path, colour_dict=colour_dict,
                                use_random_colours=use_random_colours, domain_name=domain_name)
        num_datasets = len(os.listdir(top_path))


def generate_colour_count_scenario(rules, num_tower=2):
    less_than_constraints = {}
    bigger_equal_constraint = {}
    for rule in rules:
        if isinstance(rule, ColourCountRule):
            less_than_constraints[rule.colour_name] = rule.number
        elif isinstance(rule, RedOnBlueRule):
            if rule.rule_type == 1:
                bigger_equal_constraint[rule.c2] = rule.c1
            else:
                bigger_equal_constraint[rule.c1] = rule.c2

    colour_counts = {colour: 0 for colour in ['red', 'green', 'blue', 'purple', 'orange', 'pink', 'yellow']}
    for colour, number in less_than_constraints.items():
        if colour not in bigger_equal_constraint.keys() and colour not in bigger_equal_constraint.values():
            colour_counts[colour] = random.choice(range(number, num_tower*number + 1))
            while sum(colour_counts.values()) > 10:
                colour_counts[colour] = random.choice(range(number, num_tower*number + 1))
        elif colour in bigger_equal_constraint.keys() and colour in bigger_equal_constraint.values():
            colour2 = bigger_equal_constraint[colour]
            colour_counts[colour] = random.choice(range(number, num_tower*number + 1))
            colour_counts[colour2] = colour_counts[colour]
            while sum(colour_counts.values()) > 10:
                colour_counts[colour] = random.choice(range(number, num_tower*number + 1))
                colour_counts[colour2] = colour_counts[colour]
        elif colour in bigger_equal_constraint.keys():
            colour2 = bigger_equal_constraint[colour]
            colour_counts[colour] = random.choice(range(number, num_tower*number + 1))
            colour_counts[colour2] = random.choice(range(colour_counts[colour] + 1))
            while sum(colour_counts.values()) > 10:
                colour_counts[colour] = random.choice(range(number, num_tower*number + 1))
                colour_counts[colour2] = random.choice(range(colour_counts[colour] + 1))
        else:
            colour2 = [key for key, value in bigger_equal_constraint.items() if value == colour][0]
            colour_counts[colour] = random.choice(range(number, num_tower*number + 1))
            colour_counts[colour2] = random.choice(range(colour_counts[colour], colour_counts[colour]+3))
            while sum(colour_counts.values()) > 10:
                colour_counts[colour] = random.choice(range(number, num_tower*number + 1))
                colour_counts[colour2] = random.choice(range(colour_counts[colour], 10))
    while sum(colour_counts.values()) < 10:
        colour = random.choice(list(colour_counts.keys()))
        if colour not in less_than_constraints.keys() and colour not in bigger_equal_constraint.keys() and colour not in bigger_equal_constraint.values():
            colour_counts[colour] += 1
    return colour_counts


def generate_biased_dataset_w_colour_count(N, rules, directory, use_random_colours=True,
                                           domain_name="blocksworld-updated", num_towers=2):
    directory = Path(directory)

    os.makedirs(directory, exist_ok=True)

    scenarios = []
    while len(scenarios) < N:
        if use_random_colours:
            cc = generate_colour_count_scenario(rules)
            colours = generate_random_colour_from_colour_count(cc)
            colour_object_dict = {f"b{i}": tuple(hsv) for i, (colour, hsv) in enumerate(colours)}
            colours = [colour for colour, hsv in colours]
        else:
            cc = generate_biased_colour_counts(rules)
            colours = generate_from_colour_count(rules, cc)

        with open('../data/tmp/generation.pddl', 'w') as f:
            problem = problem_def.ExtendedBlocksWorldProblem.generate_problem(colours, rules, domainname=domain_name,
                                                                              num_towers=num_towers).asPDDL()
            f.write(problem)
        domain_file = 'blocks-domain-unstackable.pddl' if domain_name == 'blocksworld-unstack' \
            else 'blocks-domain-updated.pddl'
        w = world.PDDLWorld(domain_file, problem_file='../data/tmp/generation.pddl')
        if not w.test_failure():
            file_name = directory / f"problem{len(scenarios)+1}.pddl"
            json_name = directory / f"colours{len(scenarios)+1}.json"
            os.rename('../data/tmp/generation.pddl', file_name)
            if use_random_colours:
                with open(json_name, 'w') as f:
                    json.dump(colour_object_dict, f)
            scenarios.append(file_name)


def generate_colour_count(max_num=4, exact_num=None):
    colour = random.choice(list(colour_dict.keys()))
    if exact_num is not None:
        number = exact_num
    else:
        number = random.choice(range(1, max_num+1))
    return ColourCountRule(colour, number)


def generate_consistent_red_on_blue(rules):
    rule = random.choice(rules)
    colour = rule.colour_name
    colour2 = random.choice(list(colour_dict.keys()))
    while colour == colour2:
        colour2 = random.choice(list(colour_dict.keys()))
    rule_type = random.choice([1,2])
    return RedOnBlueRule(colour, colour2, rule_type)


def generate_dataset_set_w_colour_count(N_datasets, N_data, num_colour_count,
                                        num_redonblue, dataset_name,
                                        colour_dict=colour_dict, use_random_colours=True,
                                        cc_num=2, cc_exact_num=None, num_towers=2,
                                        domain_name="blocksdomain-updated"):
    data_path = Path('/home/mappelgren/Desktop/correcting-agent/data')
    top_path = data_path / dataset_name
    try:
        num_datasets = len(os.listdir(top_path))
    except FileNotFoundError:
        num_datasets = 0

    while num_datasets < N_datasets:

        colour_count = [generate_colour_count(cc_num, cc_exact_num) for i in range(num_colour_count)]
        colours = [rule.colour_name for rule in colour_count]
        while len(set(colours)) != len(colours):
            colour_count = [generate_colour_count(cc_num, cc_exact_num) for i in range(num_colour_count)]
            colours = [rule.colour_name for rule in colour_count]
        if len(colour_count) > 0:
            red_on_blue = [generate_consistent_red_on_blue(colour_count) for i in range(num_redonblue)]
        else:
            red_on_blue = [generate_rule(p_primary_colour=1.0) for i in range(num_redonblue)]
        rules = colour_count + red_on_blue

        dataset_path = top_path / f'{dataset_name}{num_datasets}'
        os.makedirs(dataset_path, exist_ok=True)

        generate_biased_dataset_w_colour_count(N_data, rules, dataset_path,
                                               use_random_colours=use_random_colours, domain_name=domain_name,
                                               num_towers=num_towers)
        num_datasets = len(os.listdir(top_path))


