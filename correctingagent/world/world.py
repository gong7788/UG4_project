from collections import namedtuple
from pathlib import Path

from ..pddl import ff
from ..pddl.ff import Solved, NoPlanError, IDontKnowWhatIsGoingOnError, ImpossibleGoalError
from ..pddl import pddl_functions
from ..pddl import block_plotting
import numpy as np
import os
from ..util.util import get_config
from ..util.colour_dict import colour_names
import json
from skimage.color import rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt


Observation = namedtuple("Observation", ['objects', 'colours', 'relations', 'state'])

c = get_config()
data_dir = Path(c['data_location'])


def get_world(problem_name, problem_number, domain_file='blocks-domain.pddl', world_type='PDDL', use_hsv=False):
    world_types = {'PDDL': PDDLWorld,
                   'RandomColours': RandomColoursWorld}
    return world_types[world_type](domain_file, problem_directory=problem_name,
                                   problem_number=problem_number, use_hsv=use_hsv)


class World(object):

    def update(self, action, args):
        raise NotImplementedError()

    def sense(self):
        raise NotImplementedError()

    def draw(self):
        raise NotImplementedError()


class PDDLWorld(World):

    def __init__(self, domain_file, problem_directory=None, problem_number=None, problem_file=None, use_hsv=False):
        config = get_config()
        data_dir = Path(config['data_location'])
        self.data_dir = data_dir

        domain_file = data_dir / 'domain' / domain_file
        if problem_file is None:
            problem_file = data_dir / problem_directory / f'problem{problem_number}.pddl'

        self.domain, self.problem = pddl_functions.parse(domain_file, problem_file)
        self.domain_file = domain_file
        self.objects = pddl_functions.get_objects(self.problem)
        self.previous_state = None
        self.state = self.problem.initialstate
        self.use_hsv = use_hsv
        if use_hsv:
            self.colours = {o: rgb2hsv([[np.array(c)]])[0][0] for o, c in
                            zip(self.objects, block_plotting.get_colours(self.objects, self.state))}
        else:
            self.colours = {o:np.array(c) for o, c in zip(self.objects, block_plotting.get_colours(self.objects, self.state))}
        # next set up conditions for drawing of the current state
        self.start_positions = block_plotting.generate_start_position(self.problem)
        self.reward = 0

        self.tmp = data_dir / 'tmp' / 'world'
        n = len(os.listdir(self.tmp))
        self.tmp_file = os.path.join(self.tmp, '{}.pddl'.format(n))
        with open(self.tmp_file, 'w') as f:
            f.write('this one is mine!')

    def clean_up(self):
        os.remove(self.tmp_file)

    def update(self, action, args):
        actions = pddl_functions.create_action_dict(self.domain)
        self.previous_state = self.state
        self.state = pddl_functions.apply_action(args, actions[action], self.state)
        self.reward += -1

    def back_track(self):
        self.state = self.previous_state
        self.previous_state = None
        self.reward += -1

    def sense(self, obscure=True):
        relations = block_plotting.get_predicates(self.objects, self.state, obscure=obscure)
        if obscure:
            obscured_state = pddl_functions.obscure_state(self.state, colour_names)
        else:
            obscured_state = self.state
        return Observation(self.objects, self.colours, relations,  obscured_state)

    def get_actions(self):
        return self.domain.actions

    def draw(self):

        positions = block_plotting.place_objects(self.objects, self.state, self.start_positions)

        objects = pddl_functions.filter_tower_locations(self.objects, get_locations=False)
        if self.use_hsv:
            block_plotting.plot_blocks(positions, [hsv2rgb([[self.colours[o]]])[0][0] for o in objects])
        else:
            block_plotting.plot_blocks(positions, [self.colours[o] for o in objects])

    def to_pddl(self):
        problem = self.problem
        problem.initialstate = self.state
        problem_pddl = problem.asPDDL()
        #problem_file_name = os.path.join(data_dir, 'tmp/world_problem.domain')
        with open(self.tmp_file, 'w') as f:
            f.write(problem_pddl)
        return self.domain_file, self.tmp_file

    def test_success(self):
        domain, problem = self.to_pddl()
        try:
            ff.run(domain, problem)
        except Solved:
            return True
        except (NoPlanError, IDontKnowWhatIsGoingOnError, ImpossibleGoalError):
            return False
        return False

    def test_failure(self):
        domain, problem = self.to_pddl()
        try:
            ff.run(domain, problem)
        except (NoPlanError, IDontKnowWhatIsGoingOnError, ImpossibleGoalError):
            return True
        except Solved:
            return False
        return False

    def objects_not_in_tower(self):
        observation = self.sense()
        objects = observation.objects
        rels = observation.relations

        out_objects = []

        for o in objects:
            if 'in-tower' not in rels[o]:
                out_objects.append(o)
        return out_objects


class RandomColoursWorld(PDDLWorld):
    def __init__(self, domain_file, problem_directory=None, problem_number=None, problem_file=None, use_hsv=False):
        super().__init__(domain_file, problem_directory=problem_directory,
                         problem_number=problem_number, problem_file=problem_file, use_hsv=use_hsv)
        colour_file = self.data_dir / problem_directory / f'colours{problem_number}.json'
        self.colours = self.load_colours(colour_file)

    def load_colours(self, colour_file):
        colours = json.load(open(colour_file))
        if not self.use_hsv:
            colours = {o: np.array(hsv2rgb([[colour]])[0][0]) for o, colour in colours.items()}
        else:
            colours = {o: np.array(colour) for o, colour in colours.items()}
        return colours


class CNNPDDLWorld(PDDLWorld):

    def __init__(self, domain_file, problem_file, net):
        super(CNNPDDLWorld, self).__init__(domain_file, problem_file)
