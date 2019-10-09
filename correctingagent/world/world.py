import copy
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


Observation = namedtuple("Observation", ['objects', 'colours', 'state'])


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

    def __init__(self, domain_file='blocks-domain.pddl', problem_directory=None, problem_number=None, problem_file=None, use_hsv=False):

        self.use_metric_ff = "updated" in domain_file

        config = get_config()
        data_dir = Path(config['data_location'])
        self.data_dir = data_dir

        domain_file = data_dir / 'domain' / domain_file
        if problem_file is None:
            problem_file = data_dir / problem_directory / f'problem{problem_number}.pddl'

        self.domain, self.problem = pddl_functions.parse(domain_file, problem_file)
        self.state = pddl_functions.PDDLState.from_problem(self.problem)
        self.domain_file = domain_file
        self.objects = pddl_functions.get_objects(self.problem)
        self.previous_state = None
        #self.state = self.problem.initialstate
        self.use_hsv = use_hsv
        self.colours = self.state.get_colours(use_hsv=use_hsv)

        # next set up conditions for drawing of the current state
        self.start_positions = block_plotting.generate_start_position(self.problem)
        self.reward = 0

        self.tmp = data_dir / 'tmp' / 'world'
        n = len(os.listdir(self.tmp))
        self.tmp_file = os.path.join(self.tmp, f'{n}.pddl')
        with open(self.tmp_file, 'w') as f:
            f.write('this one is mine!')

        self.actions = {action.name: pddl_functions.Action.from_pddl(action) for action in self.domain.actions}

    def clean_up(self):
        os.remove(self.tmp_file)

    def update(self, action, args):
        self.previous_state = copy.deepcopy(self.state)
        self.actions[action].apply_action(self.state, args)
        self.reward += -1

    def back_track(self):
        self.state = self.previous_state
        self.previous_state = None
        self.reward += -1

    def sense(self, obscure=True):
        # relations = block_plotting.get_predicates(self.objects, self.state, obscure=obscure)
        if obscure:
            obscured_state = self.state.obscure_state()
        else:
            obscured_state = copy.deepcopy(self.state)

        return Observation(self.objects, self.colours, obscured_state)

    def draw(self, debug=False):

        positions = block_plotting.place_objects(self.objects, self.state, self.start_positions)
        if debug:
            print(positions)
        objects = [obj for obj in self.objects if 't' not in obj]
        if debug:
            print(objects)
        if self.use_hsv:
            block_plotting.plot_blocks(positions, [hsv2rgb([[self.colours[o]]])[0][0] for o in objects])
        else:
            block_plotting.plot_blocks(positions, [self.colours[o] for o in objects])

    def asPDDL(self):
        problem = self.problem
        problem.initialstate = self.state.to_formula()
        problem_pddl = problem.asPDDL()
        with open(self.tmp_file, 'w') as f:
            f.write(problem_pddl)
        return self.domain_file, self.tmp_file

    def find_plan(self):
        domain, problem = self.asPDDL()
        return ff.run(domain, problem, use_metric_ff=self.use_metric_ff)

    def test_success(self):
        try:
            plan = self.find_plan()
            return plan == []
        except Solved:
            return True
        except (NoPlanError, IDontKnowWhatIsGoingOnError, ImpossibleGoalError):
            return False
        return False

    def test_failure(self):
        try:
            self.find_plan()
        except (NoPlanError, IDontKnowWhatIsGoingOnError, ImpossibleGoalError):
            return True
        except Solved:
            return False
        return False

    def objects_not_in_tower(self):
        out_objects = []
        for o in self.objects:
            if not self.state._predicate_holds(pddl_functions.Predicate('in-tower', [o])):
                out_objects.append(o)
        return out_objects


class RandomColoursWorld(PDDLWorld):
    def __init__(self, domain_file='blocks-domain.pddl', problem_directory=None, problem_number=None, problem_file=None, use_hsv=False):
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
