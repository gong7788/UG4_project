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
import json
from skimage.color import hsv2rgb


Observation = namedtuple("Observation", ['objects', 'colours', 'state'])
ActionRecord = namedtuple("ActionRecord", ['action', 'args'])

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

    def __init__(self, domain_file='blocks-domain.pddl', problem_directory=None,
                 problem_number=None, problem_file=None, use_hsv=False):

        if domain_file == 'blocks-domain-colour-unknown-cc.pddl':
            domain_file = 'blocks-domain-updated.pddl'
        elif domain_file == 'blocks-domain-colour-unknown.pddl':
            domain_file = 'blocks-domain.pddl'

        self.use_metric_ff = ("updated" in domain_file or "unstack" in domain_file)

        self.settings = {"domain_file":domain_file,
                         "problem_directory":problem_directory,
                         "problem_number":problem_number,
                         "problem_file":problem_file,
                         "use_hsv":use_hsv}

        config = get_config()
        data_dir = Path(config['data_location'])
        self.data_dir = data_dir

        domain_file = data_dir / 'domain' / domain_file
        if problem_file is None:
            problem_file = data_dir / problem_directory / f'problem{problem_number}.pddl'

        self.domain_file = domain_file
        self.problem_file = problem_file

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
        self.num_corrections = 0

        self.tmp = data_dir / 'tmp' / 'world'
        n = len(os.listdir(self.tmp))
        self.tmp_file = os.path.join(self.tmp, f'{n}.pddl')
        with open(self.tmp_file, 'w') as f:
            f.write('this one is mine!')

        self.actions = {action.name: pddl_functions.Action.from_pddl(action)
                        for action in self.domain.actions}
        self.history = []

    def reset(self):
        self.domain, self.problem = pddl_functions.parse(self.domain_file, self.problem_file)
        self.state = pddl_functions.PDDLState.from_problem(self.problem)

    def clean_up(self):
        os.remove(self.tmp_file)

    def update(self, action, args):
        self.history.append(ActionRecord(action, args))
        self.previous_state = copy.deepcopy(self.state)
        self.actions[action].apply_action(self.state, args)
        self.reward += -1

    def remove_top_two(self, tower=None):
        top, second = self.state.get_top_two(tower)
        # unstack = self.actions['unstack']
        self.update("unstack", [top, second, tower])

    def back_track(self, tower="tower0"):
        if 'unstack' in self.actions:
            try:
                plan = self.find_plan()
                action, args = plan[0]
                self.update(action, args)
            except NoPlanError:
                self.remove_top_two(tower)
        else:
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

    def observe_object(self, obj):
        return self.colours[obj]

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
            return plan == [] or plan[0][0] == 'reach-goal'
        except Solved:
            return True
        except (NoPlanError, IDontKnowWhatIsGoingOnError, ImpossibleGoalError):
            return False
        return False

    def test_failure(self):
        try:
            plan = self.find_plan()
        except (NoPlanError, IDontKnowWhatIsGoingOnError, ImpossibleGoalError):
            return True
        except Solved:
            return False
        actions = [action.lower() for action, args in plan]
        if 'unstack' in actions:
            return True
        return False

    def objects_not_in_tower(self):
        out_objects = []

        for o in self.objects:
            if 'tower' in o:
                pass
            elif len(self.state.towers) > 0:
                if all([not(self.state.predicate_holds('in-tower', [o, t.replace('t', 'tower')])) for t in self.state.towers]):
                    out_objects.append(o)
            else:
                if not self.state.predicate_holds('in-tower', [o]):
                    out_objects.append(o)
        return out_objects

    def get_objects_in_tower(self, tower):
        objects = self.state.get_objects_in_tower(tower)


        bottom = tower.replace('tower', 't')

        tower_list = [bottom]

        current = bottom


        while not self.state.predicate_holds("clear", [current]):
            for o in objects:
                if self.state.predicate_holds("on", [o, current]):
                    tower_list.append(o)
                    current = o

        return tower_list


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
