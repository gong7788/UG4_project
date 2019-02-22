from collections import namedtuple
from ..pddl import ff
from ..pddl.ff import Solved, NoPlanError, IDontKnowWhatIsGoingOnError, ImpossibleGoalError
from ..pddl import pddl_functions
from ..pddl import block_plotting
import numpy as np
import os
from ..util.util import get_config
from ..util.colour_dict import colour_names

Observation = namedtuple("Observation", ['objects', 'colours', 'relations', 'state'])

c = get_config()
data_dir = c['data_location']


class World(object):

    def update(self, action, args):
        raise NotImplementedError()
    def sense(self):
        raise NotImplementedError()
    def draw(self):
        raise NotImplementedError()



class PDDLWorld(World):

    def __init__(self, domain_file, problem_file):
        domain_file = os.path.join(data_dir, 'domain', domain_file)
        self.domain, self.problem = pddl_functions.parse(domain_file, problem_file)
        self.domain_file = domain_file
        self.objects = pddl_functions.get_objects(self.problem)
        self.previous_state = None
        self.state = self.problem.initialstate
        self.colours = {o:np.array(c) for o, c in zip(self.objects, block_plotting.get_colours(self.objects, self.state))}
        # next set up conditions for drawing of the current state
        self.start_positions = block_plotting.generate_start_position(self.problem)
        self.reward = 0

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
        block_plotting.plot_blocks(positions, [self.colours[o] for o in objects])

    def to_pddl(self):
        problem = self.problem
        problem.initialstate = self.state
        problem_pddl = problem.asPDDL()
        problem_file_name = os.path.join(data_dir, 'tmp/world_problem.domain')
        with open(problem_file_name, 'w') as f:
            f.write(problem_pddl)
        return self.domain_file, problem_file_name

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


class CNNPDDLWorld(PDDLWorld):

    def __init__(self, domain_file, problem_file, net):
        super(CNNPDDLWorld, self).__init__(domain_file, problem_file)
        
