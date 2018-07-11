from collections import namedtuple
import ff
from ff import Solved, NoPlanError
import pddl_functions
import block_plotting
import numpy as np
Observation = namedtuple("Observation", ['objects', 'colours', 'relations', 'state'])
from colour_dict import colour_names

class World(object):

    def update(self, action, args):
        raise NotImplementedError()
    def sense(self):
        raise NotImplementedError()
    def draw(self):
        raise NotImplementedError()



class PDDLWorld(World):

    def __init__(self, domain_file, problem_file):
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

    def sense(self):
        relations = block_plotting.get_predicates(self.objects, self.state, obscure='True')
        obscured_state = pddl_functions.obscure_state(self.state, colour_names)
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
        problem_file_name = 'tmp/world_problem.pddl'
        with open(problem_file_name, 'w') as f:
            f.write(problem_pddl)
        return self.domain_file, problem_file_name

    def test_success(self):
        domain, problem = self.to_pddl()
        try:
            ff.run(domain, problem)
        except Solved:
            return True
        except NoPlanError:
            return False
        return False

    def test_failure(self):
        domain, problem = self.to_pddl()
        try:
            ff.run(domain, problem)
        except NoPlanError:
            return True
        except Solved:
            return False
        return False
