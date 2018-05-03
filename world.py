from collections import namedtuple
import ff
import pddl_functions
import block_plotting

Observation = namedtuple("Observation", ['objects', 'colours', 'relations'])

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
        self.objects = pddl_functions.get_objects(self.problem)
        self.state = self.problem.initialstate
        self.colours = {o:c for o, c in zip(self.objects, block_plotting.get_colours(self.objects, self.state))}
        # next set up conditions for drawing of the current state
        self.start_positions = block_plotting.generate_start_position(self.problem)

    def update(self, action, args):
        self.state = pddl_functions.apply_action(args, action, self.state)


    def sense(self):
        relations = block_plotting.get_predicates(self.objects, self.state, obscure='True')
        return Observation(self.objects, self.colours, relations)

    def get_actions(self):
        return self.domain.actions

    def draw(self):
        positions = block_plotting.place_objects(self.objects, self.state, self.start_positions)
        block_plotting.plot_blocks(positions, [self.colours[o] for o in self.objects])
