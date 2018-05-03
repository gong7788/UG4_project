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



class PDDLWordl(World):

    def __init__(self, domain_file, problem_file):
        self.domain, self.problem = pddl_functions.parse(domain_file, problem_file)
        self.objects = pddl_functions.get_objects(self.problem)
        self.state = self.problem.initialstate
        self.colours = {o:c for o, c in zip(self.objects, block_plotting.get_colours(objects, state))}

    def update(self, action, args):
        self.state = pddl_functions.apply_action(args, action, self.state)


    def sense(self):
        relations = block_plotting.getPredicates(self.objects, self.state, obscure='True')
        return Observation(self.objects, self.colours, relations)
