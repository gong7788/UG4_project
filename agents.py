import copy
import ff
import dmrs_functions

class Agent(object):


    def plan(self):
        raise NotImplementedError()

    def act(self, action):
        raise NotImplementedError()



class CorrectingAgent(Agent):

    def __init__(self, world, domain_file='blocks-domain.pddl'):
        self.world = world
        self.domain = world.domain
        self.domain_file = domain_file
        observation = world.sense()
        self.problem = copy.deepcopy(world.problem)
        self.problem.initialstate = observation.state

    def plan(self):
        with open('tmp/problem.pddl', 'w') as f:
            f.write(self.problem.asPDDL())

        plan = ff.run(self.domain_file, 'tmp/problem.pddl')
        return plan

    def get_correction(self, user_input):
        message = sent_to_tripple


    def build_model(self):
        pass
