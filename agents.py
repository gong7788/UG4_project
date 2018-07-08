import copy
import ff
import dmrs_functions
from collections import namedtuple
import goal_updates
import prob_model
import pddl_functions
from ff import NoPlanError

Message = namedtuple('Message', ['rel', 'o1', 'o2', 'T', 'o3'])




class Agent(object):


    def plan(self):
        raise NotImplementedError()

    def act(self, action):
        raise NotImplementedError()

def read_sentence(sentence, use_dmrs=True):
    sentence = sentence.lower().strip('no,')
    o3 = None
    T = 'tower'
    if 'tower because you must' in sentence:
        T = 'table'
        o3 = sentence.split('put')[1].split('in')[0].strip()
        sentence = sentence.split('because you must')[1].strip()
    if use_dmrs:
        rel, o1, o2= dmrs_functions.sent_to_tripple(sentence)
        o1 = dmrs_functions.get_adjectives(o1)
        o2 = dmrs_functions.get_adjectives(o2)
        rel = dmrs_functions.get_pred(rel['predicate'])
        return Message(rel, o1, o2, T, o3)
    else:
        rel = 'on'
        o1, o2 = sentence.strip('no, ').strip('put').split('on')
        o1 = o1.strip().split()[0]
        o2 = o2.strip().split()[0]
        return Message(rel, [o1], [o2], T, o3)


class CorrectingAgent(Agent):

    def __init__(self, world, colour_models = {}, rule_beliefs = {}, domain_file='blocks-domain.pddl', teacher=None):
        self.world = world
        self.domain = world.domain
        self.domain_file = domain_file
        observation = world.sense()
        self.problem = copy.deepcopy(world.problem)
        self.goal = goal_updates.create_default_goal()#self.problem.goal # change this to ``forall x in-tower(x)
        self.tmp_goal = None
        self.problem.initialstate = observation.state
        self.colour_models = colour_models
        self.rule_beliefs = rule_beliefs
        self.threshold = 0.7
        self.tau = 0.6
        self.rule_models = {}
        self.teacher = teacher


    def new_world(self, world):
        self.world = world
        observation = world.sense()
        self.problem = copy.deepcopy(world.problem)
        self.tmp_goal = None
        self.problem.initialstate = observation.state


    def plan(self):
        self.problem.goal = goal_updates.update_goal(self.goal,self.tmp_goal)
        self.sense()
        with open('tmp/problem.pddl', 'w') as f:
            f.write(self.problem.asPDDL())

        try:
            plan = ff.run(self.domain_file, 'tmp/problem.pddl')
        except NoPlanError:
            self.problem.goal = goal_updates.update_goal(goal_updates.create_default_goal(), self.tmp_goal)
            with open('tmp/problem.pddl', 'w') as f:
                f.write(self.problem.asPDDL())
            plan = ff.run(self.domain_file, 'tmp/problem.pddl')
        return plan

    def _print_goal(self):
        print(self.goal.asPDDL())

    def get_correction(self, user_input, action, args):
        not_on_xy = pddl_functions.create_formula('on', args, op='not')
        self.tmp_goal = goal_updates.update_goal(self.tmp_goal, not_on_xy)
        message = read_sentence(user_input, use_dmrs=False)
        rule_model, rules = self.build_model(message)

        if rule_model.rules in self.rule_models.keys():
            rule_model = rule_model

        data = self.get_data(message, args)

        rule_beliefs = rule_model.update_belief_r(data)
        print(rule_beliefs)

        self.rule_beliefs[rule_model.rules] = rule_beliefs
        if rule_beliefs[0] > self.threshold:
            self.goal = goal_updates.update_goal(self.goal, rules[0])


        elif rule_beliefs[1] > self.threshold:
            self.goal = goal_updates.update_goal(self.goal, rules[1])

        else:
            question = 'Is the top object {}?'.format(message.o1[0])
            print("R:", question)
            answer = self.teacher.answer_question(question, self.world)
            print("T:", answer)

            bin_answer = int(answer.lower() == 'yes')
            rule_beliefs = rule_model.update_belief_r(data, visible={message.o1[0]:bin_answer})
            #print(rule_beliefs)
            self.rule_beliefs[rule_model.rules] = rule_beliefs
            if rule_beliefs[0] > self.threshold:
                self.goal = goal_updates.update_goal(self.goal, rules[0])


            elif rule_beliefs[1] > self.threshold:
                self.goal = goal_updates.update_goal(self.goal, rules[1])


        rule_model.update_c(data)
        self.world.back_track()
        self.sense()
        self.rule_models[rule_model.rules] = rule_model


    def no_correction(self, action, args):
        for rule_model in self.rule_models.values():
            message = read_sentence('no, put {} blocks on {} blocks'.format(rule_model.c1.name, rule_model.c2.name), use_dmrs=False)
            data = self.get_data(message, args)
            rule_model.update_c_no_corr(data)

    def get_data(self, message, args):
        observation = self.sense()

        c1 = message.o1[0]
        c2 = message.o2[0]
        c3 = '{}/{}'.format(c1, c2)
        o1 = args[0]
        o2 = args[1]
        o3 = message.o3
        colour_data = observation.colours
        try:
            if 't' not in o2:
                data_dict = {c1:colour_data[o1], c2:colour_data[o2], c3:colour_data[o3]}
            else:
                data_dict = {c1:colour_data[o1], c2:None, c3:colour_data[o3]}
        except KeyError:
            if 't' not in o2:
                data_dict = {c1:colour_data[o1], c2:colour_data[o2]}
            else:
                data_dict = {c1:colour_data[o1], c2:None}
        return data_dict


    def build_model(self, message):
        rules = goal_updates.create_goal_options(message.o1, message.o2)
        rule_names = tuple(map(lambda x: x.asPDDL(), rules))

        if rule_names in self.rule_beliefs.keys():
            rule_probs = self.rule_beliefs[rule_names] # this will probably have to change
        else:
            rule_probs = (0.5, 0.5)

        c1 = message.o1[0]
        c2 = message.o2[0]
        try:
            colour_model1 = self.colour_models[c1]
        except KeyError:
            colour_model1 = prob_model.ColourModel(c1)
            self.colour_models[c1] = colour_model1
        try:
            colour_model2 = self.colour_models[c2]
        except KeyError:
            colour_model2 = prob_model.ColourModel(c2)
            self.colour_models[c2] = colour_model2

        if message.T == 'tower':
            rule_model = prob_model.CorrectionModel(rule_names, colour_model1, colour_model2, rule_belief=rule_probs)

        else:
            rule_model = prob_model.TableCorrectionModel(rule_names, colour_model1, colour_model2, rule_belief=rule_probs)
        return rule_model, rules


    def update_goal(self):
        pass

    def sense(self):
        observation = self.world.sense()
        self.problem.initialstate = observation.state

        for colour, model in self.colour_models.items():
            for obj in pddl_functions.filter_tower_locations(observation.objects, get_locations=False): # these objects include tower locations, which they should not
                data = observation.colours[obj]
                p_colour = model.p(1, data)
                if p_colour > self.tau:
                    colour_formula = pddl_functions.create_formula(colour, [obj])
                    self.problem.initialstate.append(colour_formula)

        return observation

    def act(self, action, args):
        self.world.update(action, args)
        self.sense()
