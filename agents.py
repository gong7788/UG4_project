import copy
import ff
import dmrs_functions
from collections import namedtuple, defaultdict
import goal_updates
import prob_model
import pddl_functions
import numpy as np
import multiprocessing as mp
import time
from ff import NoPlanError, IDontKnowWhatIsGoingOnError
import logging



def log_cm(cm):
    logger.debug(cm.name)
    logger.debug('Positive class attributs: ' + str(cm.mu0) + ' ' + str(cm.sigma0))
    logger.debug('Negative class attributs: ' + str(cm.mu1) + ' ' + str(cm.sigma1))

logger = logging.getLogger('agent')
handler = logging.StreamHandler()
#logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

dialogue = logging.getLogger('dialogue')
#dialogue.addHandler(handler)
#logger.setLevel(logging.INFO)



Message = namedtuple('Message', ['rel', 'o1', 'o2', 'T', 'o3'])


def queuer(q, domain, problem):
    try:
        q.put(ff.run(domain, problem))
    except NoPlanError:
        q.put('ERROR')





class Priors(object):
    def __init__(self, objects):
        self.priors = {o: defaultdict(lambda: 0.5) for o in objects}

    def get_priors(self, message, args):
        o1 = self.priors[args[0]][message.o1[0]]
        o2 = self.priors[args[1]][message.o2[0]]
        names = message.o1[0] + ' ' + message.o2[0]
        prior = [o1, o2]
        if message.T == 'table':
            c1 = self.priors[message.o3][message.o1[0]]
            c2 = self.priors[message.o3][message.o2[0]]
            o3 = c1/(c1+c2)
            prior.append(o3)
        #logger.debug('priors for {}: ({})'.format(names, ','.join(map(str, prior))))
        return tuple(prior)

    def update(self, prior_dict):
        for obj, dict_ in prior_dict.items():
            for colour, value in dict_.items():
                self.priors[obj][colour] = value


    def to_dict(self):
        for o, d in self.priors.items():
            self.priors[o] = dict(d)

class Agent(object):


    def plan(self):
        raise NotImplementedError()

    def act(self, action, args):
        self.world.update(action, args)
        self.sense()

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
        o1, o2 = sentence.strip('no, ').strip('put').split(' on ')
        o1 = o1.strip().split()[0]
        o2 = o2.strip().split()[0]
        return Message(rel, [o1], [o2], T, o3)


class CorrectingAgent(Agent):
    def __init__(self, world, colour_models=None, rule_beliefs=None,
                 domain_file='blocks-domain.pddl', teacher=None, threshold=0.7, update_negative=True):
        self.name = 'correcting'
        self.world = world
        self.domain = world.domain
        self.domain_file = domain_file
        observation = world.sense()
        self.problem = copy.deepcopy(world.problem)
        self.goal = goal_updates.create_default_goal()
        self.tmp_goal = None
        self.problem.initialstate = observation.state
        if colour_models is None:
            self.colour_models = {}
        else:
            self.colour_models = colour_models
        if rule_beliefs is None:
            self.rule_beliefs = {}
        else:
            self.rule_beliefs = rule_beliefs
        self.threshold = threshold
        self.tau = 0.6
        self.rule_models = {}
        self.teacher = teacher
        self.priors = Priors(world.objects)
        logger.debug('rule beliefs: ' + str(self.rule_beliefs))
        logger.debug('rule_models: ' + str(self.rule_models))
        logger.debug('colour_models: ' + str(self.colour_models))
        self.update_negative=update_negative

    def new_world(self, world):
        self.world = world
        observation = world.sense()
        self.problem = copy.deepcopy(world.problem)
        self.tmp_goal = None
        self.problem.initialstate = observation.state
        self.priors = Priors(world.objects)

    def plan(self):
        self.problem.goal = goal_updates.update_goal(self.goal, self.tmp_goal)
        tau = 0.6
        while True:
            self.sense(threshold=tau)
            with open('tmp/problem.pddl', 'w') as f:
                f.write(self.problem.asPDDL())

            try:
                plan = ff.run(self.domain_file, 'tmp/problem.pddl')
                return plan
            except (NoPlanError, IDontKnowWhatIsGoingOnError):
                # self.problem.goal = goal_updates.update_goal(goal_updates.create_default_goal(), self.tmp_goal)
                # with open('tmp/problem.pddl', 'w') as f:
                #     f.write(self.problem.asPDDL())
                # plan = ff.run(self.domain_file, 'tmp/problem.pddl')
                tau += 0.1
        # return plan

    def _print_goal(self):
        print(self.goal.asPDDL())

    def get_correction(self, user_input, action, args):
        visible = {}
        # since this action is incorrect, ensure it is not done again
        not_on_xy = pddl_functions.create_formula('on', args, op='not')
        self.tmp_goal = goal_updates.update_goal(self.tmp_goal, not_on_xy)

        # get the relevant parts of the message
        message = read_sentence(user_input, use_dmrs=False)

        # build the rule model
        rule_model, rules = self.build_model(message)


        log_cm(rule_model.c1)
        log_cm(rule_model.c2)
        if message.T.lower() == 'table':
            log_cm(rule_model.c3)

        logger.debug('rule priors' + str(rule_model.rule_prior))

        # gets F(o1), F(o2), and optionally F(o3)
        data = self.get_data(message, args)

        priors = self.priors.get_priors(message, args)
        logger.debug('object priors: ' + str(priors))
        r1, r2 = rule_model.get_message_probs(data, priors=priors)
        logger.debug('predictions: ' + str((r1, r2)))


        # if there is no confidence in the update then ask for help
        if max(r1, r2) < self.threshold:
            logger.debug('asking question')
            question = 'Is the top object {}?'.format(message.o1[0])
            dialogue.info("R: " + question)
            answer = self.teacher.answer_question(question, self.world)
            dialogue.info("T: " +  answer)

            bin_answer = int(answer.lower() == 'yes')
            visible[message.o1[0]] = bin_answer
            message_probs = rule_model.get_message_probs(data, visible=copy.copy(visible), priors=priors)

        objs = [args[0], args[1], message.o3]
        prior_updates = rule_model.updated_object_priors(data, objs, priors, visible=copy.copy(visible))

        # update the goal belief
        #logger.debug(prior_updates)

        rule_model.update_belief_r(r1, r2)
        rule_model.update_c(data, priors=self.priors.get_priors(message, args), visible=visible, update_negative=self.update_negative)

        self.priors.update(prior_updates)
        self.update_goal()
        self.world.back_track()
        self.sense()
        self.rule_models[(rule_model.rule_names, message.T)] = rule_model

    def tracking(self):
        for colour in self.colour_models.values():
            logger.debug(colour.mu0)
            logger.debug(colour.sigma0)

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

        # If this this rule model already exists, keep using the same
        if (rule_names, message.T) in self.rule_models.keys():
            logger.debug('reusing a rule model')
            return self.rule_models[(rule_names, message.T)], rules

        # If a table correction exists then use the same rule beliefs for tower correction or vis versa
        try:
            equivalent_rule_model = list(filter(lambda x: x[0] == rule_names, self.rule_models.keys()))[0]
            rule_belief = self.rule_models[equivalent_rule_model].rule_belief
        except IndexError:
            rule_belief = None


        # if we have colour models for the relevant colours, use these for this model
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

        # build the model
        if message.T == 'tower':
            rule_model = prob_model.CorrectionModel(rule_names, rules, colour_model1, colour_model2, rule_belief=rule_belief)
        else:
            rule_model = prob_model.TableCorrectionModel(rule_names, rules, colour_model1, colour_model2, rule_belief=rule_belief)
        return rule_model, rules


    def update_goal(self):
        rules = []
        used_models = set()
        for (rule_name, T), correction_model in self.rule_models.items():
            if rule_name in used_models:
                continue
            else:
                rule_belief = correction_model.rule_belief
                rules.extend(rule_belief.get_best_rules())
                used_models.add(rule_name)

        self.goal = goal_updates.goal_from_list(rules)


    def sense(self, threshold=0.6):
        observation = self.world.sense()
        self.problem.initialstate = observation.state

        for colour, model in self.colour_models.items():
            # these objects include tower locations, which they should not # I don't htink thats true?
            for obj in pddl_functions.filter_tower_locations(observation.objects, get_locations=False):
                data = observation.colours[obj]
                p_colour = model.p(1, data)
                if p_colour > threshold:
                    colour_formula = pddl_functions.create_formula(colour, [obj])
                    self.problem.initialstate.append(colour_formula)

        return observation

    def act(self, action, args):
        self.world.update(action, args)
        self.sense()


class NeuralCorrectingAgent(CorrectingAgent):

    def __init__(self, world, colour_models=None, rule_beliefs=None,
                 domain_file='blocks-domain.pddl', teacher=None, threshold=0.7, H=4):
        self.H = H
        super().__init__(world, colour_models=colour_models, rule_beliefs=rule_beliefs, domain_file=domain_file, teacher=teacher, threshold=threshold)

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
            colour_model1 = prob_model.NeuralColourModel(c1, H=self.H)
            self.colour_models[c1] = colour_model1
        try:
            colour_model2 = self.colour_models[c2]
        except KeyError:
            colour_model2 = prob_model.NeuralColourModel(c2, H=self.H)
            self.colour_models[c2] = colour_model2

        if message.T == 'tower':
            rule_model = prob_model.CorrectionModel(rule_names, rules, colour_model1, colour_model2, rule_belief=rule_probs)

        else:
            rule_model = prob_model.TableCorrectionModel(rule_names, rules, colour_model1, colour_model2, rule_belief=rule_probs)
        return rule_model, rules




class RandomAgent(Agent):
    def __init__(self, world, colour_models = {}, rule_beliefs = {}, domain_file='blocks-domain.pddl', teacher=None, threshold=0.7):
        self.name = 'random'
        self.world = world
        self.domain = world.domain
        self.domain_file = domain_file
        observation = world.sense()
        self.problem = copy.deepcopy(world.problem)
        self.goal = goal_updates.create_default_goal()
        self.tmp_goal = None
        self.problem.initialstate = observation.state
        #self.colour_models = colour_models
        #self.rule_beliefs = rule_beliefs
        #self.threshold = threshold
        #self.tau = 0.6
        #self.rule_models = {}
        self.teacher = teacher
        #self.priors = Priors(world.objects)


    def new_world(self, world):
        self.world = world
        observation = world.sense()
        self.problem = copy.deepcopy(world.problem)
        self.tmp_goal = None
        self.problem.initialstate = observation.state
        #self.priors = Priors(world.objects)


    def plan(self):
        self.problem.goal = goal_updates.update_goal(goal_updates.create_default_goal(), self.tmp_goal)
        with open('tmp/problem.pddl', 'w') as f:
            f.write(self.problem.asPDDL())
        plan = ff.run(self.domain_file, 'tmp/problem.pddl')
        return plan


    def sense(self):
        observation = self.world.sense()
        self.problem.initialstate = observation.state

        return observation

    def get_correction(self, user_input, action, args):
        # since this action is incorrect, ensure it is not done again
        not_on_xy = pddl_functions.create_formula('on', args, op='not')
        self.tmp_goal = goal_updates.update_goal(self.tmp_goal, not_on_xy)
        self.world.back_track()
        self.sense()


class RLAgent(Agent):
    pass
