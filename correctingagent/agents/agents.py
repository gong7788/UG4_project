import copy
from ..pddl import ff
from ..language import dmrs_functions
from collections import namedtuple, defaultdict
from ..pddl import goal_updates
from ..models import prob_model
from ..pddl import pddl_functions
import numpy as np
import logging
from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from ..util.colour_dict import colour_names, colour_dict
import pickle
from ..models import search

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
    except ff.NoPlanError:
        q.put('ERROR')





class Priors(object):
    def __init__(self, objects, known_colours=[]):
        """known colours = list of pairs"""
        if not known_colours:
            self.priors = {o: defaultdict(lambda: 0.5) for o in objects}
        else:
            self.priors = {o: defaultdict(float) for o in objects}
            for o, c in known_colours:
                self.priors[o][c] = 1.0


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


def pickle_agent(agent, f):
    cms = {}
    for colour, cm in agent.colour_models.items():
        datas = (cm.data, cm.weights, cm.data_neg, cm.weights_neg)
        cms[colour] = datas


    if isinstance(agent, NoLanguageAgent):
        pass
    else:
        for r, rule_model in agent.rule_models.items():
            rule_model.c1 = rule_model.c1.name
            rule_model.c2 = rule_model.c2.name
            try:
                rule_model.c3 = rule_model.c3.name
            except AttributeError:
                pass
            agent.rule_models[r] = rule_model

    agent.colour_models = None
    output = (agent, cms)

    agent.build_model = None

    pickle.dump(output, f)

def load_agent(f):

    agent, cms = pickle.load(f)
    out_cms = {}
    for colour, cm_data in cms.items():
        data, weights, data_neg, weights_neg = cm_data
        cm = prob_model.KDEColourModel(colour, data=data, weights=weights, data_neg=data_neg, weights_neg=weights_neg)
        out_cms[colour] = cm

    if isinstance(agent, NoLanguageAgent):
        agent.build_model = NoLanguageAgent.build_model
    else:
        for r, rule_model in agent.rule_models.items():
            rule_model.c1 = out_cms[rule_model.c1]
            rule_model.c2 = out_cms[rule_model.c2]
            try:
                c1, c2 = rule_model.c3.split('/')
                rule_model.c3 = prob_model.KDEColourModel(rule_model.c3, data=out_cms[c1].data, weights=out_cms[c1].weights, data_neg=out_cms[c2].data, weights_neg=out_cms[c2].weights)
            except AttributeError:
                pass
            agent.rule_models[r] = rule_model
        agent.build_model = CorrectingAgent.build_model
    agent.colour_models = out_cms

    return agent


def colour_tuple_to_dict(colour_tuple):
    out = defaultdict(list)
    for obj, c in colour_tuple:
        out[obj].append(c)
    return dict(out)

main_colours = list(colour_dict.keys())

class Tracker(object):

    def __init__(self):
        pass

    def log_accuracy(self, cm):
        pass

    def store_colour_accuracy(self, results, true_colours, threshold, goal):
        true_colour_dict = colour_tuple_to_dict(true_colours)
        #print(true_colour_dict)
        #print(results)
        #print(threshold)
        for obj, val in results.items():
            for colour, prediction in val.items():
                predict = colour if prediction > threshold else '-' + colour

                # print(obj, predict, next(filter(lambda x: x in main_colours, true_colour_dict[obj])), prediction)


        # check if the prediction is correct, ie does the prediction of red or not red match whether the object is actually red
        # record what the result was, what mistakes were made, at what threshold and what the goal was.
        # How am a saving the results?
        # Probably a dataframe where each row is one of the instances. However things like the correct state might be difficult to store in DF form.
        #





class CorrectingAgent(Agent):
    def __init__(self, world, colour_models=None, rule_beliefs=None,
                 domain_file='blocks-domain-colour-unknown.domain', teacher=None, threshold=0.7,
                 update_negative=True, update_once=True, colour_model_type='default',
                 model_config={}, tracker=Tracker()):
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
        # logger.debug('rule beliefs: ' + str(self.rule_beliefs))
        # logger.debug('rule_models: ' + str(self.rule_models))
        # logger.debug('colour_models: ' + str(self.colour_models))
        self.update_negative=update_negative
        self.objects_used_for_update = set()
        self.update_once = update_once
        assert(colour_model_type in ['default', 'kde'])
        self.colour_model_type = colour_model_type
        self.model_config = model_config
        self.tracker = tracker


    def new_world(self, world):
        self.world = world
        observation = world.sense()
        self.problem = copy.deepcopy(world.problem)
        self.tmp_goal = None
        self.problem.initialstate = observation.state
        self.priors = Priors(world.objects)
        self.objects_used_for_update = set()

    def plan(self):
        observation, results = self.sense()
        planner = search.Planner(results, observation, self.goal, self.tmp_goal, self.problem, domain_file=self.domain_file)
        return planner.plan()


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

        # if isinstance(self, CorrectingAgent):
        #     log_cm(rule_model.c1)
        #     log_cm(rule_model.c2)
        #     if message.T.lower() == 'table':
        #         log_cm(rule_model.c3)

        logger.debug('rule priors' + str(rule_model.rule_prior))

        # gets F(o1), F(o2), and optionally F(o3)
        data = self.get_data(message, args)

        priors = self.priors.get_priors(message, args)
        logger.debug('object priors: ' + str(priors))
        r1, r2 = rule_model.get_message_probs(data, priors=priors)
        logger.debug('predictions: ' + str((r1, r2)))

        #print(r1, r2)

        # if there is no confidence in the update then ask for help
        if max(r1, r2) < self.threshold:
            # logger.debug('asking question')
            question = 'Is the top object {}?'.format(message.o1[0])
            dialogue.info("R: " + question)
            answer = self.teacher.answer_question(question, self.world)
            dialogue.info("T: " +  answer)

            bin_answer = int(answer.lower() == 'yes')
            visible[message.o1[0]] = bin_answer
            message_probs = rule_model.get_message_probs(data, visible=copy.copy(visible), priors=priors)
            r1, r2 = message_probs


        objs = [args[0], args[1], message.o3]
        prior_updates = rule_model.updated_object_priors(data, objs, priors, visible=copy.copy(visible))

        # update the goal belief
        #logger.debug(prior_updates)

        rule_model.update_belief_r(r1, r2)
        which_to_update = [1,1,1]
        if self.update_once:
            for i, o in enumerate(objs):
                if o in self.objects_used_for_update:
                    which_to_update[i] = 0
                self.objects_used_for_update.add(o)

        rule_model.update_c(data, priors=self.priors.get_priors(message, args), visible=visible, update_negative=self.update_negative, which_to_update=which_to_update)

        self.rule_models[(rule_model.rule_names, message.T)] = rule_model
        self.priors.update(prior_updates)
        self.update_goal()
        logger.debug(self.goal.asPDDL())
        self.world.back_track()
        #self.sense()


    def tracking(self):
        for colour in self.colour_models.values():
            logger.debug(colour.mu0)
            logger.debug(colour.sigma0)

    def no_correction(self, action, args):
        for rule_model in [model for (rule_name, model_type), model in self.rule_models.items() if model_type == 'tower']:
            message = read_sentence('no, put {} blocks on {} blocks'.format(rule_model.c1.name, rule_model.c2.name), use_dmrs=False)
            data = self.get_data(message, args)
            rule_model.update_c_no_corr(data)

    def get_data(self, message, args):
        observation = self.world.sense()

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
            # logger.debug('reusing a rule model')
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
            if self.colour_model_type == 'default':
                colour_model1 = prob_model.ColourModel(c1)
            elif self.colour_model_type == 'kde':
                colour_model1 = prob_model.KDEColourModel(c1, **self.model_config)
            else:
                raise ValueError('Unexpected colour_model_type: got {}'.format(self.colour_model_type))
            self.colour_models[c1] = colour_model1
        try:
            colour_model2 = self.colour_models[c2]
        except KeyError:
            if self.colour_model_type == 'default':
                colour_model2 = prob_model.ColourModel(c2)
            elif self.colour_model_type == 'kde':
                colour_model2 = prob_model.KDEColourModel(c2, **self.model_config)
            else:
                raise ValueError('Unexpected colour_model_type: got {}'.format(self.colour_model_type))
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
                # print(rule_name, correction_model)
                logger.debug(rule_name)
                rule_belief = correction_model.rule_belief
                # print(rule_belief.belief)
                rules.extend(rule_belief.get_best_rules())
                logger.debug(rules)
                used_models.add(rule_name)

        self.goal = goal_updates.goal_from_list(rules)
        # print(self.goal.asPDDL())

    def sense(self, threshold=0.5):
        observation = self.world.sense()
        self.problem.initialstate = observation.state
        results = defaultdict(dict)
        for colour, model in self.colour_models.items():
            # these objects include tower locations, which they should not # I don't htink thats true?
            for obj in pddl_functions.filter_tower_locations(observation.objects, get_locations=False):
                data = observation.colours[obj]
                p_colour = model.p(1, data)
                # if p_colour > threshold:
                #     colour_formula = pddl_functions.create_formula(colour, [obj])
                #     self.problem.initialstate.append(colour_formula)

                results[obj][colour] = p_colour
        results = dict(results)
        self.state = search.State(observation, results, threshold)
        self.problem.initialstate = self.state.to_pddl()



        true_colours = get_colours(self.world.sense(obscure=False))
        self.tracker.store_colour_accuracy(results, true_colours, threshold, self.goal)

        return observation, results

    def act(self, action, args):
        self.world.update(action, args)
        self.sense()


class NeuralCorrectingAgent(CorrectingAgent):

    def __init__(self, world, colour_models=None, rule_beliefs=None,
                 domain_file='blocks-domain.domain', teacher=None, threshold=0.7, H=4, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False, optimiser='Adam'):
        self.H = H
        super().__init__(world, colour_models=colour_models, rule_beliefs=rule_beliefs, domain_file=domain_file, teacher=teacher, threshold=threshold)




    def build_model(self, message):

        rules = goal_updates.create_goal_options(message.o1, message.o2)
        rule_names = tuple(map(lambda x: x.asPDDL(), rules))

        # If this this rule model already exists, keep using the same
        if (rule_names, message.T) in self.rule_models.keys():
            # logger.debug('reusing a rule model')
            return self.rule_models[(rule_names, message.T)], rules

        # If a table correction exists then use the same rule beliefs for tower correction or vis versa
        try:
            equivalent_rule_model = list(filter(lambda x: x[0] == rule_names, self.rule_models.keys()))[0]
            rule_belief = self.rule_models[equivalent_rule_model].rule_belief
        except IndexError:
            rule_belief = None

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
            rule_model = prob_model.CorrectionModel(rule_names, rules, colour_model1, colour_model2, rule_belief=rule_belief)

        else:
            rule_model = prob_model.TableCorrectionModel(rule_names, rules, colour_model1, colour_model2, rule_belief=rule_belief)
        return rule_model, rules




class RandomAgent(Agent):
    def __init__(self, world, colour_models = {}, rule_beliefs = {}, domain_file='blocks-domain.domain', teacher=None, threshold=0.7):
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
        with open('tmp/problem.domain', 'w') as f:
            f.write(self.problem.asPDDL())
        plan = ff.run(self.domain_file, 'tmp/problem.domain')
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

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(6, 128)
        self.affine2 = nn.Linear(128, 1)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

def get_top(relations):
    stuff = list(filter(lambda x: 'in-tower' in x[1], relations.items()))
    if len(stuff) == 1:
        return stuff[0][0]
    stuff = dict(stuff)
    bottom = 't0'
    while True:
        on = stuff.pop(bottom)['on']
        bottom = on.args.args[0].arg_name
        if len(stuff) == 1:
            return bottom

def get_state(o1, o2, obs):
    try:
        c1 = obs.colours[o1]
    except KeyError:
        c1 = np.array([-1, -1, -1])
    c2 = obs.colours[o2]
    return np.concatenate([c1, c2])

class RLAgent(Agent):

    def __init__(self, world, teacher=None):
        self.name = 'correcting'
        self.world = world
        self.teacher = teacher
        self.policy = Policy()
        self.optimiser =  torch.optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.explored_objects = set()

    def new_world(self, world):
        self.world = world
        self.explored_objects = set()

    def get_all_states(self):
        obs = self.world.sense()
        top_object = get_top(obs.relations)
        available_objects = list(filter(lambda x: (top_object, x) not in self.explored_objects, self.world.objects_not_in_tower()))
        states = np.array([get_state(top_object, o, obs) for o in available_objects])
        return states, top_object, available_objects


    def select_action(self, state, top_object, available_objects):

        state= torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs.squeeze())
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return 'put', [top_object, available_objects[action.item()]] # translate this into put bi bj

    def sense(self, threshold=0.6):
        observation = self.world.sense()
        self.problem.initialstate = observation.state

        # for colour, model in self.colour_models.items():
        #     # these objects include tower locations, which they should not # I don't htink thats true?
        #     for obj in pddl_functions.filter_tower_locations(observation.objects, get_locations=False):
        #         data = observation.colours[obj]
        #         p_colour = model.p(1, data)
        #         if p_colour > threshold:
        #             colour_formula = pddl_functions.create_formula(colour, [obj])
        #             self.problem.initialstate.append(colour_formula)

        return observation

    def plan(self):
        w = copy.deepcopy(self.world)
        obs = w.sense()
        while not w.test_success():
            s = self.get_all_states()
            top = get_top(obs.relations)
            a = self.select_action(s)


    def get_correction(self, user_input, action, args):
        visible = {}
        # since this action is incorrect, ensure it is not done again
        not_on_xy = pddl_functions.create_formula('on', args, op='not')
        self.tmp_goal = goal_updates.update_goal(self.tmp_goal, not_on_xy)

        # get the relevant parts of the message
        message = read_sentence(user_input, use_dmrs=False)

        # build the rule model
        rule_model, rules = self.build_model(message)

        # if isinstance(self, CorrectingAgent):
        #     log_cm(rule_model.c1)
        #     log_cm(rule_model.c2)
        #     if message.T.lower() == 'table':
        #         log_cm(rule_model.c3)

        logger.debug('rule priors' + str(rule_model.rule_prior))

        # gets F(o1), F(o2), and optionally F(o3)
        data = self.get_data(message, args)

        priors = self.priors.get_priors(message, args)
        logger.debug('object priors: ' + str(priors))
        r1, r2 = rule_model.get_message_probs(data, priors=priors)
        logger.debug('predictions: ' + str((r1, r2)))            # these objects include tower locations, which they should not # I don't htink thats true?



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

def get_colours(obs):
    for obj, value in obs.relations.items():
        for c in value:
            if c in colour_names:
                yield (obj, c)

def get_least_likely_object(results, colour_model_name):
    return min([(predictions[colour_model_name], obj) for obj, predictions in results.items()])[0]

class NoLanguageAgent(CorrectingAgent):


    def __init__(self, *args, **kwargs):
        self.rules = []
        self.unsure = []
        self.unsure_rules = []
        self.unsure_rules_test = []
        super().__init__(*args, **kwargs)

    def get_correction(self, user_input, action, args):
        # visible = {}
        # since this action is incorrect, ensure it is not done again
        not_on_xy = pddl_functions.create_formula('on', args, op='not')
        self.tmp_goal = goal_updates.update_goal(self.tmp_goal, not_on_xy)

        # get the relevant parts of the message
        message = read_sentence(user_input, use_dmrs=False)

        # build the rule model
        self.build_model(message, args)

        self.update_goal()
        # logger.debug(self.goal.asPDDL())
        self.world.back_track()
        # #self.sense()


    def plan(self):

        observation, results = self.sense()
        planner = search.NoLanguagePlanner(results, observation, self.unsure, self.goal, self.tmp_goal, self.problem, domain_file=self.domain_file)
        return planner.plan()


    def get_data(self, message, args):
        observation = self.world.sense()

        c1 = message.o1[0]
        c2 = message.o2[0]
        c3 = '{}/{}'.format(c1, c2)
        o1 = args[0]
        o2 = args[1]
        o3 = message.o3
        colour_data = observation.colours
        # try:
        #     if 't' not in o2:
        #         data_dict = {c1:colour_data[o1], c2:colour_data[o2], c3:colour_data[o3]}
        #     else:
        #         data_dict = {c1:colour_data[o1], c2:None, c3:colour_data[o3]}self.unsure_rules
        # except KeyError:
        #     if 't' not in o2:
        #         data_dict = {c1:colour_data[o1], c2:colour_data[o2]}
        #     else:
        #         data_dict = {c1:colour_data[o1], c2:None}


        try:
            if 't' not in o2:
                return {'o1':colour_data[o1], 'o2':colour_data[o2], 'o3':colour_data[o3]}
            else:
                return {'o1':colour_data[o1], 'o2':None, 'o3':colour_data[o3]}
        except KeyError:
            if 't' not in o2:
                return {'o1':colour_data[o1], 'o2':colour_data[o2]}
            else:
                return {'o1':colour_data[o1], 'o2':None}
        # return data_dict

    def find_matching_cm(self, dp):
        if dp is None:
            return None
        dp_prim = dp.tolist()

        for c, cm in self.colour_models.items():
            if dp_prim in cm.data.tolist():
                print('found match', dp, cm.data, dp in cm.data)
                return c, cm
        else:
            return None

    def build_model(self, message, args):
        data = self.get_data(message, args)
        i = 0
        cm1 = self.find_matching_cm(data['o1'])
        cm2 = self.find_matching_cm(data['o2'])
        print('cm1', args[0], cm1, data['o1'])
        print('cm2', args[1], cm2, data['o2'])
        n = len(self.colour_models)
        if cm1 is not None and cm2 is not None:
            c1, c1_model = cm1
            c2, c2_model = cm2
        elif cm1 is not None:
            c1, c1_model = cm1

            c2 = "C{}".format(n)
            i += 1
            if data['o2'] is not None:
                c2_model = prob_model.KDEColourModel(c2, data=np.array([data['o2']]),
                                                     weights=np.array([1]), **self.model_config)
            else:
                c2_model = None
        elif cm2 is not None:
            c2, c2_model = cm2

            i += 1
            c1 = "C{}".format(n)
            c1_model = prob_model.KDEColourModel(c1, data=np.array([data['o1']]),
                                                 weights=np.array([1]), **self.model_config)
        else:

            c1 = "C{}".format(n)
            c1_model = prob_model.KDEColourModel(c1, data=np.array([data['o1']]),
                                                 weights=np.array([1]), **self.model_config)

            c2 = "C{}".format(n+1)
            if data['o2'] is not None:
                c2_model = prob_model.KDEColourModel(c2, data=np.array([data['o2']]),
                                                     weights=np.array([1]), **self.model_config)
            else:
                c2_model = None
            i += 2

        #If tower: create the negated rule and two new colour variables.
        if message.T == 'tower':
            if data['o2'] is None:
                return
            rule = goal_updates.create_negative_goal([c1], [c2])

            self.colour_models.update({c1:c1_model, c2:c2_model})
            self.rule_models['not {} and {}'.format(c1, c2)] = rule
            return

        elif message.T == 'table':
            cm3 = self.find_matching_cm(data['o3'])
            if cm3 is not None:
                c3, c3_model = cm3
            else:
                c3 = "C{}".format(n+i)
                c3_model = prob_model.KDEColourModel(c3, data=np.array([data['o3']]), weights=np.array([1]), **self.model_config)


            rule1 = goal_updates.create_goal([c3], [c2])
            rule2 = goal_updates.create_goal([c3], [c1], ['?y', '?x'])
            if data['o2'] is None:
                self.colour_models.update({c1:c1_model, c3:c3_model})
                self.rule_models['{}(x) -> {}(y)'.format(c3, c1)] = rule2
            else:
                self.colour_models.update({c1:c1_model, c2:c2_model, c3:c3_model})
                self.rule_models['{}(x) -> {}(y) or {}(y) -> {}(x)'.format(c3, c1, c3, c2)] = (rule1, rule2)
                self.unsure_rules.append((rule1, rule2))
                obs, results = self.sense()
                most_different = get_least_likely_object(results, c3)
                test = pddl_functions.create_formula('on', [args[1], most_different])
                self.unsure_rules_test.append(test)

            return

    def update_goal(self):
        self.unsure = []
        self.rules = []
        for rule_name, rule in self.rule_models.items():
            try:
                rule1, rule2 = rule
                self.unsure.append(rule)
            except:
                self.rules.append(rule)

        self.goal = goal_updates.goal_from_list(self.rules)
        # print(self.goal.asPDDL())


class PerfectColoursAgent(CorrectingAgent):

    def new_world(self, world):
        self.world = world
        observation = world.sense(obscure=False)
        self.problem = copy.deepcopy(world.problem)
        self.tmp_goal = None
        self.problem.initialstate = observation.state

        known_colours = get_colours(observation)

        self.priors = Priors(world.objects, known_colours=known_colours)


    def sense(self, threshold=0.6):
        observation = self.world.sense(obscure=False)
        self.problem.initialstate = observation.state
        #
        # for colour, model in self.colour_models.items():
        #     for obj in pddl_functions.filter_tower_locations(observation.objects, get_locations=False):
        #         data = observation.colours[obj]
        #         p_colour = model.p(1, data)
        #         if p_colour > threshold:
        #             colour_formula = pddl_functions.create_formula(colour, [obj])
        #             self.problem.initialstate.append(colour_formula)
        return observation
