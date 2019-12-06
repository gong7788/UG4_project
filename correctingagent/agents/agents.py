import copy
import time

import correctingagent.world.rules
from correctingagent.models.search import TestFailed
from correctingagent.util import util
from ..pddl import ff
from ..language import dmrs_functions
from collections import namedtuple, defaultdict
from correctingagent.world import goals, Path
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
from correctingagent.world.rules import ColourCountRule, RedOnBlueRule, NotRedOnBlueRule


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


# def read_sentence(sentence, use_dmrs=True):
#     sentence = sentence.lower().strip('no,')
#     o3 = None
#     T = 'tower'
#     if 'tower because you must' in sentence:
#         T = 'table'
#         o3 = sentence.split('put')[1].split('in')[0].strip()
#         sentence = sentence.split('because you must')[1].strip()
#     if use_dmrs:
#         rel, o1, o2= dmrs_functions.sent_to_tripple(sentence)
#         o1 = dmrs_functions.get_adjectives(o1)
#         o2 = dmrs_functions.get_adjectives(o2)
#         rel = dmrs_functions.get_pred(rel['predicate'])
#         return Message(rel, o1, o2, T, o3)
#     else:
#         rel = 'on'
#         o1, o2 = sentence.strip('no, ').strip('put').split(' on ')
#         o1 = o1.strip().split()[0]
#         o2 = o2.strip().split()[0]
#         return Message(rel, [o1], [o2], T, o3)
def read_sentence(sentence, use_dmrs=False):
    sentence = sentence.lower()
    if sentence == 'no':
        return Message(None, None, None, 'no', None)
    sentence = sentence.replace('no,', '')
    o3 = None
    T = 'tower'
    if "same reason" in sentence:
        return Message(None, None, None, 'same reason', None)
    elif "cannot put more than" in sentence:
        o3 = None
        addon = ""
        if "and" in sentence:
            sentence, red_on_blue = sentence.split(' and ')
            red_on_blue = red_on_blue.replace('you must', '').strip()
            o3 = read_sentence(red_on_blue)
            addon = "+tower"
        sentence = sentence.replace('you cannot put more than', '').replace('blocks in a tower', '').strip()
        number, colour = sentence.split(' ')
        return Message(None, colour, int(number), "colour count"+addon, o3)
    elif "you put" in sentence:
        T = 'evidence'
        sentence = sentence.replace('you', '').strip()
        sentence = sentence.replace('a ', '')

    elif "that is not" in sentence:
         sentence = sentence.replace('that is not', '').strip()
         sentence = sentence.replace('again', '').strip()
         return Message(None, sentence, None, 'partial.neg', None)

    elif "don't" in sentence:
        T = 'neg'
        sentence = sentence.replace("don't", '').strip()

    if 'tower because you' in sentence:
        T = 'table'
        o3 = sentence.split('put')[1].split('in')[0].strip()
        sentence = sentence.split('because you')[1].strip()
        if 'cannot' in sentence:
            T = 'table.neg'
            sentence = sentence.replace('cannot', '').strip()
        else:
            sentence = sentence.replace('must', '').strip()

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
                 domain_file='blocks-domain.pddl', teacher=None, threshold=0.7,
                 update_negative=True, update_once=True, colour_model_type='default',
                 model_config={}, tracker=Tracker(), **kwargs):
        self.name = 'correcting'
        self.world = world
        self.domain = world.domain
        self.domain_file = domain_file
        observation = world.sense()
        self.problem = copy.deepcopy(world.problem)
        self.goal = goals.create_default_goal(domain_file)
        self.tmp_goal = None
        self.problem.initialstate = observation.state.to_formula()
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
        self.problem.initialstate = observation.state.to_formula()
        self.priors = Priors(world.objects)
        self.objects_used_for_update = set()

    def plan(self):
        print(self.domain_file, self.world.use_metric_ff)
        observation, results = self.sense()
        planner = search.Planner(results, observation, self.goal, self.tmp_goal,
                                 self.problem, domain_file=self.domain_file, use_metric_ff=self.world.use_metric_ff)
        step = time.time()

        plan = planner.plan()

        step_1 = time.time()
        delta = step_1 - step
        print(f"planning {delta} time")

        return plan

    def _print_goal(self):
        print(self.goal.asPDDL())

    def get_correction(self, user_input, action, args, test=False):
        visible = {}
        # since this action is incorrect, ensure it is not done again
        not_on_xy = pddl_functions.create_formula('on', args, op='not')
        self.tmp_goal = goals.update_goal(self.tmp_goal, not_on_xy)
        if not test:
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

            colour1 = message.o1[0]
            colour2 = message.o2[0]

            _, results = self.sense()
            try:
                result_c1 = results[args[0]][colour1]
            except KeyError:
                result_c1 = 0.
            try:
                result_c2 = results[args[1]][colour2]
            except KeyError:
                result_c2 = 0.
            # if there is no confidence in the update then ask for help
            if max(r1, r2) < self.threshold or (result_c1 == 0.5 or result_c2 == 0.5):
                # logger.debug('asking question')
                question = 'Is the top object {}?'.format(message.o1[0])
                dialogue.info("R: " + question)
                if len(args) == 2:
                    tower = None
                else:
                    tower = args[-1]
                answer = self.teacher.answer_question(question, self.world, tower)
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
            # print(r1, r2)
            # print(rule_model, rule_model.rule_belief.belief)
            # print(args)
            # print(self.sense()[1])
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
        rules = correctingagent.world.rules.Rule.generate_red_on_blue_options(message.o1[0], message.o2[0])

        #TODO change downstreem to expect Rule class rather than formula
        rules = [rule.to_formula() for rule in rules]
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

        self.goal = goals.goal_from_list(rules, self.domain_file)
        # print(self.goal.asPDDL())

    def sense(self, threshold=0.5):
        observation = self.world.sense()
        self.problem.initialstate = observation.state.to_formula()
        results = defaultdict(dict)
        for colour, model in self.colour_models.items():
            # these objects include tower locations, which they should not # I don't htink thats true?

            for obj in [obj for obj in observation.objects if 't' not in obj]:
                data = observation.colours[obj]
                p_colour = model.p(1, data)
                # if p_colour > threshold:
                #     colour_formula = pddl_functions.create_formula(colour, [obj])
                #     self.problem.initialstate.append(colour_formula)

                results[obj][colour] = p_colour
        results = dict(results)
        self.state = correctingagent.world.rules.State(observation, results, threshold)
        self.problem.initialstate = self.state.asPDDL()

        true_colours = [(obj, self.world.state.get_colour_name(obj)) for obj in self.world.state.objects]
        #rue_colours = get_colours(self.world.sense(obscure=False))
        self.tracker.store_colour_accuracy(results, true_colours, threshold, self.goal)

        return observation, results

    def act(self, action, args):
        self.world.update(action, args)
        self.sense()
#

class RandomAgent(Agent):
    def __init__(self, world, teacher=None, **kwargs):
        self.name = 'random'
        self.world = world
        self.domain = world.domain
        self.domain_file = world.domain_file
        observation = world.sense()
        self.problem = copy.deepcopy(world.problem)
        self.goal = goals.create_default_goal(str(self.domain_file))
        self.tmp_goal = None
        self.problem.initialstate = observation.state.to_formula()

        self.teacher = teacher

    def new_world(self, world):
        self.world = world
        observation = world.sense()
        self.problem = copy.deepcopy(world.problem)
        self.tmp_goal = None
        self.problem.initialstate = observation.state.to_formula()

    def plan(self):
        print(self.domain_file)
        config = util.get_config()
        data_dir = Path(config['data_location'])
        tmp_dir = data_dir / 'tmp' / 'problem.pddl'
        self.problem.goal = goals.update_goal(goals.create_default_goal(str(self.domain_file)), self.tmp_goal)
        with open(tmp_dir, 'w') as f:
            f.write(self.problem.asPDDL())
        print(f"running planner with {self.domain_file} {tmp_dir}")
        plan = ff.run(str(self.domain_file), str(tmp_dir), use_metric_ff=("updated" in str(self.domain_file)))
        return plan


    def sense(self):
        observation = self.world.sense()
        self.problem.initialstate = observation.state.to_formula()

        return observation

    def get_correction(self, user_input, action, args, test=False):
        # since this action is incorrect, ensure it is not done again
        args = args[:2]
        not_on_xy = pddl_functions.create_formula('on', args, op='not')
        self.tmp_goal = goals.update_goal(self.tmp_goal, not_on_xy)
        self.world.back_track()
        self.sense()


def get_colours(obs):
    for obj, value in obs.relations.items():
        for c in value:
            if c in colour_names:
                yield (obj, c)


class NoLanguageAgent(CorrectingAgent):

    def __init__(self, world, *args, domain_file='blocks-domain-colour-unknown.pddl', **kwargs):
        self.rules = []
        self.active_tests = []
        if 'updated' in str(world.domain_file):
            domain_file = "blocks-domain-colour-unknown-cc.pddl"
            self.use_metric_ff = True
        else:
            self.use_metric_ff = False

        super().__init__(world, *args, domain_file=domain_file, **kwargs)
        if 'updated' in str(self.world.domain_file):
            self.domain_file = "blocks-domain-colour-unknown-cc.pddl"
            self.use_metric_ff = True

    def new_world(self, world):
        self.active_tests = []
        super().new_world(world)

    def get_correction(self, user_input, action, args, test=False):
        # visible = {}
        # since this action is incorrect, ensure it is not done again
        not_on_xy = pddl_functions.create_formula('on', args[:2], op='not')
        self.tmp_goal = goals.update_goal(self.tmp_goal, not_on_xy)

        if not test:

            for test in self.active_tests:
                if (test.objects[0] == args[0] and test.objects[1] == args[1]) or test.failed:
                    correct_rule = test.rule1
                    self.active_tests.remove(test)
                else:
                    continue
                self.goal = goals.update_goal(self.goal, correct_rule.to_formula())

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

        planner = search.NoLanguagePlanner(results, observation, self.active_tests, self.goal, self.tmp_goal, self.problem, domain_file=self.domain_file, use_metric_ff=self.use_metric_ff)

        try:
            return planner.plan()
        except ValueError as e:
            print(self.goal.asPDDL())
            raise e

    def get_data(self, message, args):
        observation = self.world.sense()

        o1 = args[0]
        o2 = args[1]

        colour_data = observation.colours

        if not isinstance(message.o3, str):
            o3 = None
        else:
            o3 = message.o3

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

    def find_matching_cm(self, dp):
        """If a stored colour model uses the same data point as dp then return that cm"""
        if dp is None:
            return None
        dp_prim = dp.tolist()

        for c, cm in self.colour_models.items():
            if dp_prim in cm.data.tolist():
                # print('found match', dp, cm.data, dp in cm.data)
                return c, cm
            elif cm.p(1, dp_prim) > 0.90:
            #elif all([abs(val1-val2) < 0.01 for val1, val2 in zip(dp_prim, cm.data.tolist()[0])]):
                cm.update(dp, cm.p(1, dp_prim))
                return c, cm
        else:
            n = len(self.colour_models) - int('blue' in self.colour_models.keys())
            if n > 37:
                raise ValueError("Too many colour models, stopped learning")
            c1_model = prob_model.KDEColourModel(f"C{n}", data=np.array([dp]),
                                                 weights=np.array([1]), **self.model_config)
            return f"C{n}", c1_model

    def no_correction(self, action, args):
        for test in self.active_tests:
            if (test.objects[0] == args[0] and test.objects[1] == args[1]):
                correct_rule = test.rule2
                self.active_tests.remove(test)
            elif test.failed:
                correct_rule = test.rule1
                self.active_tests.remove(test)
            else:
                continue
            self.goal = goals.update_goal(self.goal, correct_rule.to_formula())

    def build_model(self, message, args):
        data = self.get_data(message, args)
        i = 0
        # To reduce the number of individual colour names reuse colour models where the colour is the same
        # This only works because we have a set number of colour terms
        # First try to find colour models that match particular data points
        if data['o2'] is None:
            print("o2 is bottom position")
            return
        try:
            c1, c1_model = self.find_matching_cm(data['o1'])
            c2, c2_model = self.find_matching_cm(data['o2'])
        except ValueError:
            print("too many colour models, stopped learning")
            return

        #If tower: create the negated rule and two new colour variables.
        if message.T == 'tower':
            if data['o2'] is None:
                return
            rule = correctingagent.world.rules.NotRedOnBlueRule(c1, c2).to_formula()

            self.colour_models.update({c1:c1_model, c2:c2_model})
            self.rule_models['not {} and {}'.format(c1, c2)] = rule
            return

        elif message.T == 'table':
            c3, c3_model = self.find_matching_cm(data['o3'])


            rule1 = RedOnBlueRule(c3, c2, 1)
            rule2 = RedOnBlueRule(c1, c3, 2)

            # rule1 = correctingagent.world.rules.create_red_on_blue_rule([c3], [c2])
            # rule2 = correctingagent.world.rules.create_red_on_blue_rule([c3], [c1], ['?y', '?x'])
            if data['o2'] is None:
                self.colour_models.update({c1:c1_model, c3:c3_model})
                self.rule_models[rule2] = rule2
            else:
                self.colour_models.update({c1:c1_model, c2:c2_model, c3:c3_model})

                obs = self.world.sense()
                try:
                    test = search.ActiveLearningTest(rule1.to_formula(), rule2.to_formula(),
                                                 obs.colours, c3_model, args[1],
                                                 self.world)
                    self.active_tests.append(test)
                except TestFailed:
                    self.rule_models[rule2] = rule2.to_formula()
            return
        elif message.T == 'colour count':
            n = 0
            tower = args[-1]

            try:
                cm = self.colour_models['blue']
            except KeyError:
                cm = prob_model.KDEColourModel('blue', data=np.array([data['o1']]),
                                        weights=np.array([1]), **self.model_config)

            blocks_in_tower = self.world.state.get_objects_in_tower(tower)
            for block in blocks_in_tower:
                datum = self.world.observe_object(block)
                if cm.p(1, datum) > 0.5:
                    n = min(3, n+1)
            if n > 0:
                rule = ColourCountRule('blue', n)
                self.colour_models.update({'blue':cm})
                self.rule_models["colour_count"] = rule.to_formula()
            return
        elif message.T == 'colour count+tower':
            if 'blue' in self.colour_models.keys():
                rule = RedOnBlueRule('blue', c1, 2)
                self.colour_models.update({c1:c1_model})
                self.rule_models[str(rule)] = rule.to_formula()
                return
            else:
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

        self.goal = goals.goal_from_list(self.rules, self.domain_file)
        # print(self.goal.asPDDL())


class PerfectColoursAgent(CorrectingAgent):

    def new_world(self, world):
        self.world = world
        observation = world.sense(obscure=False)
        self.problem = copy.deepcopy(world.problem)
        self.tmp_goal = None
        self.problem.initialstate = observation.state.to_formula()

        known_colours = get_colours(observation)

        self.priors = Priors(world.objects, known_colours=known_colours)

    def sense(self, threshold=0.6):
        observation = self.world.sense(obscure=False)
        self.problem.initialstate = observation.state.to_formula()

        return observation, {}
