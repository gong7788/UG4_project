

import numpy as np
from correctingagent.agents.agents import CorrectingAgent, Tracker
from correctingagent.agents.teacher import FaultyTeacherAgent
from correctingagent.experiments.colour_model_evaluation import evaluate_colour_model
from correctingagent.pddl import pddl_functions
from correctingagent.util.CPD_generation import get_violation_type, get_predicate
from correctingagent.world import goals
from correctingagent.models.pgmmodels import PGMModel, InferenceType
from correctingagent.models.prob_model import KDEColourModel
from collections import namedtuple, defaultdict


from correctingagent.world.rules import Rule, ColourCountRule, RedOnBlueRule, CorrectionType

Message = namedtuple('Message', ['rel', 'o1', 'o2', 'T', 'o3'])


def colour_variable_name(colour, object):
    return f"{colour}({object})"


def corr_variable_name(time):
    return f"corr_{time}"


def parse_colour_count_correction(sentence):
    o3 = None
    addon = ""
    if "and" in sentence:
        sentence, red_on_blue = sentence.split(' and ')
        red_on_blue = red_on_blue.replace('you must', '').strip()
        o3 = read_sentence(red_on_blue)
        addon = "+tower"
    sentence = sentence.replace('you cannot put more than', '').replace('blocks in a tower', '').strip()
    number, colour = sentence.split(' ')
    return Message(None, colour, int(number), "colour count" + addon, o3)

def parse_negated_elaboration(sentence):
    sentence = sentence.replace('that is not', '').strip()
    sentence = sentence.replace('again', '').strip()
    return Message(None, sentence, None, 'partial.neg', None)


def parse_red_on_blue_rule(sentence):
    t = 'tower'

    o3 = None

    if 'tower because you' in sentence:  # No, now you cannot but b3 in the tower because you must put red blocks on blue blocks
        t = 'table'
        o3 = sentence.split('put')[1].split('in')[0].strip()
        sentence = sentence.split('because you')[1].strip()
        if 'cannot' in sentence:
            t = 'table.neg'
            sentence = sentence.replace('cannot', '').strip()
        else:
            sentence = sentence.replace('must', '').strip()

    rel = 'on'
    o1, o2 = sentence.strip('no, ').strip('put').split(' on ')
    o1 = o1.strip().split()[0]
    o2 = o2.strip().split()[0]
    return Message(rel, [o1], [o2], t, o3)


def read_sentence(sentence, use_dmrs=False):
    sentence = sentence.lower()

    # if sentence == 'no':
    #     return Message(None, None, None, 'no', None)

    sentence = sentence.replace('no,', '')

    if "same reason" in sentence:  # That is wrong for the same reason
        return Message(None, None, None, 'same reason', None)

    elif "cannot put more than" in sentence:  # No, you cannot put more than 2 red blocks in a tower
        return parse_colour_count_correction(sentence)

    # elif "you put" in sentence:  # No, you put a red block on a blue block
    #     T = 'evidence'
    #     sentence = sentence.replace('you', '').strip()
    #     sentence = sentence.replace('a ', '')

    elif "that is not" in sentence:  # No, that is not green again
        return parse_negated_elaboration(sentence)

    # elif "don't" in sentence:  # No, don't put red blocks on blue blocks
    #     T = 'neg'
    #     sentence = sentence.replace("don't", '').strip()

    else:
        return parse_red_on_blue_rule(sentence)


class PGMCorrectingAgent(CorrectingAgent):
    def __init__(self, world, colour_models=None, rule_beliefs=None,
                 domain_file='blocks-domain.pddl', teacher=None, threshold=0.7,
                 update_negative=True, update_once=True, colour_model_type='default',
                 model_config={}, tracker=Tracker(), debug=None, simplified_colour_count=False,
                 inference_type=InferenceType.SearchInference, max_inference_size=-1):

        super(PGMCorrectingAgent, self).__init__(world, colour_models, rule_beliefs,
                                                 domain_file, teacher, threshold,
                                                 update_negative, update_once, colour_model_type,
                                                 model_config, tracker)

        self.debug = defaultdict(lambda: False)
        if debug is not None:
            self.debug.update(debug)

        self.pgm_model = PGMModel(inference_type=inference_type, max_inference_size=max_inference_size)
        self.time = 0
        self.last_correction = -1
        self.marks = defaultdict(list)
        self.previous_corrections = []
        self.previous_args = {}
        self.simplified_colour_count = simplified_colour_count
        self.inference_times = []

    def __repr__(self):
        return "PGMCorrectingAgent"

    def __str__(self):
        return self.__repr__()

    def update_goal(self):
        rule_probs = self.pgm_model.get_rule_probs(update_prior=True)
        rules = []
        for rule, p in rule_probs.items():

            if p > 0.5:
                rule = Rule.from_string(rule)
                rules.append(rule.to_formula())
                if self.debug['show_rules']:
                    print(f'Added rule {rule} to goal')

        self.goal = goals.goal_from_list(rules, self.domain_file)

    def no_correction(self, action, args):
        if action.lower() == 'unstack':
            return
        # print("no correction", self.time)
        self.time += 1
        # print(args, self.marks.keys())
        if args[0] in self.marks.keys() or args[1] in self.marks.keys():
            marks = set(self.marks[args[0]] + self.marks[args[1]])
            marks = [rule for rule in marks if isinstance(rule, RedOnBlueRule)]
            if len(marks) == 0:
                return
            self.pgm_model.add_no_correction(args, self.time, marks)
            data = self.get_colour_data(args)
            corr = corr_variable_name(self.time)
            data[corr] = 0
            # print(data)
            time = self.pgm_model.observe(data)
            self.inference_times.append(time)
            # self.pgm_model.infer()
            self.update_cms()

    def get_relevant_data(self, args, message):
        args_for_model = args.copy()

        if message.T == 'table':
            args_for_model += [message.o3]
        data = self.get_colour_data(args_for_model)

        corr = corr_variable_name(self.time)

        data[corr] = 1

        if 't' in args[1] and message.o2 is not None:
            blue = colour_variable_name(message.o2[0], args[1])
            data[blue] = 0
        return data

    def update_model(self, user_input, args):
        message = read_sentence(user_input)
        if message.T == 'same reason':

            for prev_corr, prev_time in self.previous_corrections[::-1]:
                if 'same reason' not in prev_corr:
                    break
            print("Same reason", self.time)
            print(prev_corr, prev_time)

            prev_message = read_sentence(prev_corr, use_dmrs=False)
            user_input = prev_corr
            violations = self.build_same_reason(prev_message, args, prev_time)
            data = self.get_relevant_data(args, prev_message)
            message = prev_message

        elif message.T in ['tower', 'table']:

            if isinstance(self.teacher, FaultyTeacherAgent) and message.T == 'table':
                red = message.o1[0]
                blue = message.o2[0]
                rules = Rule.generate_red_on_blue_options(red, blue)

                tower_name = args[-1] if 't' in args[-1] else None

                red_cm = self.add_cm(red)
                blue_cm = self.add_cm(blue)

                objects = self.world.state.get_objects_in_tower(tower_name)

                violations = self.pgm_model.create_uncertain_table_model(rules, red_cm, blue_cm,
                                                                         args + [message.o3],
                                                                         objects, self.time)

                data = self.get_colour_data(objects + [message.o3])
                corr = corr_variable_name(self.time)
                data[corr] = 1
            else:
                data = self.get_relevant_data(args, message)
                violations = self.build_pgm_model(message, args)
                corr = corr_variable_name(self.time)
                data[corr] = 1

        elif 'partial.neg' == message.T:
            print("partial.neg", self.time)
            for i, (prev_corr, prev_time) in enumerate(self.previous_corrections[::-1]):
                if message.o1 in prev_corr:
                    break

            user_input = prev_corr
            prev_message = read_sentence(prev_corr, use_dmrs=False)

            violations = self.build_pgm_model(prev_message, args)
            prev_args = self.previous_args[prev_time]



            if message.o1 == prev_message.o1[0]:
                curr_negation = f'{message.o1}({args[0]})'
                prev_negation = f'{message.o1}({prev_args[0]})'
            else:
                curr_negation = f'{message.o1}({args[1]})'
                prev_negation = f'{message.o1}({prev_args[1]})'


            data = self.get_relevant_data(args, prev_message)
            data[curr_negation] = 0
            data[prev_negation] = 0
            print(data)
        elif 'colour count' == message.T:
            colour_name = message.o1
            number = message.o2
            rule = ColourCountRule(colour_name, number)
            cm = self.add_cm(colour_name)
            tower_name = args[-1]
            objects = self.world.state.get_objects_in_tower(tower_name)
            top, _ = self.world.state.get_top_two(tower_name)
            if self.simplified_colour_count:
                objects = [top]
            violations = self.pgm_model.add_colour_count_correction(rule, cm, objects, self.time)
            data = self.get_colour_data(objects)
            corr = f"corr_{self.time}"
            data[corr] = 1
            blue_top_obj = f"{colour_name}({top})"
            data[blue_top_obj] = 1
        elif 'colour count+tower' == message.T:
            colour_name = message.o1
            number = message.o2
            colour_count = ColourCountRule(colour_name, number)
            red_cm = self.add_cm(message.o3.o1[0])
            blue_cm = self.add_cm(message.o3.o2[0])
            red_on_blue = Rule.generate_red_on_blue_options(message.o3.o1[0], message.o3.o2[0])
            tower_name = args[-1]
            top, _ = self.world.state.get_top_two(tower_name)
            objects = self.world.state.get_objects_in_tower(tower_name)
            if self.simplified_colour_count:
                objects = [top]
            violations = self.pgm_model.add_cc_and_rob(colour_count, red_on_blue, red_cm,
                                                       blue_cm, objects, top, self.time)
            data = self.get_colour_data(objects)
            corr = f"corr_{self.time}"
            data[corr] = 1

        self.previous_corrections.append((user_input, self.time))
        self.previous_args[self.time] = args

        return violations, data, message

    def ask_question(self, message, args):

        if message.T in ['table', 'tower']:
            question = f'Is the top object {message.o1[0]}?'
            # dialogue.info('R: ' + question)
            print(question)
            if isinstance(message.o1, list):
                red = f'{message.o1[0]}({args[0]})'
            else:
                red = f'{message.o1}({args[0]}'

            if len(args) == 3:
                tower = args[-1]
            else:
                tower = None
            answer = self.teacher.answer_question(question, self.world, tower)
            # dialogue.info("T: " + answer)
            print(answer)
            bin_answer = int(answer.lower() == 'yes')
            time = self.pgm_model.observe({red: bin_answer})
            self.inference_times.append(time)

    def mark_block(self, most_likely_violation, message, args):
        rule = Rule.from_violation(most_likely_violation)

        if isinstance(rule, list):
            colour_count, red_on_blue = rule
            red_on_blue_options = Rule.generate_red_on_blue_options(red_on_blue.c1, red_on_blue.c2)
            self.marks[args[0]] += red_on_blue_options

        elif isinstance(rule, RedOnBlueRule):
            rules = Rule.generate_red_on_blue_options(rule.c1, rule.c2)
            if rule.rule_type == 1:
                if message.T == 'tower':
                    self.marks[args[0]] += rules
                elif message.T == 'table':
                    self.marks[args[1]] += rules
                    self.marks[message.o3] += rules
            elif rule.rule_type == 2:
                if message.T == 'tower':
                    self.marks[args[1]] += rules
                elif message.T == 'table':
                    self.marks[args[0]] += rules
                    self.marks[message.o3] += rules
        elif isinstance(rule, ColourCountRule):
            rules = [rule]
            objects_in_tower = self.world.state.get_objects_in_tower(args[-1])
            for obj in objects_in_tower:
                self.marks[obj] += rules
        else:
            raise NotImplementedError("Invalid or non implemented rule type")

    def get_correction(self, user_input, actions, args, test=False):
        self.time += 1
        self.last_correction = self.time

        not_on_xy = pddl_functions.create_formula('on', args[:2], op='not')
        self.tmp_goal = goals.update_goal(self.tmp_goal, not_on_xy)

        violations, data, message = self.update_model(user_input, args)

        time = self.pgm_model.observe(data)
        self.inference_times.append(time)

        q = self.pgm_model.query(list(violations))

        print(q)

        if max(q.values()) < self.threshold:

            if message.T in ['table', 'tower']:
                self.ask_question(message, args)
                q = self.pgm_model.query(list(violations))

        most_likely_violation = max(q, key=q.get)

        self.mark_block(most_likely_violation, message, args)

        self.update_cms()
        self.update_goal()
        self.world.back_track()

    def update_cms(self):
        colours = self.pgm_model.get_colour_predictions()
        for cm in self.colour_models.values():
            cm.reset()

        for colour, p in colours.items():

            colour, arg = get_predicate(colour)
            arg = arg[0]
            if 't' in arg:
                continue
            fx = self.get_colour_data([arg])[f'F({arg})']

            if p > 0.7:
                self.colour_models[colour].update(fx, p)
                if self.debug['show_cm_update']:
                    print(f'Updated {colour} model with: {fx} at probability {p}')
            elif self.debug['show_cm_update']:
                print(f'Did not update {colour} model with: {fx} at probability {p}')

        if self.debug['evaluate_cms']:
            for colour, cm in self.colour_models.items():
                print(f'Evaluating {colour} model')
                evaluate_colour_model(cm)

    def add_cm(self, colour_name):
        try:
            red_cm = self.colour_models[colour_name]
        except KeyError:
            red_cm = KDEColourModel(colour_name, **self.model_config)
            self.colour_models[colour_name] = red_cm
        return red_cm

    def build_same_reason(self, message, args, prev_time):
        violations = self.build_pgm_model(message, args)
        rules = Rule.generate_red_on_blue_options(message.o1[0], message.o2[0])
        previous_violations = [f'V_{prev_time}({str(rule)})' for rule in rules]
        self.pgm_model.add_same_reason(violations, previous_violations)
        return violations

    def build_pgm_model(self, message, args):

        rules = Rule.generate_red_on_blue_options(message.o1[0], message.o2[0])
        red_cm = self.add_cm(message.o1[0])
        blue_cm = self.add_cm(message.o2[0])


        if isinstance(self.teacher, FaultyTeacherAgent):
            table_empty = len(self.world.state.get_objects_on_table()) == 0
        else:
            table_empty = False

        is_table_correction = message.T == 'table'
        if message.T == 'tower':
            correction_type = CorrectionType.TOWER
        elif message.T == 'table':
            if isinstance(self.teacher, FaultyTeacherAgent):
                correction_type = CorrectionType.UNCERTAIN_TABLE
            else:
                correction_type = CorrectionType.TABLE
        else:
            raise ValueError(f"Invalid message type, expected tower or table, not {message.T}")
        print("table correction?", is_table_correction)
        if is_table_correction:
            args = args[:2]
            args += [message.o3]

        violations = self.pgm_model.extend_model(rules, red_cm, blue_cm, args, self.time,
                                                 correction_type=correction_type, table_empty=table_empty)

        return violations

    def get_colour_data(self, args):
        observation = self.world.sense()
        colour_data = observation.colours
        data = {f'F({arg})': colour_data[arg] for arg in args if 't' not in arg}
        return data

    def new_world(self, world):

        self.marks = defaultdict(list)
        self.time = 0
        self.last_correction = -1

        self.pgm_model.reset()
        for cm in self.colour_models.values():
            cm.fix()

        self.teacher.reset()
        super(PGMCorrectingAgent, self).new_world(world)


class ClassicalAdviceBaseline(PGMCorrectingAgent):

        def __init__(self, world, colour_models=None, rule_beliefs=None,
                     domain_file='blocks-domain.pddl', teacher=None, threshold=0.7,
                     update_negative=True, update_once=True, colour_model_type='default',
                     model_config={}, tracker=Tracker()):

            super(ClassicalAdviceBaseline, self).__init__(world, colour_models, rule_beliefs,
                     domain_file, teacher, threshold,
                     update_negative, update_once, colour_model_type,
                     model_config, tracker)

            rules = Rule.get_rules(self.problem.goal)
            for rule in rules:
                r = Rule.from_formula(rule)
                c1 = r.c1
                c2 = r.c2

                red_cm = self.add_cm(c1)
                blue_cm = self.add_cm(c2)
                rules = Rule.generate_red_on_blue_options(c1, c2)

                #self.pgm_model.add_rules(rules, red_cm, blue_cm)

        def do_correction(self, message, args):
            if message.o3 is None:
                violations = self.pgm_model.add_no_correction(args, self.time)
            else:
                args.append(message.o3)
                violations = self.pgm_model.add_no_table_correciton(args, self.time)
            return violations

        def get_correction(self, user_input, actions, args, test=False):
            self.time += 1
            self.last_correction = self.time
            #print(self.time)

            not_on_xy = pddl_functions.create_formula('on', args, op='not')
            self.tmp_goal = goals.update_goal(self.tmp_goal, not_on_xy)
            #print(self.time)
            args_for_model = args.copy()
            #print(actions, args, user_input)
            #print(read_sentence(user_input, use_dmrs=False))
            message = read_sentence(user_input, use_dmrs=False)
            args_for_model = args.copy()

            data = self.get_relevant_data(args, message)
            violations = self.do_correction(message, args)

            self.pgm_model.observe(data)



            # q = self.pgm_model.query(list(violations))

            #m_r1 = q[violations[0]]
            #m_r2 = q[violations[1]]
            # q = self.pgm_model.infer(list(violations))
            #
            # m_r1 = q[violations[0]].values[1]
            # m_r2 = q[violations[1]].values[1]
            #print(m_r1, m_r2)

            # print(user_input)
            #
            # if message.T == 'same reason':
            #     message = prev_message
            #
            # if max(q.values()) < self.threshold:
            #     # if message.T == 'same reason':
            #     #     message = prev_message
            #
            #     if message.T in ['table', 'tower']:
            #         question = 'Is the top object {}'.format(message.o1[0])
            #         #dialogue.info('R: ' + question)
            #         print(question)
            #         red = '{}({})'.format(message.o1[0], args[0])
            #         answer = self.teacher.answer_question(question, self.world)
            #         #dialogue.info("T: " + answer)
            #         print(answer)
            #         bin_answer = int(answer.lower() == 'yes')
            #         self.pgm_model.observe({red:bin_answer})
            #         q = self.pgm_model.query(list(violations))
            #         m_r1 = q[violations[0]]
            #         m_r2 = q[violations[1]]



            #self.update_rules()
            #TODO fix this to deal with the fact that there might be more than 2 violations
            #
            # most_likely_violation = max(q, key=q.get)
            # c1, c2, rule_type = get_rule_type(most_likely_violation)
            #
            #
            #
            # if rule_type == 'r1':
            #     if message.T == 'tower':
            #         self.marks.add(args[0])
            #     elif message.T == 'table':
            #         self.marks.add(args[1])
            #         self.marks.add(message.o3)
            # elif rule_type == 'r2':
            #     if message.T == 'tower':
            #         self.marks.add(args[1])
            #     elif message.T == 'table':
            #         self.marks.add(args[0])
            #         self.marks.add(message.o3)


            self.update_cms()
            self.update_goal()
            self.world.back_track()
            self.previous_corrections.append((user_input, self.time))
            self.previous_args[self.time] = args
            #super(PGMCorrectingAgent, self).get_correction(user_input, actions, args, test=test)
