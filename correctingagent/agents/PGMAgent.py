
from correctingagent.agents.agents import CorrectingAgent, Tracker
from correctingagent.pddl.goal_updates import create_goal_options
from correctingagent.pddl import goal_updates, pddl_functions
from correctingagent.models.pgmmodels import PGMModel, create_rules, create_neg_rule
from correctingagent.models.prob_model import KDEColourModel
from collections import namedtuple
import re
from .teacher import get_rules, get_relevant_colours


def test_rule_type():
    violation = 'V(all x.({}(x) -> exists y. ({}(y) & on(x,y))))'.format('red', 'blue')
    violation2 = 'V(all y.({}(y) -> exists x. ({}(x) & on(x,y))))'.format('blue', 'red')

    assert(get_rule_type(violation) == ('red', 'blue', 'r1'))
    assert(get_rule_type(violation2) == ('red', 'blue', 'r2'))


def get_violation_type(violation):
    rule = re.sub(r"V_[0-9]+\(", '', violation)[:-1]
    return get_rule_type(rule)

def get_rule_type(rule):
    # print(violation)
    # rule = re.sub(r"V_[0-9]+\(", '', violation)[:-1]
    # print(rule)
    #rule = violation.replace('V_(', '')[:-1]
    if rule[:3] != 'all':
        raise NotImplemented('Not implemented non put red on blue rule')
    else:
        red, blue, on = split_rule(rule)
        red_colour, o1 = get_predicate(red)
        blue_colour, o2 = get_predicate(blue)
        x, y = get_predicate_args(on)

        if o1[0] == x:
            rule_type = 'r1'
            return red_colour, blue_colour, rule_type
        elif o2[0] == x:
            rule_type = 'r2'
            return blue_colour, red_colour, rule_type
        else:
            raise ValueError('something went wrong')

def split_rule(rule):
    bits = rule.split('.(')
    red = bits[1].split('->')[0].strip()
    bits2 = bits[1].split(' (')
    blue, on = bits2[1].split('&')
    blue = blue.strip()
    on = on.replace('))', '').strip()
    return [red, blue, on]

def get_predicate_name(predicate):
    return predicate.split('(')[0]

def get_predicate_args(predicate):
    args = predicate.split('(')[1].replace(')', '')
    return [arg.strip() for arg in args.split(',')]

def get_predicate(predicate):
    pred = predicate.split('(')[0]
    args = predicate.split('(')[1].replace(')', '')
    args = [arg.strip() for arg in args.split(',')]
    return pred, args

def rule_to_pddl(rule):
    rule_split = split_rule(rule)
    red, o1 = get_predicate(rule_split[0])
    blue, o2 = get_predicate(rule_split[1])
    on, (x, y) = get_predicate(rule_split[2])


    if x == o1[0] and y == o2[0]:
        r1, r2 = create_goal_options([red], [blue])
        return r1
    if x == o2[0] and y == o1[0]:
        r1, r2 = create_goal_options([blue], [red])
        return r2
    else:
        raise ValueError('Should not get here')

Message = namedtuple('Message', ['rel', 'o1', 'o2', 'T', 'o3'])


def read_sentence(sentence, use_dmrs=False):
    sentence = sentence.lower()
    if sentence == 'no':
        return Message(None, None, None, 'no', None)
    sentence = sentence.replace('no,', '')
    o3 = None
    T = 'tower'
    if "same reason" in sentence:
        return Message(None, None, None, 'same reason', None)
    elif "you put" in sentence:
        T = 'evidence'
        sentence = sentence.replace('you', '').strip()
        sentence = sentence.replace('a ', '')

    # elif "that is" in sentence:
    #
    #     sentence = sentence.replace('that is', '').strip()
    #     if 'still' in sentence or 'either' in sentence:
    #         sentence = sentence.replace('still', '')
    #         sentence = sentence.replace('either', '').strip()
    #         T = 'still.'
    #     else:
    #         T = ''
    #     if 'not' in sentence:
    #         T = T + 'partial.neg'
    #         sentence = sentence.replace('not', '').strip()
    #     else:
    #         T = T + 'partial'
    #     return Message(None, sentence, None, T, None)


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


class PGMCorrectingAgent(CorrectingAgent):
    def __init__(self, world, colour_models=None, rule_beliefs=None,
                 domain_file='blocks-domain.pddl', teacher=None, threshold=0.7,
                 update_negative=True, update_once=True, colour_model_type='default',
                 model_config={}, tracker=Tracker()):



        super(PGMCorrectingAgent, self).__init__(world, colour_models, rule_beliefs,
                 domain_file, teacher, threshold,
                 update_negative, update_once, colour_model_type,
                 model_config, tracker)

        self.pgm_model = PGMModel()
        self.time = 0
        self.last_correction = -1
        self.marks = set()
        self.previous_corrections = []
        self.previous_args = []


    def update_goal(self):
        rule_probs = self.pgm_model.get_rule_probs(update_prior=True)
        rules = []
        for rule, p in rule_probs.items():
            #print(rule, p)

            if p > 0.5:
                rules.append(rule_to_pddl(rule))
                #print(rule)

        self.goal = goal_updates.goal_from_list(rules)
        #print(self.goal.asPDDL())



    def no_correction(self, action, args):
        self.time += 1
        #print(self.time)
        if args[0] in self.marks or args[1] in self.marks:
            #print('we did the thing')
            self.pgm_model.add_no_correction(args, self.time)
            data = self.get_colour_data(args)
            corr = 'corr_{}'.format(self.time)
            data[corr] = 0
            self.pgm_model.observe(data)
            self.pgm_model.infer()
            self.update_cms()


    def get_relevant_data(self, args, message):
        args_for_model = args.copy()

        if message.T == 'table':
            args_for_model += [message.o3]
        data = self.get_colour_data(args_for_model)

        corr = 'corr_{}'.format(self.time)

        data[corr] = 1

        # red = '{}({})'.format(message.o1[0], args[0])

        # if message.T == 'table':
        #     redo3 = '{}({})'.format(message.o1[0], message.o3)
        #     blueo3 = '{}({})'.format(message.o2[0], message.o3)
        #     # self.marks.add(message.o3)

        if 't' in args[1] and message.o2 is not None:
            blue = '{}({})'.format(message.o2[0], args[1])
            data[blue] = 0
        return data


    def get_correction(self, user_input, actions, args, test=False):
        self.time += 1
        self.last_correction = self.time
        #print(self.time)


        not_on_xy = pddl_functions.create_formula('on', args, op='not')
        self.tmp_goal = goal_updates.update_goal(self.tmp_goal, not_on_xy)
        #print(self.time)
        args_for_model = args.copy()
        #print(actions, args, user_input)
        #print(read_sentence(user_input, use_dmrs=False))
        message = read_sentence(user_input, use_dmrs=False)
        args_for_model = args.copy()
        print(message)
        variable_clamp = []
        if message.T == 'same reason':
            for prev_corr, prev_time in self.previous_corrections[::-1]:
                if 'same reason' not in prev_corr:
                    break


            prev_message = read_sentence(prev_corr, use_dmrs=False)
            violations = self.build_same_reason(prev_message, args, prev_time)


            data = self.get_relevant_data(args, prev_message)

        elif message.T == 'evidence':
            violations = self.pgm_model.add_no_correction(args, self.time)
            data = self.get_colour_data(args)
            corr = 'corr_{}'.format(self.time)
            data[corr] = 1
            red = '{}({})'.format(message.o1[0], args[0])
            blue = '{}({})'.format(message.o2[0], args[1])
            data[red] = 1
            data[blue] = 1

        elif message.T == 'no':
            violations = self.pgm_model.add_no_correction(args, self.time)
            data = self.get_colour_data(args)
            corr = 'corr_{}'.format(self.time)
            data[corr] = 1

        elif message.T in ['tower', 'table']:

            data = self.get_relevant_data(args, message)
            violations = self.build_pgm_model(message, args)

        elif message.T in ['neg', 'table.neg']:

            data =  self.get_relevant_data(args, message)
            violations = self.build_neg_model(message, args)

        # elif 'partial' in message.T:
        #     data = self.get_relevant_data(args, message)
        #     violations = self.pgm_model.add_no_correction(args, self.time)
        #     colour = message.o1
        #     redo1 = '{}({})'.format(colour, args[0])
        #     redo2 = '{}({})'.format(colour, args[1])
        #     if 'neg' in message.T:
        #         variable_clamp = [{redo1:0}, {redo2:0}]
        #     else:
        #         variable_clamp = [{redo1:1}, {redo2:1}]

        elif 'partial.neg' == message.T:
            data = self.get_relevant_data(args, message)
            for i, (prev_corr, prev_time) in enumerate(self.previous_corrections[::-1]):
                if message.o1 in prev_corr:
                    break

            prev_message = read_sentence(prev_corr, use_dmrs=False)
            violations = self.build_pgm_model(prev_message, args)
            prev_args, prev_args_time = self.previous_args[i]
            assert(prev_args_time == prev_time)


            if message.o1 == prev_message.o1:
                curr_negation = '{}({})'.format(message.o1, args[0])
                prev_negation = '{}({})'.format(message.o1, prev_args[0])
            else:
                curr_negation = '{}({})'.format(message.o1, args[1])
                prev_negation = '{}({})'.format(message.o1, prev_args[1])


            data = self.get_relevant_data(args, prev_message)
            data[curr_negation] = 0
            data[prev_negation] = 0
        else:
            raise NotImplemented('This type of correction is not implemented: {}'.format(message.T))



        self.pgm_model.observe(data)

        if variable_clamp:
            self.pgm_model.observe_uncertain(variable_clamp)

        q = self.pgm_model.query(list(violations))

        #m_r1 = q[violations[0]]
        #m_r2 = q[violations[1]]
        # q = self.pgm_model.infer(list(violations))
        #
        # m_r1 = q[violations[0]].values[1]
        # m_r2 = q[violations[1]].values[1]
        #print(m_r1, m_r2)

        print(user_input)

        if message.T == 'same reason':
            message = prev_message

        if max(q.values()) < self.threshold:
            # if message.T == 'same reason':
            #     message = prev_message

            if message.T in ['table', 'tower']:
                question = 'Is the top object {}'.format(message.o1[0])
                #dialogue.info('R: ' + question)
                print(question)
                red = '{}({})'.format(message.o1[0], args[0])
                answer = self.teacher.answer_question(question, self.world)
                #dialogue.info("T: " + answer)
                print(answer)
                bin_answer = int(answer.lower() == 'yes')
                self.pgm_model.observe({red:bin_answer})
                q = self.pgm_model.query(list(violations))
                m_r1 = q[violations[0]]
                m_r2 = q[violations[1]]



        #self.update_rules()
        #TODO fix this to deal with the fact that there might be more than 2 violations

        most_likely_violation = max(q, key=q.get)
        c1, c2, rule_type = get_rule_type(most_likely_violation)



        if rule_type == 'r1':
            if message.T == 'tower':
                self.marks.add(args[0])
            elif message.T == 'table':
                self.marks.add(args[1])
                self.marks.add(message.o3)
        elif rule_type == 'r2':
            if message.T == 'tower':
                self.marks.add(args[1])
            elif message.T == 'table':
                self.marks.add(args[0])
                self.marks.add(message.o3)


        self.update_cms()
        self.update_goal()
        self.world.back_track()
        self.previous_corrections.append((user_input, self.time))
        self.previous_args.append((args, self.time))
        #super(PGMCorrectingAgent, self).get_correction(user_input, actions, args, test=test)



    def update_cms(self):
        colours = self.pgm_model.get_colour_predictions()
        for cm in self.colour_models.values():
            cm.reset()


        for colour, p in colours.items():

            colour, arg = get_predicate(colour)
            arg = arg[0]
            if 't' in arg:
                continue
            fx = self.get_colour_data([arg])['F({})'.format(arg)]
            #print(colour, p)
            if p > 0.7:
                self.colour_models[colour].update(fx, p)


    def add_cm(self, colour_name):
        try:
            red_cm = self.colour_models[colour_name]
        except KeyError:
            red_cm = KDEColourModel(colour_name, **self.model_config)
            self.colour_models[colour_name] = red_cm
        return red_cm

    def build_same_reason(self, message, args, prev_time):
        violations = self.build_pgm_model(message, args)
        rules = create_rules(message.o1[0], message.o2[0])
        previous_violations = ['V_{}({})'.format(prev_time, rule) for rule in rules]
        self.pgm_model.add_same_reason(violations, previous_violations)
        return violations


    def build_pgm_model(self, message, args):

        rules = create_rules(message.o1[0], message.o2[0])
        red_cm = self.add_cm(message.o1[0])
        blue_cm = self.add_cm(message.o2[0])

        if message.T == 'tower':
            violations = self.pgm_model.create_tower_model(rules, red_cm, blue_cm, args, self.time)
        elif message.T == 'table':
            violations = self.pgm_model.create_table_model(rules, red_cm, blue_cm, args + [message.o3], self.time)
        #print(rules)
        return violations


    def build_neg_model(self, message, args):
        rule = create_neg_rule(message.o1[0], message.o2[0])
        red_cm = self.add_cm(message.o1[0])
        blue_cm = self.add_cm(message.o2[0])


        if message.T == 'neg':
            violations = self.pgm_model.create_negative_model(rule, red_cm, blue_cm, args, self.time)
        if message.T == 'table.neg':
            violations = self.pgm_model.create_table_neg_model(rule, red_cm, blue_cm, args + [message.o3], self.time)
        return violations

    def get_colour_data(self, args):
        observation = self.world.sense()
        colour_data = observation.colours
        data = {'F({})'.format(arg): colour_data[arg] for arg in args if 't' not in arg}
        return data


    def new_world(self, world):

        self.marks = set()
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




            super(PGMCorrectingAgent, self).__init__(world, colour_models, rule_beliefs,
                     domain_file, teacher, threshold,
                     update_negative, update_once, colour_model_type,
                     model_config, tracker)

            rules = get_rules(self.problem.goal)
            for rule in rules:
                c1, c2, impl = get_relevant_colours(rule)
                if impl == "not":
                    raise NotImplementedError('I have not implemented this baseline for "not" rules')

                red_cm = self.add_cm(c1)
                blue_cm = self.add_cm(c2)
                rules = create_rules(c1, c2)

                self.pgm_model.add_rules(rules, red_cm, blue_cm)


        def do_correction(self, message, args):
            if message.o3 is None:
                violations = self.pgm_model.add_no_correction(args, self.time)
            else:
                args.append(message.o3)
                violations = self.pgm_model.add_no_correction(args, self.time)
            return violations

        def get_correction(self, user_input, actions, args, test=False):
            self.time += 1
            self.last_correction = self.time
            #print(self.time)


            not_on_xy = pddl_functions.create_formula('on', args, op='not')
            self.tmp_goal = goal_updates.update_goal(self.tmp_goal, not_on_xy)
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
            self.previous_args.append((args, self.time))
            #super(PGMCorrectingAgent, self).get_correction(user_input, actions, args, test=test)
