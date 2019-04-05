from ..pddl import pddl_functions
from collections import namedtuple
import random
from ..util.colour_dict import colour_dict

def tower_correction(obj1, obj2):
    return "no, put {} blocks on {} blocks".format(obj1, obj2)

def table_correction(obj1, obj2, obj3):
    return "No, now you cannot put {} in the tower because you must put {} blocks on {} blocks".format(obj3, obj1, obj2)

def not_correction(obj1, obj2):
    return "No, don't put {} blocks on blue blocks".format(obj1, obj2)

def table_not_correction(obj1, obj2, obj3):
    return "No, now you cannot put {} in the tower because you cannot put {} blocks on {} blocks".format(obj3, obj1, obj2)

def get_top_two(world_):
    r = world_.sense().relations
    for o in r.keys():
        if 'clear' in r[o] and 'in-tower' in r[o]:
            o1 = o
    on = r[o1]['on']
    o2 = on.args.args[1].arg_name
    return o1, o2

def get_rules(goal):
    for f in goal.subformulas:
        if "in-tower" not in f.asPDDL():
            yield f

def get_name(formula):
    return formula.get_predicates(True)[0].name

def get_args(formula):
    return list(map(lambda x: x.arg_name, formula.get_predicates(True)[0].args.args))

def get_relevant_colours(rule):
    if rule.op == 'not':
        exists = neg_rule.subformulas[0]
        and_ = exists.subformulas[0]
        c1, c2, on = and_.subformulas
        return get_name(c1), get_name(c2), 'not'
    else:

        or_ = rule.subformulas[0]
        r1, r2 = or_.subformulas
        colour1 = (r1.subformulas[0])
        colour2 = (list(filter(lambda x: get_name(x) != 'on', r2.subformulas[0].subformulas))[0])
        on = list(filter(lambda x: get_name(x) == 'on', r2.subformulas[0].subformulas))[0]
        o1, o2 = on.get_predicates(True)[0].args.args
        out = (o1, o2)
        c1 = list(filter(lambda x: get_args(x)[0] == o1.arg_name, [colour1, colour2]))[0]
        c2 = list(filter(lambda x: get_args(x)[0] == o2.arg_name, [colour1, colour2]))[0]
        return get_name(c1), get_name(c2), get_name(colour1) # on(c1, c2) colour1 -> colour2

def check_rule_violated(rule, world_):
    c1, c2, impl = get_relevant_colours(rule)
    o1, o2 = get_top_two(world_)
    state = world_.problem.initialstate
    f1 = pddl_functions.create_formula(c1, [o1])
    f2 = pddl_functions.create_formula(c2, [o2])
    c1_o1 = pddl_functions.predicate_holds(f1.get_predicates(True)[0], state)
    c2_o2 = pddl_functions.predicate_holds(f2.get_predicates(True)[0], state)

    if c1_o1 and c2_o2 and impl == 'not':
        return True
    elif c1_o1 and not c2_o2 and c1 == impl:
        return True
    elif not c1_o1 and c2_o2 and c2 == impl:
        return True
    else:
        return False

def check_table_rule_violation(rule, world_):
    c1, c2, impl = get_relevant_colours(rule)
    o1, o2 = get_top_two(world_)
    state = world_.problem.initialstate
    f1 = pddl_functions.create_formula(c1, [o1])
    f2 = pddl_functions.create_formula(c2, [o2])
    f3 = pddl_functions.create_formula(c2, [o1])
    c1_o1 = pddl_functions.predicate_holds(f1.get_predicates(True)[0], state)
    c2_o2 = pddl_functions.predicate_holds(f2.get_predicates(True)[0], state)
    c2_o1 = pddl_functions.predicate_holds(f3.get_predicates(True)[0], state)
    table_objs = blocks_on_table(world_)
    c1_count = count_coloured_blocks(c1, table_objs, state)
    c2_count = count_coloured_blocks(c2, table_objs, state)
    if impl == 'not' and (c1_count + c2_count) == len(table_objs) and c2_o1:
        return get_block_with_colour(c1, table_objs, state)
    if not c1_o1 and c2_o2 and c1 == impl:
        if c1_count > c2_count:
            return get_block_with_colour(c1, table_objs, state)
    if c1_o1 and not c2_o2 and c2 == impl:
        if c2_count > c1_count:
            return get_block_with_colour(c2, table_objs, state)
    return False

def blocks_on_table(world_):
    r = world_.sense().relations
    objs = []
    for o in r.keys():
        if 'on-table' in r[o] and 'clear' in r[o]:
            objs.append(o)
    return objs

def count_coloured_blocks(colour, objects, state):
    count = 0
    for o in objects:
        c_o = pddl_functions.create_formula(colour, [o]).get_predicates(True)[0]
        if pddl_functions.predicate_holds(c_o, state):
            count += 1
    return count

def get_block_with_colour(colour, objects, state):
    for o in objects:
        c_o = pddl_functions.create_formula(colour, [o]).get_predicates(True)[0]
        if pddl_functions.predicate_holds(c_o, state):
            return o
    return

def teacher_correction(w):
    if not w.test_failure():
        return ""
    #Reasons for failure:
    # a->b, a -b
    # b-> a b -a
    rules = list(get_rules(w.problem.goal))
    for r in rules:
        if check_rule_violated(r, w):
            #print(r.asPDDL())
            c1, c2, _ = get_relevant_colours(r)
            return tower_correction(c1, c2)
    for r in rules:
        o3 = check_table_rule_violation(r, w)
        if o3:
            c1, c2, _ = get_relevant_colours(r)
            return table_correction(c1, c2, o3)


class Teacher(object):

    def correction(self, world_):
        raise NotImplementedError()

    def answer_question(self, question, world_):
        raise NotImplementedError()


class HumanTeacher(Teacher):

    def correction(self, world_):
        return input('Correction?')

    def answer_question(self, question, world_):
        return input(question)



class TeacherAgent(Teacher):

    def correction(self, w):
        if not w.test_failure():
            return ""
        #Reasons for failure:
        # a->b, a -b
        # b-> a b -a
        rules = list(get_rules(w.problem.goal))
        for r in rules:
            if check_rule_violated(r, w):
                #print(r.asPDDL())
                c1, c2, _ = get_relevant_colours(r)
                return tower_correction(c1, c2)
        for r in rules:
            o3 = check_table_rule_violation(r, w)
            if o3:
                c1, c2, _ = get_relevant_colours(r)

                return table_correction(c1, c2, o3)

    def answer_question(self, question, world_):
        if "Is the top object" in question:
            colour = question.replace("Is the top object", '').replace("?", '').strip()
            o1, o2 = get_top_two(world_)
            o1_colour = pddl_functions.create_formula(colour, [o1]).get_predicates(True)[0]
            o1_is_colour = pddl_functions.predicate_holds(o1_colour, world_.problem.initialstate)
            if o1_is_colour:
                return "yes"
            else:
                return "no"

Correction = namedtuple('Correction', ['rule', 'c1', 'c2', 'args', 'sentence'])


def get_colour(obj, state):
    preds = pddl_functions.get_predicates(obj, state)
    colour = [pred.name for pred in preds if pred.name in colour_dict.keys()]
    if 't' in obj:
        return None
    else:
        return colour[0]



class ExtendedTeacherAgent(TeacherAgent):

    def __init__(self):
        self.previous_correction = None
        self.rules_corrected = set()
        self.dialogue_history = []

    def reset(self):
        self.previous_correction = None
        self.rules_corrected = set()
        self.dialogue_history = []

    def correction(self, w):
        if not w.test_failure():
            return ""

        possible_corrections = []
        rules = list(get_rules(w.problem.goal))
        for r in rules:
            if check_rule_violated(r, w):
                #print(r.asPDDL())
                c1, c2, impl = get_relevant_colours(r)
                o1, o2 = get_top_two(w)
                if impl == 'not':
                    corr = Correction(r, c1, c2, [o1, o2], not_correction(c1, c2))
                else:
                    corr = Correction(r, c1, c2, [o1, o2], tower_correction(c1, c2))
                    possible_corrections.append(corr)

        for r in rules:
            o3 = check_table_rule_violation(r, w)
            if o3:
                c1, c2, impl = get_relevant_colours(r)
                o1, o2 = get_top_two(w)
                if impl == 'not':
                    corr = Correction(r, c1, c2, [o1, o2], table_not_correction(c1, c2, o3))
                else:
                    corr = Correction(r, c1, c2, [o1, o2, o3], table_correction(c1, c2, o3))
                    possible_corrections.append(corr)


        possible_corrections = self.generate_possible_corrections(possible_corrections, w)
        return self.select_correction(possible_corrections)


    def generate_possible_corrections(self, violations, world_):

        possible_sentences = []
        # if self.previous_correction is not None:
        for correction in violations:




            if self.previous_correction is not None and self.previous_correction.rule == correction.rule:

                for previous_corr in self.dialogue_history[::-1]:
                    if 'now you cannot put' not in  previous_corr.sentence and 'same reason' not in previous_corr.sentence:
                        colours = [previous_corr.c1, previous_corr.c2]
                        if correction.c1 in colours or correction.c2 in colours:

                            if correction.c1 == colours[0]:
                                current_colour = get_colour(correction.args[0], world_.problem.initialstate)
                                prev_colour = get_colour(previous_corr.args[0], world_.problem.initialstate)
                                if current_colour != correction.c1 and prev_colour != correction.c1:
                                    possible_sentences.append(('no, that is not {} again'.format(correction.c1), correction))
                            if correction.c2 == colours[1]:
                                current_colour = get_colour(correction.args[1], world_.problem.initialstate)
                                prev_colour = get_colour(previous_corr.args[1], world_.problem.initialstate)
                                if current_colour != correction.c2 and prev_colour != correction.c2:
                                    possible_sentences.append(('no, that is not {} again'.format(correction.c2), correction))
                            break

                if self.previous_correction.sentence == correction.sentence:
                     possible_sentences.append(('no, that is wrong for the same reason', self.previous_correction))

            possible_sentences.append((correction.sentence, correction))
            # if self.previous_correction is not None and self.previous_correction.rule == correction.rule and 'now you cannot put' not in self.previous_correction.sentence and 'now you cannot put' not in correction.sentence:
            #     if 'blocks on ' in self.previous_correction.sentence:
            #         possible_sentences.append(("no, that is wrong for the same reason", self.previous_correction))
            #     o1, o2 = get_top_two(world_)
            #     preds_o1 = pddl_functions.get_predicates(o1, world_.problem.initialstate)
            #     preds_o2 = pddl_functions.get_predicates(o2, world_.problem.initialstate)
            #     c1 = [pred.name for pred in preds_o1 if pred.name in colour_dict.keys()]
            #     c2 = [pred.name for pred in preds_o2 if pred.name in colour_dict.keys()]
            #
            #     o3, o4  = self.previous_correction.args[:2]
            #     preds_o3 = pddl_functions.get_predicates(o3, world_.problem.initialstate)
            #     preds_o4 = pddl_functions.get_predicates(o4, world_.problem.initialstate)
            #     c3 = [pred.name for pred in preds_o3 if pred.name in colour_dict.keys()]
            #     c4 = [pred.name for pred in preds_o4 if pred.name in colour_dict.keys()]
            #     if 't' not in o2 and 't' not in o3:
            #         c1 = c1[0]
            #         c2 = c2[0]
            #         c3 = c3[0]
            #         c4 = c4[0]
            #
            #         if c1 != correction.c1 and c3 != self.previous_correction.c1:
            #             possible_sentences.append(('no, that is not {} either'.format(correction.c1), correction))
            #         elif c2 != correction.c2 and c4 != self.previous_correction.c2:
            #             possible_sentences.append(('no, that is not {} either'.format(correction.c2), correction))
            #
            # if correction.rule in self.rules_corrected and 'now you cannot put' not in correction.sentence:
            #     possible_sentences.append(('no', correction))
            #
            #     o1, o2 = get_top_two(world_)
            #     preds_o1 = pddl_functions.get_predicates(o1, world_.problem.initialstate)
            #     preds_o2 = pddl_functions.get_predicates(o2, world_.problem.initialstate)
            #     c1 = [pred.name for pred in preds_o1 if pred.name in colour_dict.keys()]
            #     c2 = [pred.name for pred in preds_o2 if pred.name in colour_dict.keys()]
            #     if 't' not in o2:
            #         c1 = c1[0]
            #         c2 = c2[0]
            #         possible_sentences.append(('no, you put a {} block on a {} block'.format(c1, c2), correction))
            #
            #         if c1 != correction.c1:
            #             possible_sentences.append(('no, that is not {}'.format(correction.c1), correction))
            #         elif c2 != correction.c2:
            #             possible_sentences.append(('no, that is not {}'.format(correction.c2), correction))

        return possible_sentences


    def select_correction(self, possible_sentences):
        # possible_sentences = []
        # if self.previous_correction is not None:
        #     for correction in possible_corrections[:]:
        #         possible_sentences.append((correction.sentence, correction))
        #         if self.previous_correction.rule == correction.rule and 'now you cannot put' not in self.previous_correction.sentence and 'now you cannot put' not in correction.sentence:
        #             possible_sentences.append(("no, that is wrong for the same reason", self.previous_correction))
        #         if self.correction.rule in self.rules_corrected and 'now you cannot put' not in correction.sentence:
        #             possible_sentences.append('no')
        #
        #             #if self.previous_correction.args[0] == correction.args[0] and len(correction.args[]):

        sentence, correction = random.choice(possible_sentences)

        new_correction = Correction(correction.rule, correction.c1, correction.c2, correction.args, sentence)
        self.previous_correction = new_correction
        self.dialogue_history.append(new_correction)
        try:
            self.rules_corrected.add(correction.rule)
        except AttributeError as e:
            print(correction)
            raise e
        return sentence
