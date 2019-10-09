from correctingagent.world.rules import Rule
from ..pddl import pddl_functions
from collections import namedtuple
import random
from ..util.colour_dict import colour_dict


def tower_correction(obj1, obj2):
    return f"no, put {obj1} blocks on {obj2} blocks"


def table_correction(obj1, obj2, obj3):
    return f"No, now you cannot put {obj3} in the tower because you must put {obj1} blocks on {obj2} blocks"


def not_correction(obj1, obj2):
    return f"No, don't put {obj1} blocks on {obj2} blocks"


def table_not_correction(obj1, obj2, obj3):
    return f"No, now you cannot put {obj3} in the tower because you cannot put {obj1} blocks on {obj2} blocks"

def get_rules(goal):
    """Returns the individual rules which make up the goal"""
    for f in goal.subformulas:
        if "in-tower" not in f.asPDDL():
            yield f


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

    def reset(self):
        pass

    def correction(self, w):
        failure = w.test_failure()

        if not failure:
            return ""
        #Reasons for failure:
        # a->b, a -b
        # b-> a b -a
        rules = list(get_rules(w.problem.goal))
        for r in rules:
            rule = Rule.from_formula(r)
            rule_violated = rule.check_tower_violation(w.state)

            if rule_violated:

                c1, c2 = rule.c1, rule.c2

                return tower_correction(c1, c2)
        for r in rules:
            rule = Rule.from_formula(r)
            o3 = rule.check_table_violation(w.state, [Rule.from_formula(_r) for _r in rules])

            if o3:
                c1, c2 = rule.c1, rule.c2

                return table_correction(c1, c2, o3)

    def answer_question(self, question, world_):
        if "Is the top object" in question:
            colour = question.replace("Is the top object", '').replace("?", '').strip()
            o1, o2 = world_.state.get_top_two()
            o1_is_colour = world_.state.predicate_holds(colour, [o1])
            if o1_is_colour:
                return "yes"
            else:
                return "no"


Correction = namedtuple('Correction', ['rule', 'c1', 'c2', 'args', 'sentence'])


class BaseRule(object):
    pass


class ExtendedTeacherAgent(TeacherAgent):

    def __init__(self):
        self.previous_correction = None
        self.rules_corrected = set()
        self.dialogue_history = []

    def reset(self):
        self.previous_correction = None
        self.rules_corrected = set()
        self.dialogue_history = []

    def correction(self, wrld, return_possible_corrections=False):
        failure = wrld.test_failure()

        if not failure:
            return ""

        possible_corrections = []
        rules = list(get_rules(wrld.problem.goal))
        for r in rules:
            rule = Rule.from_formula(r)
            if rule.check_tower_violation(wrld.state):
                # c1, c2, impl = get_relevant_colours(r)
                o1, o2 = wrld.state.get_top_two()
                # if impl == 'not':
                #     corr = Correction(r, c1, c2, [o1, o2], not_correction(c1, c2))
                #     possible_corrections.append(corr)
                # else:
                corr = Correction(r, rule.c1, rule.c2, [o1, o2], tower_correction(rule.c1, rule.c2))
                possible_corrections.append(corr)

        for r in rules:
            rule = Rule.from_formula(r)
            o3 = rule.check_table_violation(wrld.state, [Rule.from_formula(_r) for _r in rules])
            if o3:
                #c1, c2, impl = get_relevant_colours(r)
                o1, o2 = wrld.state.get_top_two()
                # if impl == 'not':
                #     corr = Correction(r, c1, c2, [o1, o2], table_not_correction(c1, c2, o3))
                #     possible_corrections.append(corr)
                # else:
                corr = Correction(r, rule.c1, rule.c2, [o1, o2, o3], table_correction(rule.c1, rule.c2, o3))
                possible_corrections.append(corr)

        possible_corrections = self.generate_possible_corrections(possible_corrections, wrld)
        selection = self.select_correction(possible_corrections)
        if return_possible_corrections:
            return selection, possible_corrections
        else:
            return selection

    def generate_possible_corrections(self, violations, wrld):

        possible_sentences = []
        for correction in violations:
            if len(self.dialogue_history) > 0:

                if "now you cannot put" not in correction.sentence:
                    for previous_corr in self.dialogue_history[::-1]:
                        if 'now you cannot put' not in previous_corr.sentence and 'same reason' not in previous_corr.sentence:
                            colours = [previous_corr.c1, previous_corr.c2]
                            if correction.c1 in colours or correction.c2 in colours:

                                if correction.c1 == colours[0]:
                                    current_colour = wrld.state.get_colour_name(correction.args[0])
                                    prev_colour = wrld.state.get_colour_name(previous_corr.args[0])
                                    if current_colour != correction.c1 and prev_colour != correction.c1:
                                        possible_sentences.append(('no, that is not {} again'.format(correction.c1), correction))
                                if correction.c2 == colours[1]:
                                    current_colour = wrld.state.get_colour_name(correction.args[1])
                                    prev_colour = wrld.state.get_colour_name(previous_corr.args[1])
                                    if current_colour != correction.c2 and prev_colour != correction.c2:
                                        possible_sentences.append(('no, that is not {} again'.format(correction.c2), correction))
                                break
                        elif 'now you cannot put' in previous_corr.sentence:
                            colours = [previous_corr.c1, previous_corr.c2]
                            if correction.c1 in colours or correction.c2 in colours:
                                break

                if self.previous_correction.sentence == correction.sentence:
                     possible_sentences.append(('no, that is wrong for the same reason', self.previous_correction))

            possible_sentences.append((correction.sentence, correction))

        return possible_sentences

    def select_correction(self, possible_sentences):

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
