import random
from collections import namedtuple
from enum import Enum

import numpy as np

from correctingagent.world import PDDLWorld
from correctingagent.world.rules import Rule, RedOnBlueRule, ColourCountRule


class TeacherType(Enum):
    Human = 1
    Old = 2
    Extended = 3
    Faulty = 4

def tower_correction(obj1, obj2):
    return f"no, put {obj1} blocks on {obj2} blocks"

def joint_tower_correction(obj1, obj2):
    return f"you must put {obj1} blocks on {obj2} blocks"

def table_correction(obj1, obj2, obj3):
    return f"No, now you cannot put {obj3} in the tower because you must put {obj1} blocks on {obj2} blocks"


def not_correction(obj1, obj2):
    return f"No, don't put {obj1} blocks on {obj2} blocks"


def table_not_correction(obj1, obj2, obj3):
    return f"No, now you cannot put {obj3} in the tower because you cannot put {obj1} blocks on {obj2} blocks"

def colour_count_correction(colour, number):
    return f"no, you cannot put more than {number} {colour} blocks in a tower"

def get_rules(goal):
    """Returns the individual rules which make up the goal"""
    for f in goal.subformulas:
        if "in-tower" not in f.asPDDL():
            yield Rule.from_formula(f)


class Teacher(object):

    def correction(self, world_, args):
        raise NotImplementedError()

    def reset(self):
        pass

    def answer_question(self, question, world_, tower):
        if "Is the top object" in question:
            colour = question.replace("Is the top object", '').replace("?", '').strip()
            o1, o2 = world_.state.get_top_two(tower)
            o1_is_colour = world_.state.predicate_holds(colour, [o1])
            if o1_is_colour:
                return "yes"
            else:
                return "no"


class HumanTeacher(Teacher):

    def correction(self, world_, args):
        return input('Correction?')

    def answer_question(self, question, world_, tower):
        return input(question)


class TeacherAgent(Teacher):

    def correction(self, w, args):
        failure = w.test_failure()
        if len(args) == 3:
            tower = args[-1]
        else:
            tower = None
        #print("Failure?", failure)
        if not failure:
            return ""
        #Reasons for failure:
        # a->b, a -b
        # b-> a b -a
        rules = Rule.get_rules(w.problem.goal)
        for rule in rules:
            rule_violated = rule.check_tower_violation(w.state, tower)
            if rule_violated:
                return get_tower_correction(rule, w, tower).sentence

        for rule in rules:
            if rule.check_table_violation(w.state, rules, tower):
                return get_table_correction(rule, w, rules, tower).sentence

        return ""


Correction = namedtuple('Correction', ['rule', 'args', 'sentence'])


class BaseRule(object):
    pass


def get_tower_correction(rule: Rule, wrld: PDDLWorld, tower: str):
    if isinstance(rule, RedOnBlueRule):
        o1, o2 = wrld.state.get_top_two(tower)
        return Correction(rule, [o1, o2], tower_correction(rule.c1, rule.c2))
    elif isinstance(rule, ColourCountRule):
        correction_str = colour_count_correction(rule.colour_name, rule.number)
        for tower in wrld.state.towers:
            tower = tower.replace('t', 'tower')
            if wrld.state.get_colour_count(rule.colour_name, tower) > rule.number:
                out_tower = tower
        return Correction(rule, [out_tower], correction_str)


def get_table_correction(rule: Rule, wrld: PDDLWorld, rules: list = None, tower: str = None):
    if isinstance(rule, RedOnBlueRule):

        if rule.rule_type == 1:
            colour = rule.c1
        else:
            colour = rule.c2

        o3 = wrld.state.get_block_with_colour(colour)
        if o3 is None:
            towers = [t.replace('t', 'tower') for t in wrld.state.towers if t.replace('t', 'tower') != tower]
            o3 = [top for top, second in
                  [wrld.state.get_top_two(t) for t in towers]
                  if wrld.state.predicate_holds(colour, [top])][0]

        o1, o2 = wrld.state.get_top_two(tower)

        corr = Correction(rule, [o1, o2, o3], table_correction(rule.c1, rule.c2, o3))
        return corr
    elif isinstance(rule, ColourCountRule):
        for tower in wrld.state.towers:
            tower = tower.replace('t', 'tower')
            if wrld.state.get_colour_count(rule.colour_name, tower):

                for rule2 in rules:
                    if isinstance(rule2, RedOnBlueRule):
                        top, _ = wrld.state.get_top_two(tower)
                        if rule2.c1 == rule.colour_name and wrld.state.predicate_holds(rule2.c2, [top]):
                            correction_str1 = colour_count_correction(rule.colour_name, rule.number)
                            correction_str2 = joint_tower_correction(rule2.c1, rule2.c2)
                            correction_str = correction_str1 + ' and ' + correction_str2
                            corr = Correction(rule, [tower], correction_str)
                            return corr


class ExtendedTeacherAgent(TeacherAgent):

    def __init__(self):
        self.previous_correction = None
        self.rules_corrected = set()
        self.dialogue_history = []

    def reset(self):
        self.previous_correction = None
        self.rules_corrected = set()
        self.dialogue_history = []

    def correction(self, wrld, args, return_possible_corrections=False):
        failure = wrld.test_failure()

        if len(args) == 3:
            tower = args[-1]
        else:
            tower = None

        if not failure:
            return ""

        possible_corrections = []
        rules = Rule.get_rules(wrld.problem.goal)
        for rule in rules:

            if rule.check_tower_violation(wrld.state, tower):

                corr = get_tower_correction(rule, wrld, tower)

                possible_corrections.append(corr)

        for rule in rules:
            if rule.check_table_violation(wrld.state, rules, tower):

                corr = get_table_correction(rule, wrld, rules, tower)

                possible_corrections.append(corr)

        possible_corrections = self.generate_possible_corrections(possible_corrections, wrld)
        if len(possible_corrections) == 0:
            return ""
        selection = self.select_correction(possible_corrections)
        if return_possible_corrections:
            return selection, possible_corrections
        else:
            return selection

    def generate_possible_corrections(self, violations: list, wrld: PDDLWorld):

        possible_sentences = []
        for correction in violations:
            if len(self.dialogue_history) > 0 and isinstance(correction.rule, RedOnBlueRule):

                for previous_corr in self.dialogue_history[::-1]:

                    if 'now you cannot put' not in previous_corr.sentence and 'same reason' not in previous_corr.sentence and isinstance(previous_corr.rule, RedOnBlueRule):

                        colours = [previous_corr.rule.c1, previous_corr.rule.c2]

                        if correction.rule.c1 in colours or correction.rule.c2 in colours:
                            if correction.rule == previous_corr.rule:

                                #if correction.rule.c1 == colours[0]:  # should be doing nothing
                                current_colour = wrld.state.get_colour_name(correction.args[0])
                                prev_colour = wrld.state.get_colour_name(previous_corr.args[0])
                                if current_colour != correction.rule.c1 and prev_colour != correction.rule.c1:
                                    possible_sentences.append((f'no, that is not {correction.rule.c1} again', correction))
                                #if correction.rule.c2 == colours[1]:  # should be doing nothing
                                current_colour = wrld.state.get_colour_name(correction.args[1])
                                prev_colour = wrld.state.get_colour_name(previous_corr.args[1])
                                if current_colour != correction.rule.c2 and prev_colour != correction.rule.c2:
                                    possible_sentences.append((f'no, that is not {correction.rule.c2} again', correction))
                            break
                    elif isinstance(previous_corr.rule, ColourCountRule):
                        break
                    elif 'now you cannot put' in previous_corr.sentence:
                        colours = [previous_corr.rule.c1, previous_corr.rule.c2]
                        if correction.rule.c1 in colours or correction.rule.c2 in colours:
                            break

                if isinstance(correction.rule, RedOnBlueRule) and self.previous_correction.sentence == correction.sentence:
                    if self.previous_correction.rule == correction.rule:
                        possible_sentences.append(('no, that is wrong for the same reason', self.previous_correction))
                elif self.previous_correction == 'no, that is wrong for the same reason':
                    for previous_corr in self.dialogue_history[::-1]:
                        if previous_corr.sentence != 'no, that is wrong for the same reason':
                            if isinstance(correction.rule,
                                          RedOnBlueRule) and previous_corr.sentence == correction.sentence:
                                if previous_corr.rule == correction.rule:
                                    possible_sentences.append(
                                        ('no, that is wrong for the same reason', self.previous_correction))
                            break
            possible_sentences.append((correction.sentence, correction))

        return possible_sentences

    def select_correction(self, possible_sentences):
        print(possible_sentences)

        reduced = list(filter(lambda x: 'same reason' in x[0] or 'again' in x[0], possible_sentences))
        if len(reduced) > 0:
            possible_sentences = reduced

        sentence, correction = random.choice(possible_sentences)

        new_correction = Correction(correction.rule, correction.args, sentence)
        self.previous_correction = new_correction
        self.dialogue_history.append(new_correction)
        try:
            self.rules_corrected.add(correction.rule)
        except AttributeError as e:
            print(correction)
            raise e
        return sentence


class FaultyTeacherAgent(Teacher):

    def __init__(self, recall_failure_prob=0.5, recover_prob=0.1):
        self.recall_failure_prob = recall_failure_prob
        self.recover_prob = recover_prob
        self.skipped_corrections = []

    def reset(self):
        self.skipped_corrections = []

    def correction(self, w, args):

        failure = w.test_failure()
        if len(args) == 3:
            tower = args[-1]
        else:
            tower = None

        if not failure:
            return ""
        #Reasons for failure:
        # a->b, a -b
        # b-> a b -a
        rules = Rule.get_rules(w.problem.goal)
        for rule in rules:
            rule_violated = rule.check_tower_violation(w.state, tower)
            if rule_violated:
                return get_tower_correction(rule, w, tower).sentence

        r = np.random.rand()

        for rule in rules:
            if rule.check_table_violation(w.state, rules, tower):
                if r > self.recall_failure_prob:
                    return get_table_correction(rule, w, rules, tower).sentence
                else:
                    self.skipped_corrections.append(get_table_correction(rule, w, rules, tower))

        if len(w.state.get_objects_on_table()) == 0:
            if not w.test_success():
                rules = Rule.get_rules(w.problem.goal)
                for rule in rules:
                    if rule.rule_type == 2:
                        for tower in w.state.towers:
                            tower = tower.replace('t', 'tower')
                            top, _ = w.state.get_top_two(tower)
                            if w.state.predicate_holds(rule.c2, [top]):
                                return get_tower_correction(rule, w, tower).sentence

        if len(self.skipped_corrections) > 0:
            r = np.random.rand()
            if r < self.recover_prob:
                correction = np.random.choice(self.skipped_corrections)
                self.skipped_corrections.remove(correction)
                return correction

        return ""

