import copy
import heapq
import re
from collections import defaultdict

import nltk
import numpy as np
import pythonpddl
from nltk import Valuation, Model
from pythonpddl.pddl import Predicate, TypedArgList, Formula

from correctingagent.util.CPD_generation import binary_flip
from correctingagent.pddl import pddl_functions
from correctingagent.pddl.pddl_functions import make_variable_list, PDDLState
from correctingagent.world import goals


def split_rule(rule):
    bits = rule.split('.(')
    red = bits[1].split('->')[0].strip()
    bits2 = bits[1].split(' (')
    blue, on = bits2[1].split('&')
    blue = blue.strip()
    on = on.replace('))', '').strip()
    return [red, blue, on]


def get_predicate_args(predicate):
    args = predicate.split('(')[1].replace(')', '')
    return [arg.strip() for arg in args.split(',')]


def get_predicate(predicate):
    pred = predicate.split('(')[0]
    args = predicate.split('(')[1].replace(')', '')
    args = [arg.strip() for arg in args.split(',')]
    return pred, args


class Rule(object):

    def __init__(self):
        pass

    @staticmethod
    def from_formula(formula):
        rule_type = Rule.get_rule_type(formula)
        if rule_type == 1:
            c1, c2 = Rule.get_rule_colours(formula)
            return RedOnBlueRule(c1, c2, rule_type)
        elif rule_type == 2:
            c2, c1 = Rule.get_rule_colours(formula)
            return RedOnBlueRule(c1, c2, rule_type)
        elif rule_type == 3:
            c1, c2 = Rule.get_rule_colours_existential(formula)
            raise NotImplementedError("Have not implemented not red on blue rules")
        elif rule_type == 4:
            return ColourCountRule.from_formula(formula)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, Rule):
            return (str(self) == str(other))
        else:
            return False

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __hash__(self):
        return hash(str(self))

    def asPDDL(self):
        return self.to_formula().asPDDL()

    def to_formula(self):
        return self.rule_formula

    @staticmethod
    def get_rule_colours(formula):
        """This function only works if the formula has 2 colours
        (one before and one after the arrow)
        To allow for more than one object on either side of the rule more logic must be added"""
        c1, right = formula.subformulas[0].subformulas
        c1 = c1.get_predicates(False)[0]
        c1 = c1.get_predicates(True)[0]
        c1 = c1.name
        c2 = right.subformulas[0].subformulas[0].get_predicates(True)[0].name
        return c1, c2

    @staticmethod
    def get_rule_colours_existential(formula):
        """This function only works if the formula has 2 colours
        (one before and one after the arrow)
        To allow for more than one object on either side of the rule more logic must be added"""
        and_formula = formula.subformulas[0].subformulas[0]
        c1 = and_formula.subformulas[0].get_predicates(True)[0].name
        c2 = and_formula.subformulas[1].get_predicates(True)[0].name
        return c1, c2

    @staticmethod
    def get_rule_type(formula):
        if formula.op == 'not':
            return 3
        elif formula.variables.asPDDL() == '?x':
            return 1
        elif formula.variables.asPDDL() == '?y':
            return 2
        elif formula.variables.asPDDL() == '?t':
            return 4
        else:
            raise ValueError('Unknown Rule Type')

    @staticmethod
    def get_rules(goal):
        return [Rule.from_formula(subformula) for subformula in goal.subformulas[1:]]

    @staticmethod
    def generate_red_on_blue_options(c1, c2):
        return [RedOnBlueRule(c1, c2, rule_type=1), RedOnBlueRule(c1, c2, rule_type=2)]

    @staticmethod
    def rule_from_violation(violation):
        rule_string = re.sub(r"V_[0-9]+\(", '', violation)[:-1]

        if "&&" in rule_string:
            colour_count_string, red_on_blue_string = rule_string.split("&&")
            return [Rule.rule_from_violation(f"V_1({colour_count_string.strip()})"),
                    Rule.rule_from_violation(f"V_1({red_on_blue_string.strip()})")]

        elif rule_string[:3] != 'all':
            raise NotImplemented('Not implemented non put red on blue rule')

        elif "count" in rule_string:
            _, right = rule_string.split('->')
            colour_count, count = rule_string.split('>=')
            count = int(count.strip())
            colour_name = colour_count.split('-')[0].strip()
            # f"all t. tower(t) -> {colour_name}-count >= {count}"
            return ColourCountRule(colour_name, count)

        else:
            red, blue, on = split_rule(rule_string)
            red_colour, o1 = get_predicate(red)
            blue_colour, o2 = get_predicate(blue)
            x, y = get_predicate_args(on)

            if o1[0] == x:
                # return red_colour, blue_colour, rule_type
                return RedOnBlueRule(red_colour, blue_colour, rule_type=1)
            elif o2[0] == x:
                return RedOnBlueRule(blue_colour, red_colour, rule_type=2)
            else:
                raise ValueError('something went wrong')


class ColourCountRule(Rule):

    def __init__(self, colour_name: str, count: int):
        self.colour_name = colour_name
        self.number = int(count)
        self.name = f"all t. tower(t) -> {colour_name}-count >= {count}"
        self.constraint = ColourCountRuleConstraint(colour_name, count)

    @staticmethod
    def from_formula(formula: Formula):
        assert(formula.op == 'forall')
        or_formula = formula.subformulas[0]
        left, right = or_formula.subformulas
        colour_counter_name, count = right.subformulas
        colour_name = colour_counter_name.name.split('-')[0]
        count = count.val
        return ColourCountRule(colour_name, count)

    def to_formula(self):
        args = make_variable_list(['?t'])
        colour_count_head = pythonpddl.pddl.FHead(f'{self.colour_name}-count', args)
        count_value = pythonpddl.pddl.ConstantNumber(self.number)

        left = Formula([colour_count_head, count_value], op='<=')

        tower_pred = Predicate("tower", args)
        tower = Formula([tower_pred])
        right = Formula([tower], op='not')
        or_formula = Formula([right, left], op='or')
        return Formula([or_formula], op='forall', variables=args)

    def check_tower_violation(self, state: PDDLState):
        for tower in state.towers:
            tower = tower.replace('t', 'tower')
            cc = state.get_colour_count(self.colour_name, tower)
            if cc > self.number:
                return True
        return False

    def check_table_violation(self, state: PDDLState, rules: list):
        for tower in state.towers:
            tower = tower.replace('t', 'tower')
            if state.get_colour_count(self.colour_name, tower) == self.number:
                for rule in rules:
                    try:
                        top, _ = state.get_top_two(tower)
                        if rule.c1 == self.colour_name and state.predicate_holds(rule.c2, [top]):
                            if rule.rule_type == 2:
                                return True
                            else:
                                first_colour_count = state.count_coloured_blocks(rule.c1)
                                second_colour_count = state.count_coloured_blocks(rule.c2)
                                if first_colour_count == second_colour_count + 1:
                                    return True
                    except AttributeError:
                        continue
        return False

    def generateCPD(self, num_blocks_in_tower=2, table_correction=False, num_blocks_on_table=8, second_rule=None, **kwargs):
        #return self.generate_tower_cpd(num_blocks_in_tower=num_blocks_in_tower, table_correction=table_correction)
        if table_correction is False:
            return self.generate_tower_cpd(num_blocks_in_tower, table_correction=table_correction)
        else:
            # return self.generate_table_cpd(num_blocks_in_tower, num_blocks_on_table, second_rule)
            return self.generate_table_cpd(num_blocks_in_tower)

    def generate_table_cpd(self, num_blocks_in_tower):
        flippings = binary_flip(num_blocks_in_tower + 1 + 1)
        CPD = np.zeros((2, len(flippings)), dtype=np.int32)
        for i, flip_selection in enumerate(flippings):
            blocks_in_tower = flip_selection[:num_blocks_in_tower-1]
            top_block = flip_selection[num_blocks_in_tower-1]
            other_rule = flip_selection[-2]
            cc = flip_selection[-1]
            this_rule_satisfied = sum(blocks_in_tower) == self.number and top_block == 1
            violation = this_rule_satisfied and other_rule == 1 and cc == 1
            CPD[1][i] = int(violation)
            CPD[0][i] = 1 - int(violation)
        return CPD


    # def generate_table_cpd(self, num_blocks_in_tower: int, num_blocks_on_table: int, second_rule):
    #     """ output order is blocks in tower, blocks on table (red then blue), second rule, this rule
    #
    #     :param num_blocks_in_tower: len([blue(o1), blue(o2),...,red(oN)])
    #     :param num_blocks_on_table: len([red(on+1), ..., red(oN+M])
    #     :param second_rule: RedOnBlueRule
    #     :return:
    #     """
    #     flippings = binary_flip(num_blocks_in_tower + (2 * num_blocks_on_table) + 1 + 1)
    #     CPD = np.zeros((2, len(flippings)), dtype=np.int32)
    #     for i, flip_selection in enumerate(flippings):
    #
    #         blocks_in_tower = flip_selection[:num_blocks_in_tower]
    #         red_blocks_on_table = flip_selection[num_blocks_in_tower:num_blocks_in_tower+num_blocks_on_table]
    #         blue_blocks_on_table = flip_selection[num_blocks_in_tower+num_blocks_on_table:num_blocks_in_tower+2*num_blocks_on_table]
    #         other_rule = bool(flip_selection[-2])
    #         this_rule = bool(flip_selection[-1])
    #         this_rule_satisfied = sum(blocks_in_tower) == self.number
    #         red_equals_blue = sum(blue_blocks_on_table) == sum(red_blocks_on_table)
    #         more_blue_than_red = sum(blue_blocks_on_table) >= sum(red_blocks_on_table)
    #
    #         if second_rule.rule_type == 1:
    #             violated = int(this_rule_satisfied and red_equals_blue and other_rule and this_rule)
    #         elif second_rule.rule_type == 2:
    #             violated = int(this_rule_satisfied and more_blue_than_red and other_rule and this_rule)
    #         CPD[1][i] = violated
    #         CPD[0][i] = violated
    #     return CPD


    def generate_tower_cpd(self, num_blocks_in_tower, table_correction=False):
        offset = 1 if not table_correction else 0
        flippings = binary_flip(num_blocks_in_tower)
        CPD = np.zeros((2, len(flippings)), dtype=np.int32)
        for i, l in enumerate(flippings):
            rule_violated = int(sum(l[:-1]) == (self.number + offset)) * l[-1]
            CPD[1][i] = rule_violated
            CPD[0][i] = 1 - rule_violated
        return CPD


class RedOnBlueRule(Rule):

    def __init__(self, c1: str, c2: str, rule_type: int):
        self.c1 = c1
        self.c2 = c2
        self.rule_type = rule_type
        if self.rule_type == 1:
            self.name = f'all x.({self.c1}(x) -> exists y. ({self.c2}(y) & on(x,y)))'
        elif self.rule_type == 2:
            self.name = f'all y.({self.c2}(y) -> exists x. ({self.c1}(x) & on(x,y)))'
        self.constraint = RuleConstraint(rule_type, c1, c2)

    def to_formula(self):
        if self.rule_type == 1:
            variables = ("?x", "?y")
            obj1, obj2 = [self.c1], [self.c2]
        elif self.rule_type == 2:
            variables = ("?y", "?x")
            obj1, obj2 = [self.c2], [self.c1]
        else:
            raise ValueError(f'Invalid Rule Type. Expected 1 or 2 got {self.rule_type}')

        variables = pddl_functions.make_variable_list(variables)
        obj1_preds = [Predicate(p, TypedArgList([variables.args[0]])) for p in obj1]
        obj2_preds = [Predicate(p, TypedArgList([variables.args[1]])) for p in obj2]
        on = Predicate('on', pddl_functions.make_variable_list(['?x', '?y']))

        if len(obj1_preds) > 1:
            obj1_formula = Formula(obj1_preds, op='and')
        else:
            obj1_formula = Formula(obj1_preds)

        if len(obj2_preds) > 1:
            obj2_formula = Formula(obj2_preds, op='and')
        else:
            obj2_formula = Formula(obj2_preds)

        second_part = Formula([obj2_formula, on], op='and')
        existential = Formula([second_part], op='exists',
                              variables=pddl_functions.make_variable_list([variables.args[1].arg_name]))
        neg_o1 = Formula([obj1_formula], op='not')
        subformula = Formula([neg_o1, existential], op='or')

        return Formula([subformula], op='forall',
                       variables=pddl_functions.make_variable_list([variables.args[0].arg_name]))

    def generate_tower_cpd(self):
        cpd_line_corr0 = []
        cpd_line_corr1 = []
        for i in range(2):
            for j in range(2):
                for r in range(2):
                    result = r * (1 - int(self.evaluate_rule(i, j)))
                    cpd_line_corr1.append(result)
                    cpd_line_corr0.append(1 - result)

        return [cpd_line_corr0, cpd_line_corr1]

    def evaluate_rule(self, value1: int, value2: int):
        c1_set = set()
        c2_set = set()
        if value1 == 1:
            c1_set.add('o1')
        if value2 == 1:
            c2_set.add('o2')
        v = [(self.c1, c1_set), (self.c2, c2_set),
             ('on', set([('o1', 'o2')]))]
        val = Valuation(v)
        dom = val.domain
        m = Model(dom, val)
        g = nltk.sem.Assignment(dom)
        return m.evaluate(str(self), g)

    def generateCPD(self, table_correction=False, **kwargs):
        if table_correction:
            return self.generate_table_cpd()
        else:
            return self.generate_tower_cpd()

    def generate_table_cpd(self):
        cpd_line_corr0 = []
        cpd_line_corr1 = []

        for redo1 in range(2):
            for blueo2 in range(2):
                for redo3 in range(2):
                    for blueo3 in range(2):
                        for r1 in range(2):
                            if self.rule_type == 1:
                                result = r1 * int(not (redo1) and blueo2 and redo3)
                            elif self.rule_type == 2:
                                result = r1 * int(redo1 and not (blueo2) and blueo3)
                            cpd_line_corr1.append(result)
                            cpd_line_corr0.append(1 - result)
        return [cpd_line_corr0, cpd_line_corr1]

    def check_tower_violation(self, state):

        o1, o2 = state.get_top_two()

        c1_o1 = state.predicate_holds(self.c1, [o1])
        c2_o2 = state.predicate_holds(self.c2, [o2])

        if c1_o1 and not c2_o2 and self.rule_type == 1:
            return True
        elif not c1_o1 and c2_o2 and self.rule_type == 2:
            return True
        else:
            return False

    def get_all_relevant_colours(self, rules):
        """

        :param rules:
        :return:
        """

        additional_red_constraints = []
        additional_blue_constraints = []
        for rule in rules:

            if self.c1 == rule.c1 and self.rule_type == 2 and rule.rule_type == 2:
                additional_red_constraints.append(rule.c2)
            if self.c2 == rule.c2 and self.rule_type == 1 and rule.rule_type == 1:
                additional_blue_constraints.append(rule.c1)
        return additional_red_constraints, additional_blue_constraints

    def check_table_violation(self, state, additional_rules=[]):
        top_object, second_object = state.get_top_two()
        additional_relevant_rules = [rule for rule in additional_rules if isinstance(rule, RedOnBlueRule)]
        additional_bottom_constrained_objects, additional_top_constrained_objects = self.get_all_relevant_colours(additional_relevant_rules)

        top_object_is_red = state.predicate_holds(self.c1, [top_object])
        second_object_is_blue = state.predicate_holds(self.c2, [second_object])

        number_red_blocks = state.count_coloured_blocks(self.c1)
        number_blue_blocks = state.count_coloured_blocks(self.c2)

        number_additional_bottom_objects = sum([state.count_coloured_blocks(colour) for colour in additional_bottom_constrained_objects])
        number_additional_top_objects = sum([state.count_coloured_blocks(colour) for colour in additional_top_constrained_objects])

        # put red(x) -> blue(y) on(x,y) is table violated only in this case:
        if not top_object_is_red and second_object_is_blue and self.rule_type == 1 and number_red_blocks > 0:
            if (number_red_blocks + number_additional_top_objects) > number_blue_blocks:
                return True
        if top_object_is_red and not second_object_is_blue and self.rule_type == 2 and number_blue_blocks > 0:
            if (number_blue_blocks + number_additional_bottom_objects) > number_red_blocks:
                return True
        return False


class RuleConstraint(object):

    def __init__(self, rule_type, c1, c2):
        if rule_type == 1:
            self.name = '#{c1} <= #{c2}'.format(**{'c1':c1, 'c2':c2})
        elif rule_type == 2:
            self.name = '#{c2} <= #{c1}'.format(**{'c1':c1, 'c2':c2})
        elif rule_type == 3:
            self.name = 'no constraint'
        self.rule_type = rule_type
        self.c1 = c1
        self.c2 = c2

    def evaluate(self, colour_counts):
        if self.rule_type == 1:
            return colour_counts[self.c1] <= colour_counts[self.c2]
        elif self.rule_type == 2:
            return colour_counts[self.c2] <= colour_counts[self.c1]
        elif self.rule_type == 3:
            return True


class ColourCountRuleConstraint(object):

    def __init__(self, colour, number, num_towers=2):
        self.name = f"{colour}-count <= {number}"
        self.colour = colour
        self.number = number
        self.num_towers = num_towers

    def evaluate(self, colour_counts):
        return colour_counts[self.colour] <= (self.number * self.num_towers)

class ConstraintCollection(object):

    def __init__(self, constraints):
        self.constraints = constraints

    def from_rules(rules):
        return ConstraintCollection([r.constraint for r in rules])

    def evaluate(self, state):
        pass_all = True
        increase = set()
        decrease = set()
        for constraint in self.constraints:
            result = constraint.evaluate(state.colour_counts)
            if result is False:
                pass_all = False
                if isinstance(constraint, RuleConstraint):
                    if constraint.rule_type == 1:
                        increase.add(constraint.c2)
                        decrease.add(constraint.c1)
                    elif constraint.rule_type == 2:
                        increase.add(constraint.c1)
                        decrease.add(constraint.c2)
                elif isinstance(constraint, ColourCountRuleConstraint):
                    decrease.add(constraint.colour)
        return pass_all, increase, decrease


class GoalState(object):

    def __init__(self, rules, scores=None):
        self.decision = [0] * len(rules)
        self.rules = rules

    def flip(self, i):
        self.decision[i] = 1-self.decision[i]

    def to_goal(self, goal):
        out_rules = [self.rules[i][d] for i, d in enumerate(self.decision)]
        out_goal = None
        for rule in out_rules:
            out_goal = goals.update_goal(out_goal, rule)
        g = goals.update_goal(goal, out_goal)

        return g


class State(object):

    def __init__(self, obs, colour_choices={}, threshold=0.5):
        self.initialstate = obs.state
        clear_objs = obs.state.get_clear_objs()

        self.objects = list(filter(lambda x: x in clear_objs, colour_choices.keys()))
        self.colour_choices = {o:c for o,c in colour_choices.items() if o in self.objects}
        self.colours = None
        if len(self.objects) > 0:
            self.colours = list(self.colour_choices[self.objects[0]].keys())
        self.state, self.score, self.colour_counts = self.colour_choices_to_state(self.colour_choices, threshold=threshold)
        self.make_heap()

    def make_heap(self):
        try:
            self.colour_priority_positive = {c:[] for c in self.colour_choices[self.objects[0]].keys()}
            self.colour_priority_negative = {c:[] for c in self.colour_choices[self.objects[0]].keys()}
            for o in self.colour_choices.keys():
                for c in self.colour_choices[o].keys():
                    if (o, c) in self.state:
                        # p2 = 1-p1
                        # delta_p = - -log p1 + -log p2
                        p1 = self.colour_choices[o][c]
                        p2 = 1-p1
                        delta_p = np.log(p1) - np.log(p2)
                        heapq.heappush(self.colour_priority_positive[c], (delta_p, o))
                    else:
                        # p2 = 1-p1
                        # delta_p =  -log p1 - -log p2
                        p1 = self.colour_choices[o][c]
                        p2 = 1-p1
                        delta_p = -np.log(p1) + np.log(p2)
                        heapq.heappush(self.colour_priority_negative[c], (delta_p, o))
        except IndexError:
            self.colour_priority_positive = {}
            self.colour_priority_negative = {}

    def colour_choices_to_state(self, colour_choices, threshold=0.5):
        cost = 0
        obj_list = []
        colour_counts = defaultdict(int)
        for o, val in colour_choices.items():
            for c, pred in val.items():
                if pred > threshold:
                    obj_list.append((o, c))
                    cost -= np.log(pred)
                    colour_counts[c] += 1
                else:
                    cost -= np.log(1-pred)
        return obj_list, cost, colour_counts

    def _pop(self, colour, increase_count):
        if increase_count:
            return heapq.heappop(self.colour_priority_negative[colour])
        else:
            return heapq.heappop(self.colour_priority_positive[colour])

    def flip_colour(self, colour, increase_count=True):
        """flip positive means flipping a positive to a negative
        I'm not sure if this is the most intutive..."""
        try:
            delta_p, obj = self._pop(colour, increase_count)
        except IndexError:
            return
        self.score += delta_p
        if increase_count:
            self.state.append((obj, colour))
            self.colour_counts[colour] += 1
        else:
            try:
                self.state.remove((obj, colour))
                self.colour_counts[colour] -= 1
            except ValueError as e:
                print(self.state)
                print(obj, colour)
                raise e

    def asPDDL(self):
        pddl_state = copy.copy(self.initialstate.to_formula())
        for o, c in self.state:
            colour_formula = pddl_functions.create_formula(c, [o])
            pddl_state.append(colour_formula)
        return pddl_state

    def __eq__(self, other):
        if isinstance(other, State):
            return set(self.state) == set(other.state)
        else:
            return False

    def __lt__(self, other):
        if isinstance(other, State):
            return len(self.state) < len(other.state)
        else:
            return False

    def __gt__(self, other):
        if isinstance(other, State):
            return len(self.state) > len(other.state)
        else:
            return False
