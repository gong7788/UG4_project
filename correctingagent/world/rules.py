import copy
import heapq
from collections import defaultdict

import nltk
import numpy as np
from nltk import Valuation, Model
from pythonpddl.pddl import Predicate, TypedArgList, Formula

from correctingagent.pddl import pddl_functions
from correctingagent.world import goals


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

    def to_pddl(self):
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
        else:
            raise ValueError('Unknown Rule Type')

    @staticmethod
    def get_rules(goal):
        return [Rule(subformula) for subformula in goal.subformulas[1:]]

    @staticmethod
    def generate_red_on_blue_options(c1, c2):
        return [RedOnBlueRule(c1, c2, rule_type=1), RedOnBlueRule(c1, c2, rule_type=2)]


class RedOnBlueRule(Rule):

    def __init__(self, c1, c2, rule_type):
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

    def generate_CPD(self):
        cpd_line_corr0 = []
        cpd_line_corr1 = []
        for i in range(2):
            for j in range(2):
                for r in range(2):
                    result = r * (1 - int(self.evaluate_rule(i, j)))
                    cpd_line_corr1.append(result)
                    cpd_line_corr0.append(1 - result)

        return [cpd_line_corr0, cpd_line_corr1]

    def evaluate_rule(self, value1, value2):
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

    def generate_table_cpd(self):
        cpd_line_corr0 = []
        cpd_line_corr1 = []

        for redo1 in range(2):
            for blueo2 in range(2):
                for redo3 in range(2):
                    for blueo3 in range(2):
                        for r1 in range(2):  # rule 1 or 0
                            if self.rule_type == 1:
                                result = r1 * int(not (redo1) and blueo2 and redo3)
                            elif self.rule_type == 2:
                                result = r1 * int(redo1 and not (blueo2) and blueo3)
                            cpd_line_corr1.append(result)
                            cpd_line_corr0.append(1 - result)
        return [cpd_line_corr0, cpd_line_corr1]


#
# class Rule(object):
#
#     def __init__(self, rule_formula):
#         self.rule_type = Rule.get_rule_type(rule_formula)
#         if self.rule_type == 1:
#             self.c1, self.c2 = Rule.get_rule_colours(rule_formula)
#         elif self.rule_type == 2:
#             self.c2, self.c1 = Rule.get_rule_colours(rule_formula)
#         elif self.rule_type == 3:
#             self.c1, self.c2 = Rule.get_rule_colours_existential(rule_formula)
#         self.constraint = RuleConstraint(self.rule_type, self.c1, self.c2)
#         self.rule_formula = rule_formula
#         if self.rule_type in [1, 2]:
#             self.name = 'r_{}^({},{})'.format(self.rule_type, self.c1, self.c2)
#         else:
#             self.name = 'r_not^({},{})'.format(self.c1, self.c2)
#
#     def __repr__(self):
#         return self.__str__()
#
#     def __str__(self):
#         if self.rule_type == 1:
#             return f'all x.({self.c1}(x) -> exists y. ({self.c2}(y) & on(x,y)))'
#         elif self.rule_type == 2:
#             return f'all y.({self.c2}(y) -> exists x. ({self.c1}(x) & on(x,y)))'
#         elif self.rule_type == 3:
#             return f"- exists x. exists y. ({self.c1}(x) and {self.c2}(y) and on(x,y))"
#
#     def __eq__(self, other):
#         if isinstance(other, Rule):
#             return (str(self) == str(other))
#         else:
#             return False
#
#     def __ne__(self, other):
#         return (not self.__eq__(other))
#
#     def __hash__(self):
#         return hash(str(self))
#
#     def to_pddl(self):
#         return self.rule_formula.asPDDL()
#
#     def to_formula(self):
#         return self.rule_formula
#
#     @staticmethod
#     def create_red_on_blue_rule(obj1, obj2, rule_type=None):
#
#         if rule_type == 1:
#             variables = ("?x", "?y")
#         elif rule_type == 2:
#             variables = ("?y", "?x")
#             obj1, obj2 = obj2, obj1
#         else:
#             raise ValueError(f'Invalid Rule Type. Expected 1 or 2 got {rule_type}')
#
#         variables = pddl_functions.make_variable_list(variables)
#         obj1_preds = [Predicate(p, TypedArgList([variables.args[0]])) for p in obj1]
#         obj2_preds = [Predicate(p, TypedArgList([variables.args[1]])) for p in obj2]
#         on = Predicate('on', pddl_functions.make_variable_list(['?x', '?y']))
#
#         if len(obj1_preds) > 1:
#             obj1_formula = Formula(obj1_preds, op='and')
#         else:
#             obj1_formula = Formula(obj1_preds)
#
#         if len(obj2_preds) > 1:
#             obj2_formula = Formula(obj2_preds, op='and')
#         else:
#             obj2_formula = Formula(obj2_preds)
#
#         second_part = Formula([obj2_formula, on], op='and')
#         existential = Formula([second_part], op='exists',
#                               variables=pddl_functions.make_variable_list([variables.args[1].arg_name]))
#         neg_o1 = Formula([obj1_formula], op='not')
#         subformula = Formula([neg_o1, existential], op='or')
#
#         return Rule(Formula([subformula], op='forall',
#                        variables=pddl_functions.make_variable_list([variables.args[0].arg_name])))
#
#     @staticmethod
#     def create_not_red_on_blue_rule(obj1, obj2, variables=("?x", "?y")):
#         variables = pddl_functions.make_variable_list(variables)
#         obj1_preds = [Predicate(p, TypedArgList([variables.args[0]])) for p in obj1]
#         obj2_preds = [Predicate(p, TypedArgList([variables.args[1]])) for p in obj2]
#         on = Predicate('on', pddl_functions.make_variable_list(['?x', '?y']))
#
#         if len(obj1_preds) > 1:
#             obj1_formula = Formula(obj1_preds, op='and')
#         else:
#             obj1_formula = Formula(obj1_preds)
#
#         if len(obj2_preds) > 1:
#             obj2_formula = Formula(obj2_preds, op='and')
#         else:
#             obj2_formula = Formula(obj2_preds)
#
#         conjunction_part = Formula([obj1_formula, obj2_formula, on], op='and')
#         existential = Formula([conjunction_part], op='exists', variables=variables)
#         neg_formula = Formula([existential], op='not')
#
#         return Rule(neg_formula)
#
#     @staticmethod
#     def get_rule_colours(formula):
#         """This function only works if the formula has 2 colours
#         (one before and one after the arrow)
#         To allow for more than one object on either side of the rule more logic must be added"""
#         c1, right = formula.subformulas[0].subformulas
#         c1 = c1.get_predicates(False)[0]
#         c1 = c1.get_predicates(True)[0]
#         c1 = c1.name
#         c2 = right.subformulas[0].subformulas[0].get_predicates(True)[0].name
#         return c1, c2
#
#     @staticmethod
#     def get_rule_colours_existential(formula):
#         """This function only works if the formula has 2 colours
#         (one before and one after the arrow)
#         To allow for more than one object on either side of the rule more logic must be added"""
#         and_formula = formula.subformulas[0].subformulas[0]
#         c1 = and_formula.subformulas[0].get_predicates(True)[0].name
#         c2 = and_formula.subformulas[1].get_predicates(True)[0].name
#         return c1, c2
#
#     @staticmethod
#     def get_rule_type(formula):
#         if formula.op == 'not':
#             return 3
#         elif formula.variables.to_pddl() == '?x':
#             return 1
#         elif formula.variables.to_pddl() == '?y':
#             return 2
#         else:
#             raise ValueError('Unknown Rule Type')
#
#     @staticmethod
#     def get_rules(goal):
#         return [Rule(subformula) for subformula in goal.subformulas[1:]]
#
#     @staticmethod
#     def generate_red_on_blue_options(red, blue):
#         return[Rule.create_red_on_blue_rule(red, blue, rule_type=1),
#               Rule.create_red_on_blue_rule(red, blue, rule_type=2)]
#
#     def generate_CPD(self):
#         cpd_line_corr0 = []
#         cpd_line_corr1 = []
#         for i in range(2):
#             for j in range(2):
#                 for r in range(2):
#                     result = r * (1 - int(self.evaluate_rule(i, j)))
#                     cpd_line_corr1.append(result)
#                     cpd_line_corr0.append(1 - result)
#
#         return [cpd_line_corr0, cpd_line_corr1]
#
#     def evaluate_rule(self, value1, value2):
#         c1_set = set()
#         c2_set = set()
#         if value1 == 1:
#             c1_set.add('o1')
#         if value2 == 1:
#             c2_set.add('o2')
#         v = [(self.c1, c1_set), (self.c2, c2_set),
#              ('on', set([('o1', 'o2')]))]
#         val = Valuation(v)
#         dom = val.domain
#         m = Model(dom, val)
#         g = nltk.sem.Assignment(dom)
#         return m.evaluate(str(self), g)
#
#     def generate_table_cpd(self):
#         cpd_line_corr0 = []
#         cpd_line_corr1 = []
#
#
#         for redo1 in range(2):
#             for blueo2 in range(2):
#                 for redo3 in range(2):
#                     for blueo3 in range(2):
#                         for r1 in range(2): # rule 1 or 0
#                             if self.rule_type == 1:
#                                 result = r1 * int(not (redo1) and blueo2 and redo3)
#                             elif self.rule_type == 2:
#                                 result = r1 * int(redo1 and not (blueo2) and blueo3)
#                             cpd_line_corr1.append(result)
#                             cpd_line_corr0.append(1 - result)
#         return [cpd_line_corr0, cpd_line_corr1]


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
            if not result:
                pass_all = False
                if constraint.rule_type == 1:
                    increase.add(constraint.c2)
                    decrease.add(constraint.c1)
                elif constraint.rule_type == 2:
                    increase.add(constraint.c1)
                    decrease.add(constraint.c2)
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

    def to_pddl(self):
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
