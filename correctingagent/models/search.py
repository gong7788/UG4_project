import numpy as np
import os
import heapq
import copy
from collections import defaultdict

from ..pddl import goal_updates
from ..pddl import pddl_functions
from ..pddl.ff import NoPlanError, IDontKnowWhatIsGoingOnError, ImpossibleGoalError
from ..pddl import ff
from ..util.util import get_config


c = get_config()
data_location = c['data_location']



class ActiveLearningTest(object):

    def __init__(self, rule1, rule2, data, c3_model, c2_obj, w=None):
        self.rule1 = Rule(rule1)
        self.rule2 = Rule(rule2)
        self.failed = False
        results = {}

        # for obj in pddl_functions.filter_tower_locations(data.keys(), get_locations=False):
        for obj in w.objects_not_in_tower():
            datum = data[obj]
            p_colour = c3_model.p(1, datum)
            results[obj] = p_colour

        least_likely_obj = min(results, key=results.get)
        self.objects = (least_likely_obj, c2_obj)
        self.test_formula = pddl_functions.create_formula('on', [least_likely_obj, c2_obj])


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
            out_goal = goal_updates.update_goal(out_goal, rule)
        g = goal_updates.update_goal(goal, out_goal)

        return g

class State(object):

    def __init__(self, obs, colour_choices={}, threshold=0.5):
        self.initialstate = obs.state
        clear_objs = pddl_functions.get_clear_objs(obs)
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
        pddl_state = copy.copy(self.initialstate)
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

def get_rule_type(formula):
    if formula.op == 'not':
        return 3
    elif formula.variables.asPDDL() == '?x':
        return 1
    elif formula.variables.asPDDL() == '?y':
        return 2
    else:
        raise ValueError('Unknown Rule Type')


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

def get_rule_colours_existential(formula):
    """This function only works if the formula has 2 colours
    (one before and one after the arrow)
    To allow for more than one object on either side of the rule more logic must be added"""
    and_formula = formula.subformulas[0].subformulas[0]
    c1 = and_formula.subformulas[0].get_predicates(True)[0].name
    c2 = and_formula.subformulas[1].get_predicates(True)[0].name
    return c1, c2

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


class Rule(object):

    def __init__(self, rule_formula):
        self.rule_type = get_rule_type(rule_formula)
        if self.rule_type == 1:
            self.c1, self.c2 = get_rule_colours(rule_formula)
        elif self.rule_type == 2:
            self.c2, self.c1 = get_rule_colours(rule_formula)
        elif self.rule_type == 3:
            self.c1, self.c2 = get_rule_colours_existential(rule_formula)
        self.constraint = RuleConstraint(self.rule_type, self.c1, self.c2)
        self.rule_formula = rule_formula
        if self.rule_type in [1, 2]:
            self.name = 'r_{}^({},{})'.format(self.rule_type, self.c1, self.c2)
        else:
            self.name = 'r_not^({},{})'.format(self.c1, self.c2)


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

def get_rules(goal):
    return goal.subformulas[1:]


class Planner(object):

    def __init__(self, colour_choices, obs, goal, tmp_goal, problem, domain_file='blocks-domain.pddl'):
        self.current_state = State(obs, colour_choices)
        rules = get_rules(goal)
        rules = [Rule(r) for r in rules]
        self.constraints = ConstraintCollection.from_rules(rules)
        self.searched_states = {tuple(self.current_state.state)}
        self.domain_file = os.path.join(data_location, 'domain', domain_file)
        self.goal = goal
        self.tmp_goal = tmp_goal
        self.problem = problem
        self.state_queue = []
        # print(colour_choices)

    def _pop(self):
        return heapq.heappop(self.state_queue)

    def _push(self, state):
        heapq.heappush(self.state_queue, (state.score, state))


    def evaluate_current_state(self, default_plan=False):
        if default_plan:
            self.current_state.state = []
            self.current_state.colour_counts = {c:0 for c in self.current_state.colour_counts.keys()}
        success, increase, decrease = self.constraints.evaluate(self.current_state)
        #print(self.current_state.state)
        #print(success)
        if success:
            self.problem.goal = goal_updates.update_goal(self.goal, self.tmp_goal)
            self.problem.initialstate = self.current_state.to_pddl()
            tmp_problem = os.path.join(data_location, 'tmp/search-problem.pddl')
            print(self.domain_file)
            with open(tmp_problem, 'w') as f:
                f.write(self.problem.asPDDL())
            try:
                plan = ff.run(self.domain_file, tmp_problem)
                return plan
            except (NoPlanError, IDontKnowWhatIsGoingOnError, ImpossibleGoalError) as e:
                # print(e)
                # for p in self.problem.initialstate:
                #     print(p.asPDDL())
                # print(self.problem.goal.asPDDL())
                # n = len(os.listdir('errors/domain'))
                # with open('errors/domain/error{}.domain'.format(n), 'w') as f:
                #     f.write(self.problem.asPDDL())
                try:
                    score, self.current_state = self._pop()
                    return False
                except IndexError:
                    self.generate_candidates(self.current_state, increase, decrease)

        else:
            self.generate_candidates(self.current_state, increase, decrease)

        try:
            score, self.current_state = self._pop()
            return False
        except IndexError:
            raise NoPlanError('Search could not find a possible plan')


    def plan(self):
        # print(self.goal.asPDDL())
        plan = False
        for i in range(200):
            # print(self.current_state.score, self.current_state.state)
            try:
                plan = self.evaluate_current_state()
            except NoPlanError:
                break
            if plan:
                return plan

        return self.evaluate_current_state(default_plan=True)


    def add_candidate(self, colour, increase_count=True):
        new_state = copy.deepcopy(self.current_state)
        new_state.flip_colour(colour, increase_count=increase_count)
        if tuple(new_state.state) not in self.searched_states:
            self.searched_states.add(tuple(new_state.state))
            self._push(new_state)

    def generate_candidates(self, state, increase, decrease):
        if not increase and not decrease:
            for colour in self.current_state.colours:
                # flip the best candidate of each colour

                # flip it to increase the count
                self.add_candidate(colour, increase_count=True)
                self.add_candidate(colour, increase_count=False)
                # flip it to decrease the count
        else:
            for colour in increase:
                self.add_candidate(colour, increase_count=True)

            for colour in decrease:
                self.add_candidate(colour, increase_count=False)



class NoLanguagePlanner(Planner):

    def __init__(self, colour_choices, obs, tests, goal, tmp_goal, problem, domain_file='blocks-domain.pddl', **kwargs):

        super().__init__(colour_choices, obs, goal, tmp_goal, problem, domain_file=domain_file, **kwargs)
        self.initial_state = self.current_state
        self.initial_goal = self.goal
        self.tests = tests

        tmp_goal = goal_updates.update_goal(self.goal, tmp_goal)
        for test in tests:

            test_goal = goal_updates.update_goal(tmp_goal, test.test_formula)

            self.problem.goal = test_goal
            self.problem.initialstate = self.current_state.to_pddl()
            # print(test_goal.asPDDL())
            with open('tmp/problem.pddl', 'w') as f:
                f.write(self.problem.asPDDL())
            try:
                plan = ff.run(self.domain_file, 'tmp/problem.pddl')
            except (ImpossibleGoalError, IDontKnowWhatIsGoingOnError):

                test.failed = True
                continue
            except (IDontKnowWhatIsGoingOnError, NoPlanError):
                self.goal = goal_updates.update_goal(self.goal, test.test_formula)
                tmp_goal = goal_updates.update_goal(tmp_goal, test.test_formula)
            else:
                self.goal = goal_updates.update_goal(self.goal, test.test_formula)
                tmp_goal = goal_updates.update_goal(tmp_goal, test.test_formula)
