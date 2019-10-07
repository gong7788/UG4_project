import os
import heapq
import copy
from pathlib import Path

from correctingagent.world.rules import Rule, ConstraintCollection, State
from correctingagent.world import goals
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


class Planner(object):

    def __init__(self, colour_choices, obs, goal, tmp_goal, problem, domain_file='blocks-domain.pddl', use_metric_ff=False):
        c = get_config()
        data_location = Path(c['data_location'])
        self.use_metric_ff = use_metric_ff
        self.current_state = State(obs, colour_choices)
        rules = Rule.get_rules(goal)
        self.constraints = ConstraintCollection.from_rules(rules)
        self.searched_states = {tuple(self.current_state.state)}
        self.domain_file = data_location / 'domain' / domain_file
        self.goal = goal
        self.tmp_goal = tmp_goal
        self.problem = problem
        self.state_queue = []
        # print(colour_choices)
        self.search_dir = data_location / 'tmp' / 'search_problem'
        n = len(os.listdir(self.search_dir))
        self.search_file = os.path.join(self.search_dir, f'{n}.pddl')

    def _pop(self):
        return heapq.heappop(self.state_queue)

    def _push(self, state):
        heapq.heappush(self.state_queue, (state.score, state))

    def evaluate_current_state(self, default_plan=False):
        if default_plan:
            self.current_state.state = []
            self.current_state.colour_counts = {c: 0 for c in self.current_state.colour_counts.keys()}
        success, increase, decrease = self.constraints.evaluate(self.current_state)

        if success:
            self.problem.goal = goals.update_goal(self.goal, self.tmp_goal)
            self.problem.initialstate = self.current_state.to_pddl()

            with open(self.search_file, 'w') as f:
                f.write(self.problem.asPDDL())
            try:
                plan = ff.run(self.domain_file, self.search_file, use_metric_ff=self.use_metric_ff)
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
        for i in range(20):
            # print(self.current_state.score, self.current_state.state)
            try:
                plan = self.evaluate_current_state()
                if plan:
                    return plan
            except NoPlanError:
                break
            #if plan:
            #    return plan

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

        tmp_goal = goals.update_goal(self.goal, tmp_goal)
        for test in tests:

            test_goal = goals.update_goal(tmp_goal, test.test_formula)

            self.problem.goal = test_goal
            self.problem.initialstate = self.current_state.to_pddl()
            # print(test_goal.asPDDL())
            with open(self.search_file, 'w') as f:
                f.write(self.problem.asPDDL())
            try:
                plan = ff.run(self.domain_file, self.search_file, use_metric_ff=self.use_metric_ff)
            except (ImpossibleGoalError, IDontKnowWhatIsGoingOnError):

                test.failed = True
                continue
            except (IDontKnowWhatIsGoingOnError, NoPlanError):
                self.goal = goals.update_goal(self.goal, test.test_formula)
                tmp_goal = goals.update_goal(tmp_goal, test.test_formula)
            else:
                self.goal = goals.update_goal(self.goal, test.test_formula)
                tmp_goal = goals.update_goal(tmp_goal, test.test_formula)
