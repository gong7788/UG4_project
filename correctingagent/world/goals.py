from correctingagent.pddl import pddl_functions
from pythonpddl.pddl import Predicate, Formula
import copy


class Goal(object):

    def __inti__(self):
        self.basic_goal = create_default_goal()


def create_default_goal(domain_file):
    if 'updated' not in domain_file and 'blocks-domain-colour-unknown-cc' not in domain_file:
        var = pddl_functions.make_variable_list(['?x'])
        pred = Predicate('in-tower', var)
        goal = Formula([pred], op='forall', variables=var)
        return goal
    else:
        var = pddl_functions.make_variable_list(['?x'])
        pred = Predicate('done', var)
        goal = Formula([pred], op='forall', variables=var)
        return goal

def create_defualt_goal_new():
    var = pddl_functions.make_variable_list(['?x'])
    pred = Predicate('done', var)
    goal = Formula([pred], op='forall', variables=var)
    return goal

def update_goal(goal, rule):
    goal = copy.deepcopy(goal)
    if rule is None:
        return goal
    if goal is None:
        goal = rule
    elif goal.op == 'and':
        goal.subformulas.append(rule)
    else:
        goal = Formula([goal, rule], op='and')
    return goal


def goal_from_list(rules, domain_file):
    goal = create_default_goal(domain_file)
    for rule in rules:
        goal = update_goal(goal, rule)
    return goal
