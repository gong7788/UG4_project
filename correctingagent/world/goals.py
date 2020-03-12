from correctingagent.pddl import pddl_functions
from pythonpddl.pddl import Predicate, Formula
import copy


class Goal(object):

    def __inti__(self):
        self.basic_goal = create_default_goal()


def create_default_goal(domain_file):
    #print("domain file is", domain_file)

    new_domain_type = 'updated' in domain_file or \
                      'blocks-domain-colour-unknown-cc' in domain_file or \
                      'blocks-domain-unstackable' in domain_file

    if new_domain_type:
        #print("New domain used to create goal")
        var = pddl_functions.make_variable_list(['?x'])
        pred = Predicate('done', var)
        goal = Formula([pred], op='forall', variables=var)
        return goal
    else:
        #print("old domain used to create goal")
        var = pddl_functions.make_variable_list(['?x'])
        pred = Predicate('in-tower', var)
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
