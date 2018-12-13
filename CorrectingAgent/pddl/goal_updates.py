import pddl_functions
from pythonpddl.pddl import Predicate, TypedArgList, Formula
import copy


def create_goal(obj1, obj2, variables=("?x", "?y")):
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
    existential = Formula([second_part], op='exists', variables=pddl_functions.make_variable_list([variables.args[1].arg_name]))
    neg_o1 = Formula([obj1_formula], op='not')
    subformula = Formula([neg_o1, existential], op='or')

    return Formula([subformula], op='forall', variables=pddl_functions.make_variable_list([variables.args[0].arg_name]))

def create_negative_goal(obj1, obj2, variables=("?x", "?y")):
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

    conjunction_part = Formula([obj1_formula, obj2_formula, on], op='and')
    existential = Formula([conjunction_part], op='exists', variables=variables)
    neg_formula = Formula([existential], op='not')

    return neg_formula

def create_default_goal():
    var = pddl_functions.make_variable_list(['?x'])
    pred = Predicate('in-tower', var)
    goal = Formula([pred], op='forall', variables=var)
    return goal

def create_goal_options(obj1, obj2):
    return [create_goal(obj1, obj2), create_goal(obj2, obj1, variables=("?y", "?x"))]


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

def goal_from_list(rules):
    goal = create_default_goal()
    for rule in rules:
        goal = update_goal(goal, rule)
    return goal
