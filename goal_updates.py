import pddl_functions
import ff
import block_plotting
from pythonpddl2.pddl import Predicate, TypedArg, TypedArgList, Formula

def make_variable_list(variables):
    var_list = [TypedArg(arg) for arg in variables]
    return TypedArgList(var_list)


def create_goal(obj1, obj2, variables=("?x", "?y")):
    variables = make_variable_list(variables)
    obj1_preds = [Predicate(p, TypedArgList([variables.args[0]])) for p in obj1]
    obj2_preds = [Predicate(p, TypedArgList([variables.args[1]])) for p in obj2]
    on = Predicate('on', make_variable_list(['?x', '?y']))

    if len(obj1_preds) > 1:
        obj1_formula = Formula(obj1_preds, op='and')
    else:
        obj1_formula = Formula(obj1_preds)

    if len(obj2_preds) > 1:
        obj2_formula = Formula(obj2_preds, op='and')
    else:
        obj2_formula = Formula(obj2_preds)
    neg_o1 = Formula([obj1_formula], op='not')
    subformula = Formula([neg_o1, obj2_formula], op='or')

    return Formula([subformula], op='forall', variables=variables)


def create_goal_options(obj1, obj2):
    return [create_goal(obj1, obj2), create_goal(obj2, obj1, variables=("?y", "?x"))]
