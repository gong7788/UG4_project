import pddl_functions
from collections import namedtuple
import goal_updates
from pythonpddl.pddl import Problem

Ruledef = namedtuple('Ruledef', ['first_obj', 'second_obj', 'constrained_obj'])

def create_objects(n):
    obj_names = ['b{}'.format(i) for i in range(n)]
    return pddl_functions.make_variable_list(obj_names)

def generate_default_position(objects):
    """Generates the base predicates placing all objects on the table"""
    initstate = [pddl_functions.create_formula('arm-empty', [])]
    for o in objects.args:
        obj = o.arg_name
        #create on-table
        initstate.append(pddl_functions.create_formula('on-table', [obj]))
        #create clear
        initstate.append(pddl_functions.create_formula('clear', [obj]))
    return initstate


def add_colours(state, objects, colours):
    for o, c in zip(objects.args, colours):
        o = o.arg_name
        state.append(pddl_functions.create_formula(c, [o]))
    return state

def generate_rule(ruledef):
    return goal_updates.create_goal_options(
            ruledef.first_obj, ruledef.second_obj
        )[int(ruledef.constrained_obj == 'second')]

def create_problem(colours, ruledefs, name='block-problem', domainname='blocksworld'):
    objects = create_objects(len(colours))
    initialstate = generate_default_position(objects)
    initialstate = add_colours(initialstate, objects, colours)
    goal = goal_updates.create_default_goal()
    for rule in ruledefs:
        rule = generate_rule(rule)
        goal = goal_updates.update_goal(goal, rule)
    return Problem(name, domainname, objects, initialstate, goal)

def save_problem(colours, ruledefs, name='blocks-problem', domainnname='blocksworld'):
    problem = create_problem(colours, ruledefs, name=name, domainnname=domainnname)
    with open('pddl/{}'.format(name), 'w') as f:
        f.write(problem.asPDDL())

if __name__ == '__main__':
    rule1 = Ruledef(['red'], ['blue'], 'first')
    rule2 = Ruledef(['green'], ['pink'], 'second')
    print(create_problem(['red', 'pink', 'green', 'red', 'blue'], [rule1, rule2]).asPDDL())
