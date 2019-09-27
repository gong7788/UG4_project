import correctingagent.world.rules
from correctingagent.pddl import pddl_functions
from collections import namedtuple
from correctingagent.world import goals
from pythonpddl.pddl import Problem
from correctingagent.util.colour_dict import colour_dict



Ruledef = namedtuple('Ruledef', ['first_obj', 'second_obj', 'constrained_obj'])


class BlocksWorldProblem(Problem):

    def __init__(self, name='blocks-problem', domainname='blocks-domain', objects=None, n=10, m=1,
                 colours=[], rules=None, metric=None):
        if objects is not None:
            objects = pddl_functions.make_variable_list(objects)
        else:
            objects = BlocksWorldProblem.create_objects(n, m)

        initialstate = BlocksWorldProblem.generate_default_position(objects)
        initialstate = BlocksWorldProblem.add_colours(initialstate, objects, colours)
        goal = goals.create_default_goal()
        goal = goals.update_goal(goal, rules)
        super(BlocksWorldProblem, self).__init__(name, domainname, objects, initialstate, goal, metric)

    @staticmethod
    def create_objects(n, m=1):
        obj_names = ['b{}'.format(i) for i in range(n)]
        tower_posns = ['t{}'.format(i) for i in range(m)]
        obj_names.extend(tower_posns)
        return pddl_functions.make_variable_list(obj_names)

    @staticmethod
    def generate_default_position(objects):
        """Generates the base predicates placing all objects on the table"""
        initstate = [pddl_functions.create_formula('arm-empty', [])]
        for o in objects.args:
            obj = o.arg_name
            #create on-table
            if 't' in obj:
                initstate.append(pddl_functions.create_formula('in-tower', [obj]))
            else:
                initstate.append(pddl_functions.create_formula('on-table', [obj]))
            #create clear
            initstate.append(pddl_functions.create_formula('clear', [obj]))

        return initstate

    @staticmethod
    def add_colours(state, objects, colours):
        for o, c in zip(objects.args, colours):
            o = o.arg_name
            state.append(pddl_functions.create_formula(c, [o]))
            for key, value in colour_dict.items():
                if c in value and c != key:
                    state.append(pddl_functions.create_formula(key, [o]))
                    break
        return state

    @staticmethod
    def generate_problem(colours, rules, name='blocks-problem', domainname='blocksworld'):
        return BlocksWorldProblem(name=name, domainname=domainname, n=len(colours), colours=colours, rules=rules)


def generate_rule(ruledef):
    #TODO make downstream task use Rule rather than formula
    return correctingagent.world.rules.Rule.generate_red_on_blue_options(
            ruledef.first_obj, ruledef.second_obj
        )[int(ruledef.constrained_obj == 'second')].asFormula()


if __name__ == '__main__':
    rule1 = Ruledef(['red'], ['blue'], 'first')
    rule2 = Ruledef(['green'], ['pink'], 'second')
    print(BlocksWorldProblem.generate_problem(['red', 'pink', 'green', 'red', 'blue'], [rule1, rule2]).asPDDL())
