import correctingagent.world.rules
from correctingagent.pddl import pddl_functions
from collections import namedtuple

from correctingagent.pddl.pddl_functions import PDDLState
from correctingagent.world import goals
from pythonpddl.pddl import Problem
from correctingagent.util.colour_dict import colour_dict



Ruledef = namedtuple('Ruledef', ['first_obj', 'second_obj', 'constrained_obj'])


class BlocksWorldProblem(Problem):

    def __init__(self, name='blocks-problem', domainname='blocks-domain', objects=None, n=10, m=1,
                 colours=[], rules=None, metric=None):
        self.domainname = domainname
        if objects is not None:
            objects = pddl_functions.make_variable_list(objects)
        else:
            objects = BlocksWorldProblem.create_objects(n, m)

        if isinstance(rules[0], Ruledef):
            rules = [generate_rule(rule) for rule in rules]
        elif isinstance(rules[0], correctingagent.world.rules.Rule):
            rules = [rule.to_formula() for rule in rules]

        initialstate = self.generate_default_position(objects)
        initialstate = BlocksWorldProblem.add_colours(initialstate, objects, colours)
        goal = self.create_default_goal()
        for rule in rules:
            goal = goals.update_goal(goal, rule)
        super(BlocksWorldProblem, self).__init__(name, domainname, objects, initialstate, goal, metric)

    def create_default_goal(self):
        return goals.create_default_goal(self.domainname)

    @staticmethod
    def create_objects(n, m=1):
        obj_names = [f'b{i}' for i in range(n)]
        tower_posns = [f't{i}' for i in range(m)]
        obj_names.extend(tower_posns)
        return pddl_functions.make_variable_list(obj_names)

    def generate_default_position(self, objects):
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


class ExtendedBlocksWorldProblem(BlocksWorldProblem):

    def __init__(self, num_blocks=10, num_towers=2, **kwargs):
        objects = ExtendedBlocksWorldProblem.create_objects(num_blocks, num_towers)
        super(ExtendedBlocksWorldProblem, self).__init__(objects=objects, **kwargs)

    def generate_default_position(self, objects):
        """Generates the base predicates placing all objects on the table"""
        towers = []
        initstate = [pddl_functions.create_formula('arm-empty', [])]
        for o in objects.args:
            obj = o.arg_name
            #create on-table
            if 't' in obj:
                initstate.append(pddl_functions.create_formula('done', [obj]))
                if 'tower' not in obj:
                    number = obj.replace('t', '')
                    initstate.append(pddl_functions.create_formula('in-tower', [obj, f'tower{number}']))
                else:
                    towers.append(obj)
            else:
                initstate.append(pddl_functions.create_formula('on-table', [obj]))
            #create clear
            if 'tower' not in obj:
                initstate.append(pddl_functions.create_formula('clear', [obj]))

        for tower in towers:
            for colour in ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink']:
                initstate.append(pddl_functions.ColourCount(colour, tower, 0).to_formula())
            initstate.append(pddl_functions.create_formula('tower', [tower]))
        return initstate

    def create_default_goal(self):
        return goals.create_defualt_goal_new()

    @staticmethod
    def create_objects(num_blocks, num_towers=1):
        obj_names = [f'b{i}' for i in range(num_blocks)]
        tower_posns = [f't{i}' for i in range(num_towers)]
        towers = [f'tower{i}' for i in range(num_towers)]
        obj_names.extend(tower_posns)
        obj_names.extend(towers)
        return obj_names
        #return pddl_functions.make_variable_list(obj_names)

    @staticmethod
    def generate_problem(colours, rules, name='blocks-problem', domainname='blocksworld', num_towers=2):
        return ExtendedBlocksWorldProblem(name=name, domainname=domainname, num_blocks=len(colours), num_towers=num_towers, colours=colours, rules=rules)


def generate_rule(ruledef):
    #TODO make downstream task use Rule rather than formula
    return correctingagent.world.rules.RedOnBlueRule(ruledef.first_obj[0], ruledef.second_obj[0],
                                                     rule_type=(1+int(ruledef.constrained_obj == 'second'))).to_formula()


if __name__ == '__main__':
    rule1 = Ruledef(['red'], ['blue'], 'first')
    rule2 = Ruledef(['green'], ['pink'], 'second')
    print(BlocksWorldProblem.generate_problem(['red', 'pink', 'green', 'red', 'blue'], [rule1, rule2]).asPDDL())
