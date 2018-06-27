import world
import agents
from ff import Solved
import ff
import pddl_functions


def tower_correction(obj1, obj2):
    return "no, put {} blocks on {} blocks".format(obj1, obj2)

def table_correction(obj1, obj2, obj3):
    return "No, now you cannot put {} in the tower because you must put {} blocks on {} blocks".format(obj3, obj1, obj2)

def get_top_two(world_):
    r = world_.sense().relations
    for o in r.keys():
        if 'clear' in r[o] and 'in-tower' in r[o]:
            o1 = o
    on = r[o1]['on']
    o2 = on.args.args[1].arg_name
    return o1, o2

def get_rules(goal):
    for f in goal.subformulas:
        if "in-tower" not in f.asPDDL():
            yield f

def get_name(formula):
    return formula.get_predicates(True)[0].name

def get_args(formula):
    return list(map(lambda x: x.arg_name, formula.get_predicates(True)[0].args.args))

def get_relevant_colours(rule):
    or_ = rule.subformulas[0]
    r1, r2 = or_.subformulas
    colour1 = (r1.subformulas[0])
    colour2 = (list(filter(lambda x: get_name(x) != 'on', r2.subformulas[0].subformulas))[0])
    on = list(filter(lambda x: get_name(x) == 'on', r2.subformulas[0].subformulas))[0]
    o1, o2 = on.get_predicates(True)[0].args.args
    out = (o1, o2)
    c1 = list(filter(lambda x: get_args(x)[0] == o1.arg_name, [colour1, colour2]))[0]
    c2 = list(filter(lambda x: get_args(x)[0] == o2.arg_name, [colour1, colour2]))[0]
    return get_name(c1), get_name(c2), get_name(colour1) # on(c1, c2) colour1 -> colour2

def check_rule_violated(rule, world_):
    c1, c2, impl = get_relevant_colours(rule)
    o1, o2 = get_top_two(world_)
    state = world_.problem.initialstate
    f1 = pddl_functions.create_formula(c1, [o1])
    f2 = pddl_functions.create_formula(c2, [o2])
    c1_o1 = pddl_functions.predicate_holds(f1.get_predicates(True)[0], state)
    c2_o2 = pddl_functions.predicate_holds(f2.get_predicates(True)[0], state)
    if c1_o1 and not c2_o2 and c1 == impl:
        return True
    if not c1_o1 and c2_o2 and c2 == impl:
        return True
    return False

def check_table_rule_violation(rule, world_):
    c1, c2, impl = get_relevant_colours(rule)
    o1, o2 = get_top_two(world_)
    state = world_.problem.initialstate
    f1 = pddl_functions.create_formula(c1, [o1])
    f2 = pddl_functions.create_formula(c2, [o2])
    c1_o1 = pddl_functions.predicate_holds(f1.get_predicates(True)[0], state)
    c2_o2 = pddl_functions.predicate_holds(f2.get_predicates(True)[0], state)
    table_objs = blocks_on_table(world_)
    c1_count = count_coloured_blocks(c1, table_objs, state)
    c2_count = count_coloured_blocks(c2, table_objs, state)
    if not c1_o1 and c2_o2 and c1 == impl:
        if c1_count > c2_count:
            return get_block_with_colour(c1, table_objs, state)
    if c1_o1 and not c2_o2 and c2 == impl:
        if c2_count > c1_count:
            return get_block_with_colour(c2, table_objs, state)
    return False

def blocks_on_table(world_):
    r = world_.sense().relations
    objs = []
    for o in r.keys():
        if 'on-table' in r[o] and 'clear' in r[o]:
            objs.append(o)
    return objs

def count_coloured_blocks(colour, objects, state):
    count = 0
    for o in objects:
        c_o = pddl_functions.create_formula(colour, [o]).get_predicates(True)[0]
        if pddl_functions.predicate_holds(c_o, state):
            count += 1
    return count

def get_block_with_colour(colour, objects, state):
    for o in objects:
        c_o = pddl_functions.create_formula(colour, [o]).get_predicates(True)[0]
        if pddl_functions.predicate_holds(c_o, state):
            return o
    return

def teacher_correction(w):
    if not w.test_failure():
        return ""
    #Reasons for failure:
    # a->b, a -b
    # b-> a b -a
    rules = list(get_rules(w.problem.goal))
    for r in rules:
        if check_rule_violated(r, w):
            print(r.asPDDL())
            c1, c2, _ = get_relevant_colours(r)
            return tower_correction(c1, c2)
    for r in rules:
        o3 = check_table_rule_violation(r, w)
        if o3:
            c1, c2, _ = get_relevant_colours(r)
            return table_correction(c1, c2, o3)


class Teacher(object):

    def correction(self, world_):
        raise NotImplementedError()

    def answer_question(self, question, world_):
        raise NotImplementedError()


class HumanTeacher(Teacher):

    def correction(self, world_):
        return input('Correction?')

    def answer_question(self, question, world_):
        return input(question)



class TeacherAgent(Teacher):

    def correction(self, w):
        if not w.test_failure():
            return ""
        #Reasons for failure:
        # a->b, a -b
        # b-> a b -a
        rules = list(get_rules(w.problem.goal))
        for r in rules:
            if check_rule_violated(r, w):
                print(r.asPDDL())
                c1, c2, _ = get_relevant_colours(r)
                return tower_correction(c1, c2)
        for r in rules:
            o3 = check_table_rule_violation(r, w)
            if o3:
                c1, c2, _ = get_relevant_colours(r)
                return table_correction(c1, c2, o3)

    def answer_question(self, question, world_):
        if "Is the top object" in question:
            colour = question.strip("Is the top object").strip("?").strip()
            o1, o2 = get_top_two(world_)
            o1_colour = pddl_functions.create_formula(colour, [o1]).get_predicates(True)[0]
            o1_is_colour = pddl_functions.predicate_holds(o1_colour, world_.problem.initialstate)
            if o1_is_colour:
                return "yes"
            else:
                return "no"
