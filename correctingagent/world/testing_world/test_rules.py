import pytest

from correctingagent.world import world
from correctingagent.world.rules import *

def test_from_formula():
    w = world.PDDLWorld(problem_directory='testing', problem_number=1)

    goal = w.problem.goal
    r1 = goal.subformulas[-1]
    rule = Rule.from_formula(r1)
    assert(isinstance(rule, RedOnBlueRule))
    assert(rule.c1 == 'blue')
    assert(rule.c2 == 'pink')
    assert(rule.asPDDL() == r1.asPDDL())


def test_get_all_relevant_colours():
     r1 = RedOnBlueRule('red', 'blue', 2)
     r2 = RedOnBlueRule('red', 'green', 2)

     abc, atc = r1.get_all_relevant_colours([r2])
     assert(abc == ['green'])
     assert(atc == [])


def test_get_all_relevant_colours2():
     r3 = RedOnBlueRule('red', 'green', 1)
     r4 = RedOnBlueRule('pink', 'green', 1)
     abc, atc = r3.get_all_relevant_colours([r4])
     assert(abc == [])
     assert(atc == ['pink'])


def test_tower_correction():
    w = world.PDDLWorld(problem_directory='testing', problem_number=1)

    rule = RedOnBlueRule('blue', 'pink', 2)
    rule2 = RedOnBlueRule('blue', 'pink', 1)

    w.update('put', ['b8', 't0'])  # pink
    w.update('put', ['b9', 'b8'])  # blue
    w.update('put', ['b6', 'b9'])  # pink
    w.update('put', ['b4', 'b6'])  # purple

    assert(rule.check_tower_violation(w.state) is True)
    assert (rule2.check_tower_violation(w.state) is False)


def test_table_correction():
    w = world.PDDLWorld(problem_directory='testing', problem_number=1)

    rule = RedOnBlueRule('blue', 'pink', 2)
    rule2 = RedOnBlueRule('blue', 'pink', 1)

    w.update('put', ['b8', 't0'])  # pink
    w.update('put', ['b9', 'b8'])  # blue
    w.update('put', ['b7', 'b9'])  # blue

    assert(rule.check_table_violation(w.state) is True)
    assert (rule2.check_table_violation(w.state) is False)


def test_colour_count_parse():
    w = world.PDDLWorld(problem_directory='multitower', problem_number=1)

    rule = w.problem.goal.subformulas[1]

    r = ColourCountRule.from_formula(rule)
    assert(r.colour_name == 'blue')
    assert(r.number == 1)
    assert(rule.asPDDL() == r.to_formula().asPDDL())
    assert(rule.asPDDL() == r.asPDDL())

    r = Rule.from_formula(rule)
    assert(r.colour_name == 'blue')
    assert(r.number == 1)
    assert(rule.asPDDL() == r.to_formula().asPDDL())
    assert(rule.asPDDL() == r.asPDDL())


def test_colour_count_violation():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl', problem_directory='multitower', problem_number=1)

    rule = ColourCountRule('blue', 1)

    w.update('put', ['b0', 't0', 'tower0'])
    w.update('put', ['b5', 'b0', 'tower0'])

    assert(rule.check_tower_violation(w.state) is True)
    w.back_track()
    w.update('put', ['b5', 't1', 'tower1'])
    assert(rule.check_tower_violation(w.state) is False)


def test_table_correction_doesnt_overinfer():
    w = world.PDDLWorld(problem_directory='testing', problem_number=3)

    rule1 = RedOnBlueRule('pink', 'blue', 1)
    rule2 = RedOnBlueRule('red', 'blue', 1)

    w.update('put', ['b3', 't0'])  # blue
    w.update('put', ['b2', 'b3'])  # red
    w.update('put', ['b9', 'b2'])  # blue
    w.update('put', ['b8', 'b9'])  # green

    assert(rule1.check_table_violation(w.state, [rule2]) is True)
    assert(rule2.check_table_violation(w.state, [rule1]) is False)


def test_colour_count_table_violation():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl', problem_directory='multitower', problem_number=1)

    rule = ColourCountRule('blue', 1)
    rule2 = RedOnBlueRule('blue', 'red', 2)
    rule3 = ColourCountRule('green', 1)

    w.update('put', ['b0', 't0', 'tower0'])
    w.update('put', ['b1', 'b0', 'tower0'])

    assert(rule.check_table_violation(w.state, [rule2, rule3]) is True)
    w.back_track()
    w.update('put', ['b1', 't1', 'tower1'])
    assert(rule.check_table_violation(w.state, [rule2, rule3]) is False)


def test_colour_count_table_violation2():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl', problem_directory='multitower', problem_number=3)

    rule = ColourCountRule('blue', 1)
    rule2 = RedOnBlueRule('blue', 'red', 1)

    w.update('put', ['b4', 't0', 'tower0'])  # red
    w.update('put', ['b0', 'b4', 'tower0'])  # blue
    w.update('put', ['b1', 'b0', 'tower0'])  # red

    assert(rule.check_table_violation(w.state, [rule2]) is True)
    w.back_track()
    w.update('put', ['b1', 't1', 'tower1'])  # red
    assert(rule.check_table_violation(w.state, [rule2]) is False)
