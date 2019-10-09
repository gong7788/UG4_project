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

    assert(rule.check_table_violation(w.state) == 'b1')
    assert (rule2.check_table_violation(w.state) is False)
