import pytest

from correctingagent.world import world
from correctingagent.world.rules import *

def test_from_formula():
    w = world.PDDLWorld(problem_directory='testing', problem_number=1)

    goal = w.problem.goal
    r1 = goal.subformulas[-1]
    rule = BaseRule.from_formula(r1)
    assert(isinstance(rule, RedOnBlueRule))
    assert(rule.c1 == 'blue')
    assert(rule.c2 == 'pink')
    assert(rule.to_pddl() == r1.asPDDL())

