import pytest
from correctingagent.world.world_generation import *

def test_rules_consistent1():

    r1 = problem_def.Ruledef(['red'], ['blue'], 'first')
    r2 = problem_def.Ruledef(['red'], ['green'], 'first')
    r3 = problem_def.Ruledef(['green'], ['red'], 'second')
    r4 = problem_def.Ruledef(['orange'], ['red'], 'second')
    r5 = problem_def.Ruledef(['orange'], ['red'], 'first')
    r6 = problem_def.Ruledef(['green'], ['red'], 'first')

    assert(rules_consistent([r1, r2]) is False)  # red constrained twice (first and first)
    assert(rules_consistent([r3, r4]) is False)  # red constrained twice (second and second)
    assert(rules_consistent([r5, r6]) is True)  # red not constrained object twice

def test_rules_consistent2():
    r1 = problem_def.Ruledef(['red'], ['blue'], 'first')
    r2 = problem_def.Ruledef(['red'], ['green'], 'first')
    r3 = problem_def.Ruledef(['green'], ['red'], 'second')
    r4 = problem_def.Ruledef(['orange'], ['red'], 'second')
    r5 = problem_def.Ruledef(['orange'], ['red'], 'first')
    r6 = problem_def.Ruledef(['green'], ['red'], 'first')
    r7 = problem_def.Ruledef(['pink'], ['yellow'], 'second')

    assert(rules_consistent([r4, r5]) is False)  # red constrained and not constrained
    assert(rules_consistent([r2, r3]) is False)  # red constrained twice (second and first)
    assert(rules_consistent([r6, r7]) is True)  # No relation between rules
