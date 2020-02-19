import pytest

from correctingagent.world.rules import ColourCountRule
from correctingagent.world.world_generation import *
from correctingagent.world.problem_def import *

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

    assert(rules_consistent([r4, r5]) is True)  # bijection
    assert(rules_consistent([r4, r6]) is False)  # red constrained and not constrained with different colours
    assert(rules_consistent([r2, r3]) is False)  # red constrained twice (second and first)
    assert(rules_consistent([r6, r7]) is True)  # No relation between rules

def test_extended_blocks_world_problem():
    cc = ColourCountRule('blue', 2)
    colours = ['red', 'blue', 'green', 'yellow', 'pink', 'purple', 'red', 'blue', 'blue', 'blue']
    problem = problem_def.ExtendedBlocksWorldProblem(num_blocks=10, num_towers=2, rules=[cc.to_formula()], colours=colours)

    assert(problem.goal.subformulas[0].asPDDL() == "(forall (?x) (done ?x))")
    assert(problem.goal.subformulas[1].asPDDL() == cc.asPDDL())

    state = PDDLState.from_problem(problem)

    for i, colour in enumerate(colours):
        assert(state.predicate_holds(colour, [f"b{i}"]))

        assert(state.predicate_holds('clear', [f'b{i}']))
        assert(state.predicate_holds('on-table', [f'b{i}']))

    for colour in ['red', 'blue', 'green', 'yellow', 'pink', 'purple', 'orange']:
        assert(state.get_colour_count(colour, 'tower0') == 0)
        assert (state.get_colour_count(colour, 'tower1') == 0)


