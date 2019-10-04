from pathlib import Path

import pytest
from pythonpddl import pddl

from correctingagent.pddl import pddl_functions
from correctingagent.pddl.pddl_functions import *
from correctingagent.util.util import get_config


def test_fexpressions1():
    f_exp = pddl.FExpression('=',
         [pddl.FHead('blue-count',
                     pddl_functions.make_variable_list(['t0'])),
          pddl.ConstantNumber(1)])

    cc = pddl_functions.ColourCount('blue', 't0', 1)

    assert(f_exp.asPDDL() == cc.to_pddl())


def test_fexpressions2():
    f_exp = pddl.FExpression('=',
         [pddl.FHead('blue-count',
                     pddl_functions.make_variable_list(['t0'])),
          pddl.ConstantNumber(1)])

    cc = pddl_functions.ColourCount.from_fexpression(f_exp)

    assert(f_exp.asPDDL() == cc.to_pddl())


def test_fexpressions3():
    f_exp = pddl.FExpression('=',
         [pddl.FHead('blue-count',
                     pddl_functions.make_variable_list(['t0'])),
          pddl.ConstantNumber(2)])

    cc = pddl_functions.ColourCount('blue', 't0', 1)
    cc.number = 2

    assert(f_exp.asPDDL() == cc.to_pddl())


def test_fexpressions4():
    f_exp = pddl.FExpression('=',
         [pddl.FHead('blue-count',
                     pddl_functions.make_variable_list(['t0'])),
          pddl.ConstantNumber(3)])

    cc = pddl_functions.ColourCount('blue', 't0', 1)
    cc.increment()
    cc.increment()

    assert(f_exp.asPDDL() == cc.to_pddl())


def test_fexpressions5():
        f_exp = pddl.FExpression('=',
                                 [pddl.FHead('blue-count',
                                             pddl_functions.make_variable_list(['t0'])),
                                  pddl.ConstantNumber(2)])

        cc = pddl_functions.ColourCount('blue', 't0', 1)
        cc.increment()
        cc.decrement()
        cc.increment()

        assert (f_exp.asPDDL() == cc.to_pddl())

def test_fexpressions5():
    f_exp = pddl.FExpression('=',
                             [pddl.FHead('blue-count',
                                         pddl_functions.make_variable_list(['t0'])),
                              pddl.ConstantNumber(2)])

    cc = pddl_functions.ColourCount('blue', 't0', 1)
    cc.increment(2)
    cc.decrement()

    assert (f_exp.asPDDL() == cc.to_pddl())

def test_predicate1():
    assert(str(pddl_functions.Predicate('on', ['o1', 'o2'])) == "(on o1 o2)")
    assert (str(pddl_functions.Predicate('hi', ['o1'])) == "(hi o1)")


def test_pddl_state():
    config = get_config()
    data_dir = Path(config['data_location'])
    domain_file = 'blocks-domain-updated.pddl'
    domain_file = data_dir / 'domain' / domain_file
    problem_directory = 'multitower'
    problem_number = 1
    problem_file = data_dir / problem_directory / f'problem{problem_number}.pddl'
    domain, problem = pddl_functions.parse(domain_file, problem_file)

    state = pddl_functions.PDDLState.from_initialstate(problem.initialstate)

    assert([str(pred) for pred in state.get_predicates('b7')] == ["(on-table b7)", "(clear b7)", "(lightyellow b7)", "(yellow b7)"])
    assert(state.get_clear_objs() == ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'g0', 'g1'])

    assert(state.predicate_holds(pddl_functions.Predicate('on-table', ['b7'])) is True)
    assert(state.predicate_holds(pddl_functions.Predicate('red', ['b7'])) is False)
    assert(state.predicate_holds(pddl_functions.Predicate('yellow', ['b7'])) is True)

def test_pddl_state_apply_effect():
    config = get_config()
    data_dir = Path(config['data_location'])
    domain_file = 'blocks-domain-updated.pddl'
    domain_file = data_dir / 'domain' / domain_file
    problem_directory = 'multitower'
    problem_number = 1
    problem_file = data_dir / problem_directory / f'problem{problem_number}.pddl'
    domain, problem = pddl_functions.parse(domain_file, problem_file)

    state = pddl_functions.PDDLState.from_initialstate(problem.initialstate)

    assert(state.predicate_holds(pddl_functions.Predicate('on-table', ['b7'])) is True)
    assert(state.predicate_holds(pddl_functions.Predicate('red', ['b7'])) is False)
    assert(state.predicate_holds(pddl_functions.Predicate('yellow', ['b7'])) is True)

    state.apply_effect(Predicate('on-table', ['b7'], op='not'))
    state.apply_effect(Predicate('red', ['b7']))
    state.apply_effect(Predicate('yellow', ['b7'], op='not'))

    assert(state.predicate_holds(pddl_functions.Predicate('on-table', ['b7'])) is False)
    assert(state.predicate_holds(pddl_functions.Predicate('red', ['b7'])) is True)
    assert(state.predicate_holds(pddl_functions.Predicate('yellow', ['b7'])) is False)

    assert(state.fexpressions[0].number == 0.0)
    state.apply_effect(Increase('blue', 't1', 1))
    assert (state.fexpressions[0].number == 1.0)
    assert(state.get_colour_count('blue', 't1') == 1)

def test_action():
    config = get_config()
    data_dir = Path(config['data_location'])
    domain_file = 'blocks-domain-updated.pddl'
    domain_file = data_dir / 'domain' / domain_file
    problem_directory = 'multitower'
    problem_number = 1
    problem_file = data_dir / problem_directory / f'problem{problem_number}.pddl'
    domain, problem = pddl_functions.parse(domain_file, problem_file)

    state = pddl_functions.PDDLState.from_initialstate(problem.initialstate)
    action = pddl_functions.Action.from_pddl(domain.actions[0])

    assert(action.preconditions_hold(state, ['b1', 'g1', 't1']) is True)
    assert (action.preconditions_hold(state, ['b1', 'g0', 't0']) is True)
    assert (action.preconditions_hold(state, ['b1', 'g1', 't0']) is False)
    assert (action.preconditions_hold(state, ['b1', 'b2', 't1']) is False)



def test_compare_increase():
    inc = Increase('red', 't1', 1)
    inc2 = Increase('blue', 't2', 2)

    cc1 = ColourCount('red', 't1', 0)
    cc2 = ColourCount('red', 't2', 0)
    cc3 = ColourCount('blue', 't2', 0)

    assert(inc.compare_colour_count(cc1) is True)
    assert (inc.compare_colour_count(cc2) is False)
    assert (inc.compare_colour_count(cc3) is False)

    assert(inc2.compare_colour_count(cc1) is False)
    assert (inc2.compare_colour_count(cc2) is False)
    assert (inc2.compare_colour_count(cc3) is True)


def test_apply_action():
    config = get_config()
    data_dir = Path(config['data_location'])
    domain_file = 'blocks-domain-updated.pddl'
    domain_file = data_dir / 'domain' / domain_file
    problem_directory = 'multitower'
    problem_number = 1
    problem_file = data_dir / problem_directory / f'problem{problem_number}.pddl'
    domain, problem = pddl_functions.parse(domain_file, problem_file)

    state = pddl_functions.PDDLState.from_initialstate(problem.initialstate)
    action = pddl_functions.Action.from_pddl(domain.actions[0])

    action.apply_action(state, ['b1', 'g1', 't1'])
    assert(state.predicate_holds(Predicate('on', ['b1', 'g1'])))

    action.apply_action(state, ['b0', 'g0', 't0'])
    assert (state.predicate_holds(Predicate('on', ['b0', 'g0'])))
    assert(state.get_colour_count('blue', 't0') == 1)


