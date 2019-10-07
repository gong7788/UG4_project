import pytest

from correctingagent.pddl.pddl_functions import Predicate
from correctingagent.world import world


def test_update():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl', problem_directory="multitower", problem_number=1)

    assert(w.state.predicate_holds('on-table', ['b0']))
    w.update('put', ['b0', 't1', 'tower1'])
    assert(w.state.predicate_holds('on', ['b0', 't1']))
    w.back_track()
    assert(w.state.predicate_holds('on-table', ['b0']))


def test_world():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl', problem_directory="multitower", problem_number=1)

    plan = w.find_plan()

    for action, args in plan:
        assert(w.test_failure() is False)
        w.update(action, args)

    assert(w.test_success())

    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl', problem_directory="multitower", problem_number=1)

    w.update('put', ['b0', 't1', 'tower1'])
    w.update('put', ['b5', 'b0', 'tower1'])

    assert(w.test_failure())

    w.back_track()
    plan = w.find_plan()

    for action, args in plan:

        w.update(action, args)

    assert(w.test_success())


def test_world2():
    w = world.PDDLWorld(domain_file='blocks-domain.pddl', problem_directory="tworules", problem_number=1)

    plan = w.find_plan()

    for action, args in plan:
        assert(w.test_failure() is False)
        w.update(action, args)

    assert(w.test_success())


def test_world3():
    w = world.RandomColoursWorld(domain_file='blocks-domain-updated.pddl', problem_directory="multitower", problem_number=1)

    plan = w.find_plan()

    for action, args in plan:
        assert(w.test_failure() is False)
        w.update(action, args)

    assert(w.test_success())

    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl', problem_directory="multitower", problem_number=1)

    w.update('put', ['b0', 't1', 'tower1'])
    w.update('put', ['b5', 'b0', 'tower1'])

    assert(w.test_failure())

    w.back_track()
    plan = w.find_plan()

    for action, args in plan:

        w.update(action, args)

    assert(w.test_success())
