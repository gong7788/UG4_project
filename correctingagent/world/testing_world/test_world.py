import pytest

from correctingagent.pddl.pddl_functions import Predicate
from correctingagent.world import world


def test_update():
    w = world.PDDLWorld(problem_directory="multitower", problem_number=1)

    assert(w.state._predicate_holds(Predicate('on-table', ['b0'])))

