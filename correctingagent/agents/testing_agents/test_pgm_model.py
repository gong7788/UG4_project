


import pytest
from correctingagent.agents.PGMAgent import *
from correctingagent.world import world


def test_build_pgm_model():

    w = world.PDDLWorld(problem_directory='testing', problem_number=1)

    pgm_agent = PGMCorrectingAgent(w)

    message = Message(rel='on', o1=['orange'], o2=['purple'], T='tower', o3=None)

    violations = pgm_agent.build_pgm_model(message, ['b1', 'b3'])

    pgm_model = pgm_agent.pgm_model
    pgm_model.observe({'F(b1)':[1,1,1], 'F(b3)':[0,0,0], f'corr_{pgm_agent.time}':1})
    q = pgm_model.query(violations, [1, 1])
    assert(q[violations[0]] == 0.5)
    assert(q[violations[1]] == 0.5)

