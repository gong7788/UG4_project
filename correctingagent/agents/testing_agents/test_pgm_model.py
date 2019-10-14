


import pytest
from correctingagent.agents.PGMAgent import *
from correctingagent.world import world
from correctingagent.world.rules import ColourCountRule


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


def test_colour_count_CPD_generation():
    pgm_model = PGMModel()
    time = 0
    rule = ColourCountRule('blue', 1)

    cm = KDEColourModel('blue')

    violations = pgm_model.add_colour_count_correction(rule, cm, ['b1', 'b2'], time)

    pgm_model.observe({'F(b1)': [1, 1, 1], 'F(b2)': [0, 0, 0], f'corr_{time}': 1})

    q = pgm_model.query(['blue(b1)', 'blue(b2)', violations[0]])

    assert(q[violations[0]] == 1.0)
    assert(q["blue(b1)"] == 1.0)
    assert(q["blue(b2)"] == 1.0)

def test_colour_count_CPD_generation2():
    pgm_model = PGMModel()
    time = 0
    rule = ColourCountRule('blue', 1)

    cm = KDEColourModel('blue')

    violations = pgm_model.add_colour_count_correction(rule, cm, ['b1', 'b2', 'b3'], time)

    pgm_model.observe({'F(b1)': [1, 1, 1], 'F(b2)': [0, 0, 0], 'F(b3)': [0, 0, 0], f'corr_{time}': 1})

    q = pgm_model.query(['blue(b1)', 'blue(b2)', 'blue(b3)', violations[0]])

    assert (q[violations[0]] == 1.0)
    assert (q["blue(b1)"] == 2/3)
    assert (q["blue(b2)"] == 2/3)
    assert (q["blue(b3)"] == 2/3)

def test_colour_count_CPD_generation2():
    pgm_model = PGMModel()
    time = 0
    rule = ColourCountRule('blue', 1)

    cm = KDEColourModel('blue')

    objects = ['b1', 'b2', 'b3', 'b4', 'b5']

    violations = pgm_model.add_colour_count_correction(rule, cm, objects, time)

    data = {f'corr_{time}':1}
    for obj in objects:
        data[f'F({obj})'] = [1,1,1]

    pgm_model.observe(data)

    colours = [f'blue({obj})' for obj in objects]

    evidence = colours + violations
    q = pgm_model.query(evidence)

    assert (abs(q[violations[0]] -1.0) < 0.001)

    for colour in colours:
        assert(abs(q[colour] - 2/len(objects)) < 0.000001)

def test_colour_count_CPD_generation3():
    pgm_model = PGMModel()
    time = 0
    rule = ColourCountRule('blue', 1)

    cm = KDEColourModel('blue')

    objects = [f"b{o}" for o in range(10)]

    violations = pgm_model.add_colour_count_correction(rule, cm, objects, time)

    data = {f'corr_{time}':1}
    for obj in objects:
        data[f'F({obj})'] = [1,1,1]

    pgm_model.observe(data)

    colours = [f'blue({obj})' for obj in objects]

    evidence = colours + violations
    q = pgm_model.query(evidence)

    assert (abs(q[violations[0]] -1.0) < 0.001)

    for colour in colours:
        assert(abs(q[colour] - 2/len(objects)) < 0.000001)
