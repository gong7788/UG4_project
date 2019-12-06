


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
    colour_count = ColourCountRule('blue', 1)
    red_on_blue_options = Rule.generate_red_on_blue_options('blue', 'red')

    blue_cm = KDEColourModel('blue')
    red_cm = KDEColourModel('red')

    objects = [f"b{o}" for o in range(10)]
    objects_in_tower = objects[:5]
    top_object = objects[4]

    violations = pgm_model.add_cc_and_rob(colour_count, red_on_blue_options, blue_cm, red_cm, objects_in_tower, top_object, time)

    data = {f'corr_{time}': 1}
    for obj in objects:
        data[f'F({obj})'] = [1, 1, 1]

    pgm_model.observe(data)

    colours_in_tower = [f'blue({obj})' for obj in objects_in_tower[:-1]] + [f'red({top_object})']

    evidence = colours_in_tower + violations + red_on_blue_options + [colour_count]
    q = pgm_model.query(evidence)

    assert (abs(q[violations[0]] - 2/3) < 0.001)
    assert(abs(q[violations[1]] - 2/3) < 0.0001)

    assert(abs(q[colour_count] - 1.0) <  0.00001)
    assert(abs(q[red_on_blue_options[0]] - 2/3) < 0.00001)

    assert(pgm_model.colours['red'] is not None)
    assert(abs(q[colours_in_tower[-1]] - 1.0) < 0.00001)
    for colour in colours_in_tower[:-1]:
        assert(abs(q[colour] - 1/(len(colours_in_tower) - 1) < 0.0001))


def test_equals_cpd():
    pgm_model = PGMModel()

    r1, r2 = Rule.generate_red_on_blue_options('blue', 'red')
    v1 = f"V_1({r1})"
    v2 = f"V_1({r2})"
    v3 = f"V_2({r1})"
    v4 = f"V_2({r2})"

    pgm_model.add_same_reason([v1, v2], [v3, v4])

    pgm_model.infer()

    q = pgm_model.query([v1, v2, v3, v4])


#
# def test_joint_colour_count_CPD_generation1():
#     pgm_model = PGMModel()
#     time = 0
#     rule = ColourCountRule('blue', 1)
#
#     cm = KDEColourModel('blue')
#
#     objects = [f"b{o}" for o in range(10)]
#
#     violations = pgm_model.add_colour_count_correction(rule, cm, objects, time)
#
#     data = {f'corr_{time}':1}
#     for obj in objects:
#         data[f'F({obj})'] = [1,1,1]
#
#     pgm_model.observe(data)
#
#     colours = [f'blue({obj})' for obj in objects]
#
#     evidence = colours + violations
#     q = pgm_model.query(evidence)
#
#     assert (abs(q[violations[0]] -1.0) < 0.001)
#
#     for colour in colours:
#         assert(abs(q[colour] - 2/len(objects)) < 0.000001)
#
