from functools import reduce

import pytest
from skimage.color import hsv2rgb

from correctingagent.agents.PGMAgent import *
from correctingagent.agents.teacher import ExtendedTeacherAgent
from correctingagent.world import RandomColoursWorld
from correctingagent.world.rules import ColourCountRule


def test_read_sentence():

    m = read_sentence("no, put red blocks on blue blocks")
    assert(m.o1[0] == 'red')
    assert(m.o2[0] == 'blue')
    assert(m.T == 'tower')

    m = read_sentence('no, now you cannot put b3 in the tower because you must put red blocks on blue blocks')

    assert(m.o1[0] == 'red')
    assert(m.o2[0] == 'blue')
    assert(m.T == 'table')

    m = read_sentence('no, that is not red again')

    assert(m.o1 == 'red')
    assert(m.T == 'partial.neg')

    m = read_sentence('no, that is wrong for the same reason')

    assert(m.T == 'same reason')


def test_colour_count_read_sentence():
    m = read_sentence("no, you cannot put more than 1 blue blocks in a tower")

    assert(m.o1 == 'blue')
    assert(m.o2 == 1)
    assert(m.T == 'colour count')

    number = 1
    colour_name = 'blue'
    c1 = 'blue'
    c2 = 'red'
    sentence = f"no, you cannot put more than {number} {colour_name} blocks in a tower"
    sentence = sentence + ' and ' +  f"you must put {c1} blocks on {c2} blocks"

    m = read_sentence(sentence)
    assert(m.o1 == colour_name)
    assert(m.o2 == number)
    assert(m.T == 'colour count+tower')
    m2 = m.o3

    assert(m2.o1[0] == c1)
    assert(m2.o2[0] == c2)
    assert(m2.T == 'tower')


def array_equal(array, reference):
    reference = np.array(hsv2rgb([[reference]])[0][0])
    return bool(np.all(np.array(array) - reference < 0.000001))


def test_get_relevant_data():
    w = RandomColoursWorld(problem_directory='testing', problem_number=1)

    agent = PGMCorrectingAgent(w)

    b0_data = [0.6225572692994489, 0.9962646551126118, 0.9268826891413054]
    b1_data = [0.8660805949460358, 0.9556451205968787, 0.8379154344341567]
    b2_data = [0.013002763048411668, 0.8465446783455671, 0.8656983732575542]

    data = agent.get_relevant_data(['b0', 'b1'], Message(None, ['red'], ['blue'], 'tower', None))

    assert(array_equal(data['F(b0)'], b0_data) is True)
    assert(array_equal(data['F(b1)'], b1_data) is True)
    assert(array_equal(data['F(b0)'], b1_data) is False)

    data = agent.get_relevant_data(['b0', 'b1'], Message(None, ['green'], ['orange'], 'table', 'b2'))

    assert(array_equal(data['F(b0)'], b0_data))
    assert(array_equal(data['F(b1)'], b1_data))
    assert(array_equal(data['F(b2)'], b2_data))
    assert(data['corr_0'] == 1)


def test_update_model():
    w = RandomColoursWorld(problem_directory='testing', problem_number=1)

    agent = PGMCorrectingAgent(w)

    b0_data = [0.6225572692994489, 0.9962646551126118, 0.9268826891413054]
    b1_data = [0.8660805949460358, 0.9556451205968787, 0.8379154344341567]
    b2_data = [0.013002763048411668, 0.8465446783455671, 0.8656983732575542]

    violations, data, message = agent.update_model('no, put red blocks on blue blocks', ['b0', 'b1'])

    assert('V_0(all x.(red(x) -> exists y. (blue(y) & on(x,y))))' in violations)
    assert('V_0(all y.(blue(y) -> exists x. (red(x) & on(x,y))))' in violations)

    assert(message.T == 'tower')
    assert(message.o1 == ['red'])

    assert(array_equal(data["F(b0)"], b0_data))

    violations, data, message = agent.update_model('no, now you cannot put b1 in the tower because you must put red blocks on blue blocks', ['b2', 'b0'])

    assert ('V_0(all x.(red(x) -> exists y. (blue(y) & on(x,y))))' in violations)
    assert ('V_0(all y.(blue(y) -> exists x. (red(x) & on(x,y))))' in violations)

    assert (message.T == 'table')
    assert (message.o1 == ['red'])

    assert (array_equal(data["F(b0)"], b0_data))
    assert (array_equal(data["F(b1)"], b1_data))
    assert (array_equal(data["F(b2)"], b2_data))
    assert(array_equal(data["F(b2)"], b1_data) is False)


def test_update_model():

    w = RandomColoursWorld('blocks-domain-updated.pddl', problem_directory='multitower', problem_number=1)

    w.update('put', ['b0', 't1', 'tower1'])
    w.update('put', ['b2', 'b0', 'tower1'])
    w.update('put', ['b5', 'b2', 'tower1'])

    agent = PGMCorrectingAgent(w)

    rule = ColourCountRule('blue', 1)

    b0_data = [0.34910434076796404, 0.9939351750475346, 0.9020967700201346]
    b1_data = [0.8872649368468363, 0.9880529065134138, 0.9322439031222336]
    b2_data = [0.15738804617607374, 0.9835023958587732, 0.9541478602571746]
    b5_data = [0.8805041887685969, 0.9819704146691624, 0.9679969438475052]

    violations, data, message = agent.update_model(f"no, you cannot put more than 1 blue blocks in a tower", ['b5', 'b2', 'tower1'])

    assert(f'V_0({rule})' in violations)

    assert(message.T == 'colour count')
    assert(message.o1 == 'blue')
    assert(message.o2 == 1)

    assert(array_equal(data["F(b0)"], b0_data))
    assert(array_equal(data["F(b5)"], b5_data))
    assert(array_equal((data["F(b2)"]), b2_data))
    assert(data["corr_0"] == 1)

def test_update_model_w_teacher():

    w = RandomColoursWorld('blocks-domain-updated.pddl', problem_directory='multitower', problem_number=1)

    teacher = ExtendedTeacherAgent()
    agent = PGMCorrectingAgent(w, teacher=teacher)

    w.update('put', ['b0', 't1', 'tower1'])
    w.update('put', ['b2', 'b0', 'tower1'])
    w.update('put', ['b5', 'b2', 'tower1'])

    rule = ColourCountRule('blue', 1)

    b0_data = [0.34910434076796404, 0.9939351750475346, 0.9020967700201346]
    b1_data = [0.8872649368468363, 0.9880529065134138, 0.9322439031222336]
    b2_data = [0.15738804617607374, 0.9835023958587732, 0.9541478602571746]
    b5_data = [0.8805041887685969, 0.9819704146691624, 0.9679969438475052]

    correction = teacher.correction(w)

    violations, data, message = agent.update_model(correction, ['b5', 'b2', 'tower1'])

    assert(f'V_0({rule})' in violations)

    assert(message.T == 'colour count')
    assert(message.o1 == 'blue')
    assert(message.o2 == 1)

    assert(array_equal(data["F(b0)"], b0_data))
    assert(array_equal(data["F(b5)"], b5_data))
    assert(array_equal((data["F(b2)"]), b2_data))
    assert(data["corr_0"] == 1)
    assert(data['blue(b5)'] == 1)


def test_get_correction_colour_count():

    w = RandomColoursWorld('blocks-domain-updated.pddl', problem_directory='multitower', problem_number=1)

    teacher = ExtendedTeacherAgent()
    agent = PGMCorrectingAgent(w, teacher=teacher)

    w.update('put', ['b0', 't1', 'tower1'])
    w.update('put', ['b2', 'b0', 'tower1'])
    w.update('put', ['b5', 'b2', 'tower1'])

    rule = ColourCountRule('blue', 1)

    b0_data = [0.34910434076796404, 0.9939351750475346, 0.9020967700201346]
    b1_data = [0.8872649368468363, 0.9880529065134138, 0.9322439031222336]
    b2_data = [0.15738804617607374, 0.9835023958587732, 0.9541478602571746]
    b5_data = [0.8805041887685969, 0.9819704146691624, 0.9679969438475052]

    correction = teacher.correction(w)

    agent.get_correction(correction, 'put', ['b5', 'b2', 'tower1'])

    test_b5_data_in_cm = any([array_equal(datum, b5_data) for datum in agent.colour_models['blue'].data])
    assert(test_b5_data_in_cm)
    rule_probs = agent.pgm_model.get_rule_probs()
    assert(rule_probs[rule] > 0.5)


def test_get_correction_colour_count2():

    w = RandomColoursWorld('blocks-domain-updated.pddl', problem_directory='multitower', problem_number=2)

    teacher = ExtendedTeacherAgent()
    agent = PGMCorrectingAgent(w, teacher=teacher)

    w.update('put', ['b0', 't1', 'tower1'])
    w.update('put', ['b2', 'b0', 'tower1'])
    w.update('put', ['b1', 'b2', 'tower1'])

    rule = ColourCountRule('blue', 1)

    b0_data = [0.34910434076796404, 0.9939351750475346, 0.9020967700201346]
    b1_data = [0.8872649368468363, 0.9880529065134138, 0.9322439031222336]
    b2_data = [0.15738804617607374, 0.9835023958587732, 0.9541478602571746]
    b5_data = [0.8805041887685969, 0.9819704146691624, 0.9679969438475052]

    correction = teacher.correction(w)
    assert (correction == "no, you cannot put more than 1 blue blocks in a tower and you must put blue blocks on red blocks")

    agent.get_correction(correction, 'put', ['b1', 'b2', 'tower1'])

    colour_predictions = agent.pgm_model.get_colour_predictions()
    assert(abs(colour_predictions['red(b1)'] - 1.0) < 0.0001)

    assert(array_equal(agent.get_colour_data(['b1'])['F(b1)'], b1_data))

    assert(agent.colour_models['red'].data != [])
    assert(len(agent.colour_models['red'].data) == 1)
    test_b1_data_in_cm = reduce(lambda x, y: x or y, [array_equal(agent.colour_models['red'].data[i], np.array(b1_data)) for i in range(len(agent.colour_models['red'].data))])
    test_b2_data_in_cm = reduce(lambda x, y: x or y, [array_equal(datum, b2_data) for datum in agent.colour_models['red'].data])
    test_b0_data_in_cm = reduce(lambda x, y: x or y, [array_equal(datum, np.array(b0_data)) for datum in agent.colour_models['red'].data])

    assert(np.all(np.array(hsv2rgb([[b1_data]])[0][0]) == agent.colour_models['red'].data[0]))
    assert(test_b0_data_in_cm is False)
    assert(test_b2_data_in_cm is False)
    assert(test_b1_data_in_cm)
    rule_probs = agent.pgm_model.get_rule_probs()
    assert(rule_probs[rule] > 0.5)


def test_model_update():
     w = RandomColoursWorld('blocks-domain-updated.pddl', problem_directory='multitower', problem_number=2)
