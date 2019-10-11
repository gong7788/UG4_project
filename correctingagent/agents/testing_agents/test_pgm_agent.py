
import pytest
from skimage.color import hsv2rgb

from correctingagent.agents.PGMAgent import *
from correctingagent.world import RandomColoursWorld


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


