
import pytest
from correctingagent.agents.agents import NoLanguageAgent
from correctingagent.agents.teacher import TeacherAgent, ExtendedTeacherAgent
from correctingagent.models.prob_model import KDEColourModel
from correctingagent.world import world
import numpy as np

def test_nolanguage_agent():

    w = world.PDDLWorld(problem_directory='testing', problem_number=1)
    teacher = TeacherAgent()

    agent = NoLanguageAgent(w, teacher=teacher)

    plan = agent.plan()
    assert(len(plan) == 10)

    w.update('put', ['b8', 't0'])  # pink
    w.update('put', ['b9', 'b8'])  # blue
    w.update('put', ['b6', 'b9'])  # pink
    w.update('put', ['b4', 'b6'])  # purple

    correction = teacher.correction(w, ['b4', 'b6'])

    agent.get_correction(correction, 'put', ['b4', 'b6'])

    plan = agent.plan()

    assert(len(plan) == 7)

def test_find_matching_cm():
    w = world.PDDLWorld(problem_directory='testing', problem_number=1)
    teacher = TeacherAgent()

    agent = NoLanguageAgent(w, teacher=teacher)

    agent.colour_models['red'] = KDEColourModel('red', data=np.array([[0.1034, 0.989, 0.546]]), weights=[1.])

    assert(agent.find_matching_cm(np.array([0.1034, 0.989, 0.546]))[0] == 'red')
    assert(agent.find_matching_cm(np.array([0., 0., 0.])) is None)
    assert(agent.find_matching_cm(np.array([0.10341, 0.9889, 0.5460101010101]))[0] == 'red')


def test_nolanguage_agent2():
    w = world.PDDLWorld(problem_directory='testing', problem_number=1)
    teacher = TeacherAgent()

    agent = NoLanguageAgent(w, teacher=teacher)

    w.update('put', ['b8', 't0'])  # pink
    w.update('put', ['b9', 'b8'])  # blue
    w.update('put', ['b7', 'b9'])  # blue

    correction = teacher.correction(w, ['b7', 'b9'])

    agent.get_correction(correction, 'put', ['b7', 'b9'])

    plan = agent.plan()
    assert(len(plan) == 8)


def test_nolanguage_agent3():
    teacher = TeacherAgent()
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=1)

    agent = NoLanguageAgent(w, teacher=teacher)

    agent.plan()

    w.update('put', ['b0', 't0', 'tower0'])  # blue
    w.update('put', ['b5', 'b0', 'tower0'])  # blue

    correction = teacher.correction(w, args=['b5', 'b0', 'tower0'])

    agent.get_correction(correction, 'put', ['b5', 'b0', 'tower0'])

    agent.plan()


def test_nolanguage_agent4():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl', problem_directory='multitower', problem_number=3)
    teacher = ExtendedTeacherAgent()
    agent = NoLanguageAgent(w, teacher=teacher)

    w.update('put', ['b4', 't0', 'tower0'])  # red
    w.update('put', ['b0', 'b4', 'tower0'])  # blue
    w.update('put', ['b1', 'b0', 'tower0'])  # red

    correction = teacher.correction(w, args=['b1', 'b0', 'tower0'])

    agent.get_correction(correction, 'put', ['b1', 'b0', 'tower0'])
    agent.plan()


def test_nolanguage_agent5():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl', problem_directory='multitower', problem_number=3)
    teacher = ExtendedTeacherAgent()
    agent = NoLanguageAgent(w, teacher=teacher)

    w.update('put', ['b4', 't0', 'tower0'])  # red
    w.update('put', ['b0', 'b4', 'tower0'])  # blue
    w.back_track()
    w.update('put', ['b0', 'b4', 'tower0'])  # blue
    w.update('put', ['b5', 'b0', 'tower0'])  # blue

    correction = teacher.correction(w, args=['b5', 'b0', 'tower0'])
    agent.get_correction(correction, 'put', ['b1', 'b0', 'tower0'])

    agent.plan()

    # w.back_track()

    w.update('put', ['b1', 'b0', 'tower0'])  # red



    correction = teacher.correction(w, args=['b1', 'b0', 'tower0'])

    agent.get_correction(correction, 'put', ['b1', 'b0', 'tower0'])
    agent.plan()
