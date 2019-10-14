import pytest
from correctingagent.agents.teacher import *
from correctingagent.world import world


def test_tower_correction():
    w = world.PDDLWorld(problem_directory='testing', problem_number=1)
    teacher = TeacherAgent()

    w.update('put', ['b8', 't0'])  # pink
    w.update('put', ['b9', 'b8'])  # blue
    w.update('put', ['b6', 'b9'])  # pink
    w.update('put', ['b4', 'b6'])  # purple

    assert(teacher.correction(w) == "no, put blue blocks on pink blocks")


def test_table_correction():
    w = world.PDDLWorld(problem_directory='testing', problem_number=1)
    teacher = TeacherAgent()

    w.update('put', ['b8', 't0'])  # pink
    w.update('put', ['b9', 'b8'])  # blue
    w.update('put', ['b7', 'b9'])  # blue

    assert(teacher.correction(w).lower() == 'no, now you cannot put b1 in the tower because you must put blue blocks on pink blocks')


def test_table_correction_extended():
    w = world.PDDLWorld(problem_directory='testing', problem_number=1)
    teacher = ExtendedTeacherAgent()

    w.update('put', ['b8', 't0'])  # pink
    w.update('put', ['b9', 'b8'])  # blue
    w.update('put', ['b7', 'b9'])  # blue

    assert(teacher.correction(w).lower() == 'no, now you cannot put b1 in the tower because you must put blue blocks on pink blocks')

    w.back_track()  # remove blue
    w.update('put', ['b3', 'b9'])  # blue
    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True)
    possible_corrections = [language_output.lower() for language_output, correction_object in possible_corrections]

    assert('no, now you cannot put b1 in the tower because you must put blue blocks on pink blocks' in possible_corrections)
    assert('no, that is wrong for the same reason' in possible_corrections)
    assert(len(possible_corrections) == 2)


def test_tower_correction_extended():
    w = world.PDDLWorld(problem_directory='testing', problem_number=1)
    teacher = ExtendedTeacherAgent()

    w.update('put', ['b8', 't0'])  # pink
    w.update('put', ['b9', 'b8'])  # blue
    w.update('put', ['b6', 'b9'])  # pink
    w.update('put', ['b4', 'b6'])  # purple

    assert(teacher.correction(w) == "no, put blue blocks on pink blocks")

    w.back_track()  # remove purple
    w.update('put', ['b5', 'b6'])  # pink

    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True)
    possible_corrections = [language_output for language_output, correction_object in possible_corrections]
    assert("no, put blue blocks on pink blocks" in possible_corrections)
    assert("no, that is wrong for the same reason" in possible_corrections)
    assert("no, that is not blue again" in possible_corrections)
    assert(len(possible_corrections) == 3)


def test_tower_correction_extended2():
    w = world.PDDLWorld(problem_directory='testing', problem_number=2)
    teacher = ExtendedTeacherAgent()

    w.update('put', ['b8', 't0'])  # pink
    w.update('put', ['b9', 'b8'])  # blue
    w.update('put', ['b6', 'b9'])  # pink
    w.update('put', ['b4', 'b6'])  # purple

    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True)
    possible_corrections = [language_output.lower() for language_output, correction_object in possible_corrections]
    assert("no, put blue blocks on pink blocks" in possible_corrections)
    assert("no, now you cannot put b2 in the tower because you must put purple blocks on red blocks" in possible_corrections)
    assert(len(possible_corrections) == 2)

    w = world.PDDLWorld(problem_directory='testing', problem_number=2)
    teacher = ExtendedTeacherAgent()

    w.update('put', ['b8', 't0'])  # pink
    w.update('put', ['b9', 'b8'])  # blue
    w.update('put', ['b6', 'b9'])  # pink
    w.update('put', ['b5', 'b6'])  # pink

    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True)
    possible_corrections = [language_output.lower() for language_output, correction_object in possible_corrections]
    assert ("no, put blue blocks on pink blocks" in possible_corrections)
    assert(len(possible_corrections) == 1)
    #
    w.back_track()  # remove purple
    w.update('put', ['b7', 'b6'])  # blue
    w.update('put', ['b4', 'b7'])  # purple

    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True)
    possible_corrections = [language_output.lower() for language_output, correction_object in possible_corrections]
    assert("no, now you cannot put b2 in the tower because you must put purple blocks on red blocks" in possible_corrections)
    assert(len(possible_corrections) == 1)

    w.back_track()
    w.update('put', ['b5', 'b7'])  # pink
    w.update('put', ['b2', 'b5'])

    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True)
    possible_corrections = [language_output.lower() for language_output, correction_object in possible_corrections]
    assert ("no, put blue blocks on pink blocks" in possible_corrections)
    assert ("no, that is not blue again" in possible_corrections)
    assert (len(possible_corrections) == 2)


def test_failure():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=1)

    w.update('put', ['b0', 't0', 'tower0'])
    w.update('put', ['b5', 'b0', 'tower0'])

    assert(w.test_failure())


def test_rule_violated():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=1)

    w.update('put', ['b0', 't0', 'tower0'])  # blue
    w.update('put', ['b5', 'b0', 'tower0'])  # blue

    teacher = ExtendedTeacherAgent()

    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True)
    possible_corrections = [s for s, c in possible_corrections]

    assert("no, you cannot put more than 1 blue blocks in a tower" in possible_corrections)


def test_rule_violated_table():

    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl', problem_directory='multitower', problem_number=3)

    w.update('put', ['b4', 't0', 'tower0'])  # red
    w.update('put', ['b0', 'b4', 'tower0'])  # blue
    w.update('put', ['b1', 'b0', 'tower0'])  # red

    teacher = ExtendedTeacherAgent()

    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True)
    possible_corrections = [s for s, c in possible_corrections]

    assert ('no, you cannot put more than 1 blue blocks in a tower and you must put blue blocks on red blocks' in possible_corrections)


