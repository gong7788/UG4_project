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

    assert(teacher.correction(w, ['b4', 'b6']) == "no, put blue blocks on pink blocks")


def test_table_correction():
    w = world.PDDLWorld(problem_directory='testing', problem_number=1)
    teacher = TeacherAgent()

    w.update('put', ['b8', 't0'])  # pink
    w.update('put', ['b9', 'b8'])  # blue
    w.update('put', ['b7', 'b9'])  # blue

    assert(teacher.correction(w, ['b7', 'b9']).lower() == 'no, now you cannot put b1 in the tower because you must put blue blocks on pink blocks')


def test_table_correction_extended():
    w = world.PDDLWorld(problem_directory='testing', problem_number=1)
    teacher = ExtendedTeacherAgent()

    w.update('put', ['b8', 't0'])  # pink
    w.update('put', ['b9', 'b8'])  # blue
    w.update('put', ['b7', 'b9'])  # blue

    assert(teacher.correction(w, ['b7', 'b9']).lower() == 'no, now you cannot put b1 in the tower because you must put blue blocks on pink blocks')

    w.back_track()  # remove blue
    w.update('put', ['b3', 'b9'])  # blue
    correction, possible_corrections = teacher.correction(w, ['b3', 'b9'], return_possible_corrections=True)
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

    assert(teacher.correction(w, ['b4', 'b6']) == "no, put blue blocks on pink blocks")

    w.back_track()  # remove purple
    w.update('put', ['b5', 'b6'])  # pink

    correction, possible_corrections = teacher.correction(w, ['b5', 'b6'], return_possible_corrections=True)
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

    correction, possible_corrections = teacher.correction(w, args=['b4', 'b6'], return_possible_corrections=True)
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

    correction, possible_corrections = teacher.correction(w, ['b5', 'b6'], return_possible_corrections=True)
    possible_corrections = [language_output.lower() for language_output, correction_object in possible_corrections]
    assert ("no, put blue blocks on pink blocks" in possible_corrections)
    assert(len(possible_corrections) == 1)
    #
    w.back_track()  # remove purple
    w.update('put', ['b7', 'b6'])  # blue
    w.update('put', ['b4', 'b7'])  # purple

    correction, possible_corrections = teacher.correction(w, ['b4', 'b7'], return_possible_corrections=True)
    possible_corrections = [language_output.lower() for language_output, correction_object in possible_corrections]
    assert("no, now you cannot put b2 in the tower because you must put purple blocks on red blocks" in possible_corrections)
    assert(len(possible_corrections) == 1)

    w.back_track()
    w.update('put', ['b5', 'b7'])  # pink
    w.update('put', ['b2', 'b5'])

    correction, possible_corrections = teacher.correction(w, ['b2', 'b5'], return_possible_corrections=True)
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

    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True, args=['b5', 'b0', 'tower0'])
    possible_corrections = [s for s, c in possible_corrections]

    assert("no, you cannot put more than 1 blue blocks in a tower" in possible_corrections)


def test_rule_violated_table():

    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl', problem_directory='multitower', problem_number=3)

    w.update('put', ['b4', 't0', 'tower0'])  # red
    w.update('put', ['b0', 'b4', 'tower0'])  # blue
    w.update('put', ['b1', 'b0', 'tower0'])  # red

    teacher = ExtendedTeacherAgent()

    correction, possible_corrections = teacher.correction(w, args=['b1', 'b0', 'tower0'], return_possible_corrections=True)
    possible_corrections = [s for s, c in possible_corrections]

    assert ('no, you cannot put more than 1 blue blocks in a tower and you must put blue blocks on red blocks' in possible_corrections)


def test_rule_violated_table_2_towers():

    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl', problem_directory='multitower', problem_number=3)

    w.update('put', ['b6', 't1', 'tower1'])  # orange
    w.update('put', ['b4', 't0', 'tower0'])  # red
    w.update('put', ['b0', 'b4', 'tower0'])  # blue
    w.update('put', ['b1', 'b0', 'tower0'])  # red

    teacher = ExtendedTeacherAgent()

    correction, possible_corrections = teacher.correction(w, args=['b1', 'b0', 'tower0'], return_possible_corrections=True)
    possible_corrections = [s for s, c in possible_corrections]

    assert ('no, you cannot put more than 1 blue blocks in a tower and you must put blue blocks on red blocks' in possible_corrections)


def test_rule_violated_table_2_towers_2():

    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl', problem_directory='multitower', problem_number=3)

    w.update('put', ['b6', 't0', 'tower0'])  # orange
    w.update('put', ['b4', 't1', 'tower1'])  # red
    w.update('put', ['b0', 'b4', 'tower1'])  # blue
    w.update('put', ['b1', 'b0', 'tower1'])  # red

    teacher = ExtendedTeacherAgent()

    correction, possible_corrections = teacher.correction(w, args=['b1', 'b0', 'tower1'], return_possible_corrections=True)
    possible_corrections = [s for s, c in possible_corrections]

    assert ('no, you cannot put more than 1 blue blocks in a tower and you must put blue blocks on red blocks' in possible_corrections)


def test_rule_violated_2_towers():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=1)

    w.update('put', ['b6', 't1', 'tower1'])  # tower
    w.update('put', ['b0', 't0', 'tower0'])  # blue
    w.update('put', ['b5', 'b0', 'tower0'])  # blue

    teacher = ExtendedTeacherAgent()

    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True, args=['b5', 'b0', 'tower0'])
    possible_corrections = [s for s, c in possible_corrections]

    assert("no, you cannot put more than 1 blue blocks in a tower" in possible_corrections)


def test_red_on_blue_violated_2_towers():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=3)

    w.update('put', ['b1', 't1', 'tower1'])  # red
    w.update('put', ['b0', 't0', 'tower0'])  # blue

    assert(w.test_failure())

    teacher = ExtendedTeacherAgent()

    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True,
                                                          args=['b0', 't0', 'tower0'])
    possible_corrections = [s for s, c in possible_corrections]

    assert ("No, now you cannot put b4 in the tower because you must put blue blocks on red blocks" in possible_corrections)


def test_red_on_blue_violated_2_towers2():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=4)

    w.update('put', ['b1', 't1', 'tower1'])  # red
    w.update('put', ['b5', 'b1', 'tower1'])  # blue
    w.update('put', ['b4', 'b5', 'tower1'])  # red
    w.update('put', ['b0', 't0', 'tower0'])  # blue

    r = RedOnBlueRule('blue', 'red', 2)
    r2 = ColourCountRule('blue', 2)
    assert(w.test_failure())

    assert(r.check_table_violation(w.state, [r2], 'tower0'))

    teacher = TeacherAgent()

    correction = teacher.correction(w, args=['b0', 't0', 'tower0'])

    assert ("No, now you cannot put b4 in the tower because you must put blue blocks on red blocks" == correction)


def test_colour_count_joint_violation():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=3)

    w.update('put', ['b6', 't1', 'tower1'])  # orange
    w.update('put', ['b1', 't0', 'tower0'])  # red
    w.update('put', ['b5', 'b1', 'tower0'])  # blue
    w.update('put', ['b4', 'b5', 'tower0'])  # red

    assert(w.test_failure())

    teacher = TeacherAgent()

    correction = teacher.correction(w, args=['b4', 'b5', 'tower0'])

    assert( "no, you cannot put more than 1 blue blocks in a tower and you must put blue blocks on red blocks" == correction)


def test_colour_count_joint_violation2():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=5)

    w.update('put', ['b6', 't1', 'tower1'])  # orange
    w.update('put', ['b3', 'b6', 'tower1'])  # yellow
    w.update('put', ['b7', 'b3', 'tower1'])  # yellow
    w.update('put', ['b8', 'b7', 'tower1'])  # purple
    w.update('put', ['b9', 'b8', 'tower1'])  # purple
    w.update('put', ['b2', 'b9', 'tower1'])  # yellow
    w.update('put', ['b1', 't0', 'tower0'])  # red
    w.update('put', ['b5', 'b1', 'tower0'])  # blue
    w.update('put', ['b4', 'b5', 'tower0'])  # red

    assert(w.test_failure())

    teacher = TeacherAgent()

    correction = teacher.correction(w, args=['b4', 'b5', 'tower0'])

    assert( "no, you cannot put more than 1 blue blocks in a tower and you must put blue blocks on red blocks" == correction)

def test_colour_count_joint_violation3():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=6)

    w.update('put', ['b8', 't1', 'tower1'])  # purple
    w.update('put', ['b9', 'b8', 'tower1'])  # purple
    w.update('put', ['b4', 'b9', 'tower1'])  # green
    w.update('put', ['b1', 'b4', 'tower1'])  # red
    w.update('put', ['b2', 't0', 'tower0'])  # red
    w.update('put', ['b3', 'b2', 'tower0'])  # blue
    w.update('put', ['b5', 'b3', 'tower0'])  # blue
    w.update('put', ['b6', 'b5', 'tower0'])  # red

    assert(w.test_failure())

    teacher = TeacherAgent()

    correction = teacher.correction(w, args=['b4', 'b5', 'tower0'])

    assert( "no, you cannot put more than 2 blue blocks in a tower and you must put blue blocks on red blocks" == correction)

def test_colour_count_joint_violation3():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=6)

    w.update('put', ['b8', 't1', 'tower1'])  # purple
    w.update('put', ['b9', 'b8', 'tower1'])  # purple
    w.update('put', ['b4', 'b9', 'tower1'])  # green
    w.update('put', ['b2', 't0', 'tower0'])  # red
    w.update('put', ['b3', 'b2', 'tower0'])  # blue
    w.update('put', ['b5', 'b3', 'tower0'])  # blue
    w.update('put', ['b6', 'b5', 'tower0'])  # red

    assert(w.test_failure())

    teacher = TeacherAgent()

    answer1 = teacher.answer_question('Is the top object red?', w, 'tower0')
    answer2 = teacher.answer_question('Is the top object green?', w, 'tower1')
    answer3 = teacher.answer_question('Is the top object red?', w, 'tower1')

    assert("yes" == answer1)
    assert("yes" == answer2)
    assert("no" == answer3)


def test_red_on_blue_directly_on_table():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=7)

    w.update('put', ['b9', 't1', 'tower1'])  # green

    assert(w.test_failure())

    teacher = TeacherAgent()

    correction = teacher.correction(w, args=['b9', 't1', 'tower1'])

    assert('no, put green blocks on yellow blocks' == correction)

    extended_teacher = ExtendedTeacherAgent()

    correction = extended_teacher.correction(w, args=['b9', 't1', 'tower1'])
    assert('no, put green blocks on yellow blocks' == correction)

    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=7)

    w.update('put', ['b9', 't0', 'tower0'])  # green

    assert(w.test_failure())

    teacher = TeacherAgent()

    correction = teacher.correction(w, args=['b9', 't0', 'tower0'])

    assert('no, put green blocks on yellow blocks' == correction)

    extended_teacher = ExtendedTeacherAgent()

    correction = extended_teacher.correction(w, args=['b9', 't0', 'tower0'])
    assert('no, put green blocks on yellow blocks' == correction)


def test_faulty_teacher():
    # TEST THAT THE INDIRECT VIOLATION IS SKIPPED
    #
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=3)

    w.update('put', ['b1', 't1', 'tower1'])  # red
    w.update('put', ['b0', 't0', 'tower0'])  # blue

    assert(w.test_failure())

    teacher = ExtendedTeacherAgent()
    faulty_teacher = FaultyTeacherAgent(recall_failure_prob=1.0)

    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True,
                                                          args=['b0', 't0', 'tower0'])
    possible_corrections = [s for s, c in possible_corrections]

    assert ("no, now you cannot put b4 in the tower because you must put blue blocks on red blocks" in possible_corrections)
    assert ("" == faulty_teacher.correction(w, args=['b0', 't0', 'tower0']))


def test_faulty_teacher_direct():

    # TEST THAT A DIRECT VIOLATION IS SKIPPED
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=3)

    w.update('put', ['b1', 't1', 'tower1'])  # red
    w.update('put', ['b2', 'b1', 'tower1'])  # orange

    #w.update('put', [])

    assert(w.test_failure())

    teacher = ExtendedTeacherAgent()
    faulty_teacher = FaultyTeacherAgent(recall_failure_prob=0.0, p_miss_direct=1.0)

    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True,
                                                          args=['b2', 'b1', 'tower1'])
    possible_corrections = [s for s, c in possible_corrections]

    assert ("no, put blue blocks on red blocks" in possible_corrections)
    assert ("" == faulty_teacher.correction(w, args=['b2', 'b1', 'tower1']))

    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=3)


    # TEST THAT A COMBINED r1/r2 r3 INDIRECT VIOLATION IS SKIPPED
    w.update('put', ['b1', 't1', 'tower1'])  # red
    w.update('put', ['b0', 'b1', 'tower1'])  # blue

    w.update('put', ['b4', 'b0', 'tower1'])  # red
    #w.update('put', ['b5', 'b4', 'tower1'])

    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True,
                                                          args=['b2', 'b1', 'tower1'])
    possible_corrections = [s for s, c in possible_corrections]

    assert ("no, you cannot put more than 1 blue blocks in a tower and you must put blue blocks on red blocks" in possible_corrections)
    assert ("no, you cannot put more than 1 blue blocks in a tower and you must put blue blocks on red blocks" == faulty_teacher.correction(w, args=['b3', 't1', 'tower1']))

    faulty_teacher = FaultyTeacherAgent(recall_failure_prob=1.0, p_miss_direct=1.0)

    assert ("" == faulty_teacher.correction(w, args=['b3', 't1', 'tower1']))

    # TEST THAT A r3 DIRECT VIOLATION IS SKIPPED
    w.update('put', ['b5', 'b4', 'tower1'])

    correction, possible_corrections = teacher.correction(w, return_possible_corrections=True,
                                                          args=['b2', 'b1', 'tower1'])
    possible_corrections = [s for s, c in possible_corrections]

    assert ("no, you cannot put more than 1 blue blocks in a tower" in possible_corrections)
    assert ("" == faulty_teacher.correction(
            w, args=['b5', 'b4', 'tower1']))


    faulty_teacher.recover_prob = 1
    assert("no, you cannot put more than 1 blue blocks in a tower and you must put blue blocks on red blocks" in faulty_teacher.skipped_indirect_corrections)
    assert ("no, you cannot put more than 1 blue blocks in a tower and you must put blue blocks on red blocks" == faulty_teacher.correction(w, args=['b6', 'b5', 'tower1']))

def test_faulty_teacher_recovery_at_end():
    w = world.PDDLWorld(domain_file='blocks-domain-updated.pddl',
                        problem_directory="multitower", problem_number=3)

    w.update('put', ['b1', 't1', 'tower1'])  # red
    w.update('put', ['b2', 'b1', 'tower1'])  # orange
    w.update('put', ['b0', 'b2', 'tower1'])  # blue
    w.update('put', ['b4', 'b0', 'tower1'])  # red
    w.update('put', ['b5', 'b4', 'tower1'])  # blue
    w.update('put', ['b6', 'b5', 'tower1'])  #
    w.update('put', ['b7', 'b6', 'tower1'])  #
    w.update('put', ['b8', 'b7', 'tower1'])  #
    w.update('put', ['b9', 'b8', 'tower1'])  #
    w.update('put', ['b3', 'b9', 'tower1'])  #

    assert(w.objects_not_in_tower() == [])
    assert(len(w.objects_not_in_tower()) == 0)

    faulty_teacher = FaultyTeacherAgent()

    assert(w.get_objects_in_tower('tower1') == ['t1', 'b1', 'b2', 'b0', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b3'])

    assert("sorry, b0 and b1 are wrong because you must put blue blocks on red blocks" == faulty_teacher.correction(w, ['b3', 'b9', 'tower1']))
