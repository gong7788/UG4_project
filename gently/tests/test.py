import pytest


from gently.environment import ConceptGroup, RuleGroup


@pytest.fixture
def context_setup():
    from gently.environment import Context

    context1 = Context(["red", "heart"], [])
    context2 = Context(["green", "square"], [])
    context3 = Context(["blue", "circle"], [])
    context4 = Context(["red", "green"], [])
    context5 = Context(["green", "heart"], [])

    return context1, context2, context3, context4, context5


@pytest.fixture
def rule_setup():
    from gently.environment import Rule
    r1 = Rule([["red", "heart"], ["green", "square"]], [["right", "slowly"]])
    r2 = Rule([["blue", "circle"]], [["quickly"]])
    r3 = Rule([['green', 'heart']], [["left", "slowly"]])

    return r1, r2, r3

# r3 = Rule([['green', 'heart']], [["left", "slowly"]])
# context5 = Context(["green", "heart"], [])
@pytest.fixture
def agent(rule_setup):
    from gently.agent import GentlyAgent

    r1, r2, r3 = rule_setup
    agent = GentlyAgent()
    agent.add_rule(r1)
    agent.add_rule(r2)
    return agent


@pytest.fixture
def teacher_concepts():
    from gently.teacher import TeacherConcept

    right = TeacherConcept("right", "curviness", 0.3, 10)
    left = TeacherConcept("left", "curviness", -10, -0.3)
    middle = TeacherConcept("middle", "curviness", -0.29, 0.29)

    slow = TeacherConcept("slowly", "speed", 0.01, 0.45)
    fast = TeacherConcept("quickly", "speed", 0.6, 1.5)

    gently = TeacherConcept("gently", "energy", 0.0, 0.45)
    firmly = TeacherConcept("firmly", "energy", 0.65, 1.0)

    return right, left, middle, slow, fast, gently, firmly


@pytest.fixture
def curve_setup():
    from gently.BezierCurves import BehaviourCurve
    curve1 = BehaviourCurve(0.3, 0.5, 0.5)
    curve2 = BehaviourCurve(-0.3, 0.5, 0.5)

    curve3 = BehaviourCurve(0.0, 0.25, 0.5)
    curve4 = BehaviourCurve(0.0, 0.6, 0.5)

    curve5 = BehaviourCurve(0.0, 0.5, 0.45)
    curve6 = BehaviourCurve(0.0, 0.5, 0.75)

    return curve1, curve2, curve3, curve4, curve5, curve6

@pytest.fixture
def teacher(rule_setup, teacher_concepts):
    from gently.teacher import TeacherAgent

    right, left, middle, slow, fast, gently, firmly = teacher_concepts
    r1, r2, r3 = rule_setup
    teacher = TeacherAgent([r1, r2], [right, left, middle, slow, fast, gently, firmly])
    return teacher


@pytest.fixture
def teacher2(rule_setup, teacher_concepts):
    from gently.teacher import TeacherAgent

    right, left, middle, slow, fast, gently, firmly = teacher_concepts
    r1, r2, r3 = rule_setup
    teacher = TeacherAgent([r1, r2, r3], [right, left, middle, slow, fast, gently, firmly])
    return teacher

def test_rules(context_setup, rule_setup):
    context1, context2, context3, context4, _ = context_setup
    r1, _, r3 = rule_setup
    assert (r1.evaluate(context1) is True)
    assert (r1.evaluate(context2) is True)
    assert (r1.evaluate(context3) is False)
    assert (r1.evaluate(context4) is False)


def test_concept_group():
    assert (ConceptGroup([["right", "slowly"]]) == ConceptGroup([["right", "slowly"]]))
    assert (ConceptGroup([["right", "slowly"]]) == [["right", "slowly"]])


def test_rule_group(context_setup, rule_setup):
    context1, context2, context3, context4, _ = context_setup
    r1, r2, r3 = rule_setup
    rule_group = RuleGroup([r1, r2])
    assert (list(rule_group.get_behaviour_concepts(context1)) == [[["right", "slowly"]]])
    assert (list(rule_group.get_behaviour_concepts(context4)) == [])
    assert (list(rule_group.get_behaviour_concepts(context3)) == [[["quickly"]]])


def test_generate_behaviour_runs(agent, context_setup):
    context1, context2, context3, context4, _ = context_setup
    agent.generate_behaviour(context1)
    agent.generate_behaviour(context2)
    agent.generate_behaviour(context3)
    agent.generate_behaviour(context4)


def test_teacher_concepts(teacher_concepts, curve_setup):
    right, left, middle, slow, fast, gently, firmly = teacher_concepts
    curve1, curve2, curve3, curve4, curve5, curve6 = curve_setup
    assert (right.check_behaviour(curve1) is True)
    assert (left.check_behaviour(curve2) is True)
    assert (middle.check_behaviour(curve3) is True)
    assert (slow.check_behaviour(curve3) is True)
    assert (fast.check_behaviour(curve4) is True)
    assert (gently.check_behaviour(curve5) is True)
    assert (firmly.check_behaviour(curve6) is True)

    assert (right.check_behaviour(curve2) is False)
    assert (left.check_behaviour(curve3) is False)
    assert (middle.check_behaviour(curve1) is False)


def test_teacher1(teacher, curve_setup, context_setup):
    curve1, curve2, curve3, curve4, curve5, curve6 = curve_setup
    context1, context2, context3, context4, _ = context_setup
    assert (teacher.get_feedback(curve1, context1) == 'no, when you see red and heart you should do it right and slowly')


def test_teacher2(teacher, curve_setup, context_setup):
    curve1, curve2, curve3, curve4, curve5, curve6 = curve_setup
    context1, context2, context3, context4, _ = context_setup
    assert (teacher.get_feedback(curve1, context2) == 'no, when you see green and square you should do it right and slowly')


def test_teacher3(teacher, curve_setup, context_setup):
    curve1, curve2, curve3, curve4, curve5, curve6 = curve_setup
    context1, context2, context3, context4, _ = context_setup
    assert (teacher.get_feedback(curve3, context3) == 'no, when you see blue and circle you should do it quickly')


def test_teacher4(teacher, curve_setup, context_setup):
    curve1, curve2, curve3, curve4, curve5, curve6 = curve_setup
    context1, context2, context3, context4, _ = context_setup
    assert (teacher.get_feedback(curve3, context4) == "yes")


def test_teacher5(teacher, curve_setup, context_setup):
    curve1, curve2, curve3, curve4, curve5, curve6 = curve_setup
    context1, context2, context3, context4, _ = context_setup
    assert (teacher.get_feedback(curve4, context3) == "yes")
#
# r3 = Rule([['green', 'heart']], [["left", "slowly"]])
# context5 = Context(["green", "heart"], [])
#
# teacher = TeacherAgent([r, r2, r3], [right, left, middle, slow, fast, gently, firmly])
# teacher.get_feedback(curve1, context1)
# teacher.get_feedback(curve2, context1)
# teacher.get_feedback(curve3, context5)
