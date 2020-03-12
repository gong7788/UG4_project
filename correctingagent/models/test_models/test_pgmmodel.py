import pytest
from pgmpy.factors.discrete import TabularCPD

from correctingagent.models.pgmmodels import PGMModel, PGMPYInference, InferenceType
from correctingagent.models.prob_model import KDEColourModel
from correctingagent.world import rules
from correctingagent.world.rules import RedOnBlueRule, CorrectionType


def test_cpd_creation_r1():
    red = 'red'
    blue = 'blue'
    rule = rules.RedOnBlueRule(red, blue, rule_type=1)
    time = 1
    violated_rule_factor_name = f"V_{time}({rule})"
    red_o1 = f'{red}(o1)'
    blue_o2 = f'{blue}(o2)'
    violated_rule_cpd = rule.generateCPD(correction_type=CorrectionType.TOWER)

    #violated_rule_cpd = rule.generate_tower_cpd()


    cpd = TabularCPD(violated_rule_factor_name, 2, violated_rule_cpd, evidence=[red_o1, blue_o2, rule],
                     evidence_card=[2, 2, 2])

    assert(cpd.p({'red(o1)': 1, 'blue(o2)': 0, rule: 1, violated_rule_factor_name: 1}) == 1.0)
    assert(cpd.p({'red(o1)': 0, 'blue(o2)': 1, rule: 1, violated_rule_factor_name: 0}) == 1.0)
    assert(cpd.p({'red(o1)': 1, 'blue(o2)': 1, rule: 1, violated_rule_factor_name: 1}) == 0.0)
    assert(cpd.p({'red(o1)': 1, 'blue(o2)': 1, rule: 0, violated_rule_factor_name: 0}) == 1.0)


def test_cpd_creation_r2():
    red = 'red'
    blue = 'blue'
    rule = rules.RedOnBlueRule(red, blue, rule_type=2)
    time = 1
    violated_rule_factor_name = f"V_{time}({rule})"
    red_o1 = f'{red}(o1)'
    blue_o2 = f'{blue}(o2)'
    violated_rule_cpd = rule.generateCPD(correction_type=CorrectionType.TOWER)
    cpd = TabularCPD(violated_rule_factor_name, 2, violated_rule_cpd, evidence=[red_o1, blue_o2, rule],
                     evidence_card=[2, 2, 2])

    assert(cpd.p({'red(o1)': 1, 'blue(o2)': 0, rule: 1, violated_rule_factor_name: 0}) == 1.0)
    assert(cpd.p({'red(o1)': 0, 'blue(o2)': 1, rule: 1, violated_rule_factor_name: 1}) == 1.0)
    assert(cpd.p({'red(o1)': 1, 'blue(o2)': 1, rule: 1, violated_rule_factor_name: 1}) == 0.0)
    assert(cpd.p({'red(o1)': 1, 'blue(o2)': 1, rule: 0, violated_rule_factor_name: 0}) == 1.0)


def test_table_cpd_creation_r1():
    red = 'red'
    blue = 'blue'
    rule = rules.RedOnBlueRule([red], [blue], rule_type=1)
    time = 1
    violated_rule_factor_name = f"V_{time}({rule})"
    red_o1 = f'{red}(o1)'
    blue_o2 = f'{blue}(o2)'
    red_o3 = f'{red}(o3)'
    blue_o3 = f'{blue}(o3)'
    violated_rule_cpd = rule.generateCPD(correction_type=CorrectionType.TABLE)

    cpd = TabularCPD(violated_rule_factor_name, 2, violated_rule_cpd, evidence=[red_o1, blue_o2, red_o3, blue_o3, rule],
                     evidence_card=[2, 2, 2, 2, 2])

    assert(cpd.p({'red(o1)': 0, 'blue(o2)': 1, 'red(o3)':1, 'blue(o3)':0,
                  rule: 1, violated_rule_factor_name: 1}) == 1.0)

    assert (cpd.p({'red(o1)': 0, 'blue(o2)': 1, 'red(o3)': 1, 'blue(o3)': 1,
                   rule: 1, violated_rule_factor_name: 1}) == 1.0)

    assert (cpd.p({'red(o1)': 1, 'blue(o2)': 0, 'red(o3)': 1, 'blue(o3)': 0,
                   rule: 1, violated_rule_factor_name: 1}) == 0.0)

    assert (cpd.p({'red(o1)': 0, 'blue(o2)': 1, 'red(o3)': 0, 'blue(o3)': 0,
                   rule: 1, violated_rule_factor_name: 1}) == 0.0)


def test_table_cpd_creation_r2():
    red = 'red'
    blue = 'blue'
    rule = rules.RedOnBlueRule([red], [blue], rule_type=2)
    time = 1
    violated_rule_factor_name = f"V_{time}({rule})"
    red_o1 = f'{red}(o1)'
    blue_o2 = f'{blue}(o2)'
    red_o3 = f'{red}(o3)'
    blue_o3 = f'{blue}(o3)'
    violated_rule_cpd = rule.generateCPD(correction_type=CorrectionType.TABLE)

    cpd = TabularCPD(violated_rule_factor_name, 2, violated_rule_cpd, evidence=[red_o1, blue_o2, red_o3, blue_o3, rule],
                     evidence_card=[2, 2, 2, 2, 2])

    assert(cpd.p({'red(o1)': 1, 'blue(o2)': 0, 'red(o3)':1, 'blue(o3)':1,
                  rule: 1, violated_rule_factor_name: 1}) == 1.0)

    assert (cpd.p({'red(o1)': 1, 'blue(o2)': 0, 'red(o3)': 0, 'blue(o3)': 1,
                   rule: 1, violated_rule_factor_name: 1}) == 1.0)

    assert (cpd.p({'red(o1)': 0, 'blue(o2)': 1, 'red(o3)': 1, 'blue(o3)': 0,
                   rule: 1, violated_rule_factor_name: 1}) == 0.0)

    assert (cpd.p({'red(o1)': 0, 'blue(o2)': 1, 'red(o3)': 0, 'blue(o3)': 0,
                   rule: 1, violated_rule_factor_name: 1}) == 0.0)


def test_add_cm():
    pgm_model = PGMModel()

    red_cm = KDEColourModel('red')
    blue_cm = KDEColourModel('blue')

    red_b3 = pgm_model.add_cm(red_cm, 'b3')
    blue_t0 = pgm_model.add_cm(blue_cm, 't0')

    assert(red_b3 == 'red(b3)')
    assert(blue_t0 == 'blue(t0)')

    pgm_model.observe({'F(b3)':[1,1,1], 'blue(t0)':0})
    pgm_model.infer()

    query = pgm_model.query(['red(b3)'], [1])
    assert(query['red(b3)'] == 0.5)

    rule = RedOnBlueRule('red', 'blue', 1)

    pgm_model.add_prior(str(rule))

    assert(pgm_model.get_rule_prior(str(rule)) == 0.1)


def test_extend_model():
    pgm_model = PGMModel()

    red_cm = KDEColourModel('red')
    blue_cm = KDEColourModel('blue')
    time = 0
    red_on_blue_rules = rules.Rule.generate_red_on_blue_options('red', 'blue')

    violations = pgm_model.extend_model(red_on_blue_rules, red_cm, blue_cm, ['b1', 'b2'], time, correction_type=CorrectionType.TOWER)

    pgm_model.observe({'F(b1)':[1,1,1], 'F(b2)':[0,0,0], f'corr_{time}':1})
    q = pgm_model.query(violations, [1, 1])
    assert(q[violations[0]] == 0.5)
    assert(q[violations[1]] == 0.5)


def test_extend_model_table():
    pgm_model = PGMModel()

    red_cm = KDEColourModel('red')
    blue_cm = KDEColourModel('blue')
    time = 0
    red_on_blue_rules = rules.Rule.generate_red_on_blue_options('red', 'blue')

    violations = pgm_model.extend_model(red_on_blue_rules, red_cm, blue_cm, ['b1', 'b2', 'b4'], time, correction_type=CorrectionType.TABLE)

    pgm_model.observe({'F(b1)':[1,1,1], 'F(b2)':[0,0,0], f'corr_{time}':1, 'F(b4)':[0.5, 0.5, 0.5]})
    q = pgm_model.query(violations, [1, 1])
    assert((q[violations[0]] - 0.5) < 0.001)
    assert((q[violations[1]] - 0.5) < 0.001)


def test_belief_inference():
    pgm_model = PGMModel(inference_type=InferenceType.BeliefPropagation)

    red_cm = KDEColourModel('red')
    blue_cm = KDEColourModel('blue')

    time = 0
    red_on_blue_rules = rules.Rule.generate_red_on_blue_options('red', 'blue')

    violations = pgm_model.extend_model(red_on_blue_rules, red_cm, blue_cm, ['b1', 'b2'], time, correction_type=CorrectionType.TOWER)

    pgm_model.observe({'F(b1)':[1,1,1], 'F(b2)':[0,0,0], f'corr_{time}':1})


    q = pgm_model.query(violations)
    # inference = PGMPYInference(pgm_model)
    # inference.infer({'F(b1)':[1,1,1], 'F(b2)':[0,0,0], f'corr_{time}':1})
    # q = inference.query(violations)
    # #
    # q = pgm_model.query(violations, [1, 1])

    assert(q[violations[0]] == 0.5)
    assert(q[violations[1]] == 0.5)

    pgm_model.observe({'red(b1)': 1})

    q = pgm_model.query(violations)

    assert(q[violations[0]] == 1.0)
    assert(q[violations[1]] == 0.0)


