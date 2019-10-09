import pytest
from pgmpy.factors.discrete import TabularCPD

from correctingagent.world import rules


def test_cpd_creation_r1():
    red = 'red'
    blue = 'blue'
    rule = rules.RedOnBlueRule(red, blue, rule_type=1)
    time = 1
    violated_rule_factor_name = f"V_{time}({rule})"
    red_o1 = f'{red}(o1)'
    blue_o2 = f'{blue}(o2)'
    violated_rule_cpd = rule.generate_CPD()
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
    violated_rule_cpd = rule.generate_CPD()
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
    violated_rule_cpd = rule.generate_table_cpd()

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
    violated_rule_cpd = rule.generate_table_cpd()

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
