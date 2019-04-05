import nltk
from nltk.sem import Valuation, Model
from .prob_model import CorrectionModel
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.models import FactorGraph
from pgmpy.factors.continuous import ContinuousFactor
import numpy as np
from functools import reduce
from ..agents.PGMAgent import get_rule_type

def create_rules(colour1, colour2):
    r1 = 'all x.({}(x) -> exists y. ({}(y) & on(x,y)))'.format(colour1, colour2)
    r2 = 'all y.({}(y) -> exists x. ({}(x) & on(x,y)))'.format(colour2, colour1)
    return r1, r2

def create_neg_rule(red, blue):
    rule = "- exists x. exists y. ({}(x) and {}(y) and on(x,y))".format(red, blue)
    return rule

def evaluate_rule(colour1, value1, colour2, value2, rule):
    c1_set = set()
    c2_set = set()
    if value1 == 1:
        c1_set.add('o1')
    if value2 == 1:
        c2_set.add('o2')
    v = [(colour1, c1_set), (colour2, c2_set),
        ('on', set([('o1', 'o2')]))]

    val = Valuation(v)
    dom = val.domain
    m = Model(dom, val)
    g = nltk.sem.Assignment(dom)
    return m.evaluate(rule, g)

def generate_CPD(rule, c1, c2):
    cpd_line_corr0 = []
    cpd_line_corr1 = []

    cpd = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            for r in range(2):
                result = r * (1-int(evaluate_rule(c1, i, c2, j, rule)))
                cpd_line_corr1.append(result)
                cpd_line_corr0.append(1-result)

    return [cpd_line_corr0, cpd_line_corr1]


def generate_table_cpd(rule_type):
    cpd_line_corr0 = []
    cpd_line_corr1 = []

    for r1 in range(2): # rule 1 or 0
        for redo1 in range(2):
            for blueo2 in range(2):
                for redo3 in range(2):
                    for blueo3 in range(2):
                        if rule_type == 1:
                            result = r1 * int(not(redo1) and blueo2 and redo3)
                        elif rule_type == 2:
                            result = r1 * int(redo1 and not(blueo2) and blueo3)
                        cpd_line_corr1.append(result)
                        cpd_line_corr0.append(1-result)
    return [cpd_line_corr0, cpd_line_corr1]


def generate_neg_table_cpd():
    cpd_line_corr0 = []
    cpd_line_corr1 = []
    for r1 in range(2):
        for blueo1 in range(2):
            for redo3 in range(2):
                result = r1 * int(blueo1 and redo3)
                cpd_line_corr1.append(result)
                cpd_line_corr0.append(1-result)

    return [cpd_line_corr0, cpd_line_corr1]

def or_CPD():
    cpd_line1 = []
    cpd_line0 = []
    cpd = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            cpd_line1.append(int(i or j))
            cpd_line0.append(1-int(i or j))
    return [cpd_line0, cpd_line1]

def binary_flip(n):
    out = []
    def helper(n, l):
        if n == 0:
            out.append(l)
        else:
            l1 = l.copy()
            l1.append(0)
            l2 = l.copy()
            l2.append(1)
            helper(n-1, l1)
            helper(n-1, l2)

    helper(n-1, [0])
    helper(n-1, [1])
    return out

def variable_or_CPD(n):
    if n == 0:
        return []
    flippings = binary_flip(n)
    CPD = np.zeros((2, len(flippings)), dtype=np.int32)
    for i, l in enumerate(flippings):
        correction = int(reduce(lambda x, y: x or y, l))
        # correction should happen (C=1)
        CPD[1][i] = correction
        # correction should not happen (C=0)
        CPD[0][i] = 1-correction
    return CPD

def generate_flips(variables, cardinalities=None, evidence={}, output=[]):
    variables = list(variables)
    if cardinalities is None:
        cardinalities = [2]*len(variables)
    if len(variables) == 0:
        output.append(evidence)
    else:
        for i in range(cardinalities[0]):
            new_evidence = evidence.copy()
            new_evidence[variables[0]] = i
            generate_flips(variables[1:], cardinalities=cardinalities[1:], evidence=new_evidence, output=output)



def equals_CPD(seta, setb, carda=None, cardb=None):
    if len(seta) != len(setb):
        raise TypeError('Lengths do not match')
    if carda is None:
        carda = [2]*len(seta)
    if cardb is None:
        cardb = [2]*len(setb)
    results = []
    def helper(settingsa, cardsa, settingsb, cardsb):
        if len(cardsa) == 0:
            results.append(int(all([a == b for a,b in zip(settingsa, settingsb)])))
        else:
            for i in range(cardsa[0]):
                for j in range(cardsb[0]):
                    sa = settingsa.copy()
                    sb = settingsb.copy()
                    helper(sa+[i], cardsa[1:], sb+[j], cardsb[1:])
    helper([], carda, [], cardb)

    return results

def check_beam_holds(beam, worlds):
    if not worlds:
        return True
    return any([all([beam[var] == w[var] if var in beam.keys() else False for var in w.keys()]) for w in worlds])


class SearchInference(object):

    def __init__(self, model, beam_size=-1):
        self.model = model
        self.beam = [({},1)]
        self.beam_size = -1
        self.norm = 0

    def infer(self, evidence):

        variables = set(self.model.nodes()) - set(evidence.keys()) - set(self.model.factors) - set(self.beam[0][0].keys())
        flips = []
        generate_flips(variables, output=flips)
        new_beam = []
        norm = 0
        for beam, _ in self.beam:
            for flip in flips:
                new_evidence = evidence.copy()
                new_evidence.update(flip)
                new_evidence.update(beam)
                p = 1
                for factor in self.model.factors:
                    p *= factor.p(new_evidence)

                if p != 0:
                    new_beam.append((new_evidence, p))
                    norm += p

        new_beam = [(b, p/norm) for b, p in new_beam]

        self.norm = 1
        self.beam = new_beam

    def p(self, var, val):
        try:
            return sum([p for (beam, p) in self.beam if beam[var]==val])/self.norm
        except KeyError:
            return None


    def clamp(self, evidence):

        for var, val in evidence.items():
            self.beam = [(beam, p) for beam, p in self.beam if beam[var] == val]
        self.norm = sum([p for _, p in self.beam])


    def variable_clamp(self, worlds):
        self.beam = [(beam, val) for beam, val in self.beam if check_beam_holds(beam, worlds)]


    def query(self, variables, values=None):
        if values == None:
            values = [1]*len(variables)
        return {var:self.p(var, val) for var, val in zip(variables, values)}


class PGMModel(object):

    def __init__(self):
        self.model = FactorGraph()
        #self.inference = None
        self.known_rules = {}
        self.factors = set()
        self.observed = {}
        self.rule_priors = {}
        self.colour_variables = []
        self.colours = {}
        self.search_inference = SearchInference(self.model)

    def reset(self):
        self.model = FactorGraph()
        self.search_inference = SearchInference(self.model)
        self.factors = set()
        self.observed = {}
        self.colour_variables = []

    def add_rules(self, rules, cm1, cm2):
        self.colours[cm1.name] = cm1
        self.colours[cm2.name] = cm2
        for rule in rules:
            self.known_rules[rule] = (cm1.name, cm2.name)

    def add_cm(self, cm, block):

        red_rgb = 'F({})'.format(block)
        red_o1 = '{}({})'.format(cm.name, block)
        self.colours[cm.name] = cm
        self.colour_variables.append(red_o1)
        if 't' in block:
            self.model.add_nodes_from([red_o1])
            return red_o1
        if red_o1 not in self.model.nodes():
            red_distribution = lambda rgb,c: cm.p(c, rgb)
            red_evidence = [red_rgb, red_o1]
            red_cm_factor = ContinuousFactor(red_evidence, red_distribution)
            self.add_factor(red_evidence + [red_cm_factor], red_cm_factor)
        return red_o1


    def update_model_no_corr(self, args, time):
        pass

    def add_factor(self, nodes, factor):
        node_filter = lambda x: filter(lambda y: y not in self.model.nodes(), x)
        self.factors.add(factor)
        self.model.add_nodes_from(node_filter(set(nodes + [factor])))
        self.model.add_factors(factor)

#     def node_filter(self, nodes):
#         return filter(lambda y: y not in self.model.nodes(), nodes)


    def get_rule_prior(self, rule):
        try:
            return self.rule_priors[rule]
        except KeyError:
            return 0.01

    def add_prior(self, rule):
        if rule not in self.known_rules:
            rule_prior_value = self.get_rule_prior(rule)

            rule_prior = DiscreteFactor([rule], [2], [1-rule_prior_value, rule_prior_value])
            self.add_factor([rule], rule_prior)


    def add_no_correction(self, args, time):
        if not self.known_rules:
            return
        o1, o2 = args

        violations = []


        for rule, (red, blue) in self.known_rules.items():
            red_o1 = self.add_cm(self.colours[red], o1)
            blue_o2 = self.add_cm(self.colours[blue], o2)

            violated_rule = generate_CPD(rule, red, blue)
            Vrule = 'V_{}({})'.format(time, rule)

            rule_violated = TabularCPD(Vrule, 2, violated_rule, evidence=[red_o1, blue_o2, rule], evidence_card = [2,2,2])
            rule_violated_factor = rule_violated.to_factor()
            self.add_factor([Vrule, rule, rule_violated_factor], rule_violated_factor)

            violations.append(Vrule)

            #violations.append(violated_rule)
        corr = 'corr_{}'.format(time)
        correction_table = variable_or_CPD(len(violations))
        correction = TabularCPD(corr, 2, correction_table, evidence=violations, evidence_card=[2]*len(violations))
        correction_factor = correction.to_factor()
        self.add_factor([corr, correction_factor], correction_factor)
        return violations

    def add_no_table_correciton(self, args, time):
        if not self.known_rules:
            return

        violations = []

        for rule, (red, blue) in self.known_rules.items():
            violations.append(self.add_table_violation(rule, self.colours[red], self.colours[blue], args, time))

        corr = 'corr_{}'.format(time)
        correction_table = variable_or_CPD(len(violations))
        correction = TabularCPD(corr, 2, correction_table, evidence=violations, evidence_card=[2]*len(violations))
        correction_factor = correction.to_factor()
        self.add_factor([corr, correction_factor], correction_factor)
        return violations

    def create_negative_model(self, rule, red_cm, blue_cm, args, time):
        o1, o2 = args[:2]

        red_o1 = self.add_cm(red_cm, o1)
        blue_o2 = self.add_cm(blue_cm, o2)

        self.add_prior(rule)
        self.known_rules[r1] = [red_cm.name, blue_cm.name]
        violated_rule = generate_CPD(rule, red_cm.name, blue_cm.name)

        correction_table = variable_or_CPD(1)

        Vrule = 'V_{}({})'.format(time, r1)
        corr = 'corr_{}'.format(time)

        rule_violated = TabularCPD(Vrule, 2, violated_rule, evidence=[red_o1, blue_o2, rule], evidence_card = [2,2,2])
        rule_violated_factor = rule_violated.to_factor()
        self.add_factor([Vrule, rule, rule_violated_factor], rule_violated_factor)

        correction = TabularCPD(corr, 2, correction_table, evidence=[Vrule], evidence_card=[2])
        correction_factor = correction.to_factor()
        self.add_factor([corr, correction_factor], correction_factor)

        return [Vrule]

    def create_table_neg_model(self, rule, red_cm, blue_cm, args, time):
        o1, o2, o3 = args[:3]
        blue_o1 = self.add_cm(blue_cm, o1)
        red_o3 = self.add_cm(red_cm, o3)

        self.add_prior(rule)

        self.known_rules[rule] = [red_cm.name, blue_cm.name]


        rule_cpd = generate_neg_table_cpd()
        correction_table = variable_or_CPD(1)


        Vrule = 'V_{}({})'.format(time, rule)
        corr = 'corr_{}'.format(time)

        rule_evidence = [rule, blue_o1, red_o3]
        rule_violated = TabularCPD(Vrule, 2, rule_cpd, evidence=rule_evidence, evidence_card=[2,2,2,2,2])
        self.add_factor(rule_evidence + [Vrule], rule_violated.to_factor())

        correction =  TabularCPD(corr, 2, correction_table, evidence=[Vrule], evidence_card=[2])
        self.add_factor([Vrule, corr], correction.to_factor())

        #self.inference = VariableElimination(self.model)
        return [Vr1]


    def create_tower_model(self, rules, red_cm, blue_cm, args, time):


        # create the factors for the colour models

        o1, o2 = args[:2]

        red_o1 = self.add_cm(red_cm, o1)
        blue_o2 = self.add_cm(blue_cm, o2)

        r1, r2 = rules

        self.add_prior(r1)
        self.add_prior(r2)

        self.known_rules[r1] = [red_cm.name, blue_cm.name]
        self.known_rules[r2] = [red_cm.name, blue_cm.name]



        #Create the factors for the rule violation
        # Create the truth tables for each rule and their combination
        violated_r1 = generate_CPD(r1, red_cm.name, blue_cm.name)
        violated_r2 = generate_CPD(r2, red_cm.name, blue_cm.name)


        correction_table = or_CPD()

        Vr1 = 'V_{}({})'.format(time, r1)
        Vr2 = 'V_{}({})'.format(time, r2)
        corr = 'corr_{}'.format(time)



        r1_violated = TabularCPD(Vr1, 2, violated_r1, evidence=[red_o1, blue_o2, r1], evidence_card = [2,2,2])
        r1_violated_factor = r1_violated.to_factor()
        self.add_factor([Vr1, r1, r1_violated_factor], r1_violated_factor)

        r2_violated = TabularCPD(Vr2, 2, violated_r2, evidence=[red_o1, blue_o2, r2], evidence_card = [2,2,2])
        r2_violated_factor = r2_violated.to_factor()
        self.add_factor([Vr2, r2, r2_violated_factor], r2_violated_factor)


        correction = TabularCPD(corr, 2, correction_table, evidence=[Vr1, Vr2], evidence_card=[2,2])
        correction_factor = correction.to_factor()
        self.add_factor([corr, correction_factor], correction_factor)



        #self.inference = VariableElimination(self.model)

        return Vr1, Vr2


    def add_table_violation(self, rule, red_cm, blue_cm, args, time):
        rule_type = get_rule_type(rule)
        o1, o2, o3 = args[:3]
        red_o1 = self.add_cm(red_cm, o1)
        blue_o2 = self.add_cm(blue_cm, o2)
        red_o3 = self.add_cm(red_cm, o3)
        blue_o3 = self.add_cm(blue_cm, o3)
        if rule_type == 'r1':
            rule_cpd = generate_table_cpd(1)
        elif rule_type == 'r2':
            rule_cpd = generate_table_cpd(2)

        self.add_prior(rule)

        rule_cpd = generate_table_cpd(1)

        Vrule = 'V_{}({})'.format(time, rule)
        rule_evidence = [rule, red_o1, blue_o2, red_o3, blue_o3]
        rule_violated = TabularCPD(Vrule, 2, rule_cpd, evidence=rule_evidence, evidence_card=[2,2,2,2,2])
        self.add_factor(rule_evidence + [Vrule], rule_violated.to_factor())

        return Vrule

    def create_table_model(self, rules, red_cm, blue_cm, args, time, hold_correction=False):
        o1, o2, o3 = args[:3]
        red_o1 = self.add_cm(red_cm, o1)
        blue_o2 = self.add_cm(blue_cm, o2)
        red_o3 = self.add_cm(red_cm, o3)
        blue_o3 = self.add_cm(blue_cm, o3)

        r1, r2 = rules

        self.add_prior(r1)
        self.add_prior(r2)

        self.known_rules[r1] = [red_cm.name, blue_cm.name]
        self.known_rules[r2] = [red_cm.name, blue_cm.name]



        r1_cpd = generate_table_cpd(1)
        r2_cpd = generate_table_cpd(2)
        correction_table = or_CPD()


        Vr1 = 'V_{}({})'.format(time, r1)
        Vr2 = 'V_{}({})'.format(time, r2)
        corr = 'corr_{}'.format(time)

        r1_evidence = [r1, red_o1, blue_o2, red_o3, blue_o3]
        r1_violated = TabularCPD(Vr1, 2, r1_cpd, evidence=r1_evidence, evidence_card=[2,2,2,2,2])
        self.add_factor(r1_evidence + [Vr1], r1_violated.to_factor())

        r2_evidence = [r2, red_o1, blue_o2, red_o3, blue_o3]
        r2_violated = TabularCPD(Vr2, 2, r2_cpd, evidence=r2_evidence, evidence_card=[2,2,2,2,2])
        self.add_factor(r2_evidence + [Vr2], r2_violated.to_factor())

        if not hold_correction:
            correction =  TabularCPD(corr, 2, correction_table, evidence=[Vr1, Vr2], evidence_card=[2,2])
            self.add_factor([Vr1, Vr2, corr], correction.to_factor())

        #self.inference = VariableElimination(self.model)
        return Vr1, Vr2


    def add_same_reason(self, current_violations, previous_violations):
        evidence = []
        for c, p in zip(current_violations, previous_violations):
            evidence.extend([c,p])

        t = equals_CPD(current_violations, previous_violations)
        f = DiscreteFactor(evidence, [2]*len(evidence), t)
        self.add_factor(evidence, f)


    def observe(self, observable={}):
        overlapping_vars = False
        if self.search_inference.beam:
            remains = set(observable.keys()) - set(self.observed.keys())
            overlapping_vars = remains.intersection(set(self.search_inference.beam[0][0].keys()))

        if overlapping_vars:
            self.search_inference.clamp({o:observable[o] for o in overlapping_vars})
        self.observed.update(observable)
        self.infer([])

    def observe_uncertain(self, worlds):
        self.search_inference.variable_clamp(worlds)
    # def infer(self, targets=[]):
    #     #print(self.observed.keys())
    #     #print(targets)
    #     hidden = set(self.model.nodes) - set(self.observed.keys()) - self.factors - set(targets)
    #     #print(hidden)
    #     q = self.inference.query(variables=targets,
    #             evidence=self.observed,
    #             elimination_order=hidden)
    #     self.query = q
    #     return q

    def infer(self, variable=[]):
        self.search_inference.infer(self.observed)


    def query(self, variables, values=None):
        return self.search_inference.query(variables, values=values)

    # def get_rule_probs(self, update_prior=False):
    #     rules = list(self.known_rules.keys())
    #     q = self.infer(rules)
    #     if update_prior:
    #         for rule in rules:
    #             self.rule_priors[rule] = q[rule].values[1]
    #     return {rule: q[rule].values[1] for rule in rules}

    def get_rule_probs(self, update_prior=False):
        rules = list(self.known_rules.keys())
        q = self.search_inference.query(rules)
        for rule, p in q.items():
            if p is None:
                q[rule] = self.rule_priors[rule]

        if update_prior:
            for rule in rules:
                self.rule_priors[rule] = q[rule]
        return q

    # def get_colour_predictions(self):
    #     colours = [c for c in self.colours if c not in self.observed]
    #     q = self.infer(colours)
    #
    #     out = dict(((c,q[c].values[1]) if c not in self.observed else (c,self.observed[c]) for c in self.colours))
    #     return out

    def get_colour_predictions(self):
        #colours = [c for c in self.colours if c not in self.observed]
        q = self.search_inference.query(self.colour_variables)

        return q
