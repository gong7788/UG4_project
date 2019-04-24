
from .prob_model import CorrectionModel
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.models import FactorGraph
from pgmpy.factors.continuous import ContinuousFactor
import numpy as np
from ..util.rules import *
from tqdm import tqdm

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


    def infer2(self, evidence):
        variables = set(self.model.nodes()) - set(evidence.keys()) - set(self.model.factors) - set(self.beam[0][0].keys())
        # print(variables)
        if len(variables) == 0:
            return
        factors = [factor for factor in self.model.factors if len(set(factor.scope()) - variables) < len(factor.scope())]

        #factors = []
        #for factor in self.model.factors:


        # print(factors)

        for var, val in evidence.items():
            for factor in factors:
                if var in factor.scope():
                    factor.reduce([(var, val)])

        updated_factors = []
        for factor in factors:
            if isinstance(factor, ContinuousFactor) and len(factor.scope()) == 1:
                factor = DiscreteFactor(factor.scope(), [2], [factor.assignment(0), factor.assignment(1)])
            updated_factors.append(factor)

        phi = reduce(lambda x, y: x * y, updated_factors)
        variables = phi.scope()
        nonzero = np.nonzero(phi.values)
        assert(len(variables) == len(nonzero))
        viable = []
        for i in range(len(nonzero[0])):
            d = {}
            for j, var in enumerate(variables):
                d[var] = nonzero[j][i]
            viable.append(d)
        new_beams = []
        for trace in viable:
            a = []
            for var in phi.scope():
                a.append(trace[var])
            a = tuple(a)
            new_beams.append((trace, phi.values[a]))

        norm = 0
        out_beams = []
        for beam, val in self.beam:
            for new_beam, new_val in new_beams:
                #print(beam, new_beam)
                overlap = set(beam.keys()).intersection(set(new_beam.keys()))
                #print(overlap)
                #for var in overlap:
                #    print(beam[var], new_beam[var])
                if all([beam[var] == new_beam[var] for var in overlap]):
                #    print('in here')
                    out_beam = beam.copy()
                    out_beam.update(new_beam)
                    out_beam.update(evidence)
                    p = val * new_val
                    out_beams.append((out_beam, p))
                    norm += p

        out_beams = [(b, p/norm) for b, p in out_beams]
        self.norm = 1
        self.beam = out_beams
        #print(viable)

    def infer(self, evidence):

        variables = set(self.model.nodes()) - set(evidence.keys()) - set(self.model.factors) - set(self.beam[0][0].keys())
        flips = []
        generate_flips(variables, output=flips)
        new_beam = []
        norm = 0

        # if len(self.beam) > 20:
        #     print(self.beam[0])
        for beam, _ in tqdm(self.beam):
            for flip in tqdm(flips):
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

        self.search_inference.infer2(self.observed)


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
