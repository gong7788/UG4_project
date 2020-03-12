import copy
from enum import Enum
from functools import reduce

from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import BeliefPropagation
from pgmpy.models import FactorGraph, BayesianModel
from pgmpy.factors.continuous import ContinuousFactor
import numpy as np

from correctingagent.models.prob_model import KDEColourModel
from correctingagent.util.CPD_generation import variable_or_CPD, generate_neg_table_cpd, equals_CPD
from correctingagent.world.rules import ColourCountRule, RedOnBlueRule, CorrectionType




def check_beam_holds(beam, worlds):
    if not worlds:
        return True
    return any([all([beam[var] == w[var] if var in beam.keys() else False for var in w.keys()]) for w in worlds])


class PGMPYInference(object):

    def __init__(self, model):
        self.model = model
        self.evidence = {}
        self.belief_propagation = None
        self.scope = set()

    def infer(self, evidence, new_evidence):

        evidence.update(new_evidence)

        model = copy.deepcopy(self.model)
        continuous_factors = [factor for factor in model.factors if isinstance(factor, ContinuousFactor)]

        for var, val in evidence.items():
            for factor in continuous_factors:
                if var in factor.scope() and "F(" in var:  # make sure that we only reduce at this stage for continuous values, let the inference algorithm deal with reducing for binary variables
                    factor.reduce([(var, val)])


        # updated_factors = []
        # for factor in model.factors:
        #     if isinstance(factor, ContinuousFactor):
        #         if len(factor.scope()) == 1:
        #             factor = DiscreteFactor(factor.scope(), [2], [factor.assignment(0), factor.assignment(1)])
        #             updated_factors.append(factor)
        #     else:
        #         updated_factors.append(factor)
        #     model.factors = updated_factors

        new_model = FactorGraph()

        for node in model.factors:
            if isinstance(node, ContinuousFactor):
                if len(node.scope()) == 1:
                    node = DiscreteFactor(node.scope(), [2], [node.assignment(0), node.assignment(1)])
            new_model.add_nodes_from(node.scope())
            new_model.add_factors(node)
            new_model.add_nodes_from([node])

        belief_propagation = BeliefPropagation(new_model)

        self.evidence = {var: val for (var, val) in evidence.items() if "F(" not in var}
        self.belief_propagation = belief_propagation
        self.scope = get_scope(new_model)
        print(self.scope)
        return new_model

    def query(self, variables, values=None):
        output = {}
        vars_to_query = []
        vals_to_query = []

        if values is None:
            values = [1]*len(variables)

        for variable, val in zip(variables, values):
            # Some of the variables will be in the evidence and will therefore have been reduced away in the model
            # These should either have 1.0 or 0.0 probability
            # This depends on whether the evidence value matches the wanted value
            if variable in self.evidence:
                output[variable] = int(self.evidence[variable] == val)
            # Some of the variables queried will be rules which have not been added to the current model
            # The chosen convention for these is to simply use None as their value
            # This is consistent with the functionallity of SearchInference.query / SearchInference.p
            elif variable not in self.scope:
                output[variable] = None
            else:
                vars_to_query.append(variable)
                vals_to_query.append(val)


        q = self.belief_propagation.query(variables=vars_to_query, evidence=self.evidence)
        output.update({var: q[var].values[val] for var, val in zip(vars_to_query, vals_to_query)})
        return output


def get_scope(model):
    scope = set()
    for variable in model.factors:
        scope = scope.union(variable.scope())
    return scope


class SearchInference(object):

    def __init__(self, model, beam_size=-1):
        self.model = model
        self.beam = [({}, 1)]
        self.beam_size = beam_size
        self.norm = 0

    def infer(self, evidence, new_evidence):
        # print("starting inference")
        # step = time.time()

        overlapping_vars = False
        if self.search_inference.beam:
            remains = set(new_evidence.keys()) - set(self.observed.keys())
            overlapping_vars = remains.intersection(set(self.beam[0][0].keys()))

        if overlapping_vars:
            self.clamp({o: new_evidence[o] for o in overlapping_vars})

        variables = set(self.model.nodes()) - set(evidence.keys()) - set(self.model.factors) - set(self.beam[0][0].keys())
        if len(variables) == 0:
            return
        factors = [factor for factor in self.model.factors if len(set(factor.scope()) - variables) < len(factor.scope())]
        # step_1 = time.time()
        # delta = step_1-step
        # print(f"finding relevant variable {delta} time")
        # step = time.time()

        for var, val in evidence.items():
            for factor in factors:
                if var in factor.scope():
                    factor.reduce([(var, val)])
        # step_1 = time.time()
        # delta = step_1 - step
        # print(f"reducing factors in evidence {delta} time")
        # step = time.time()

        updated_factors = []
        for factor in factors:
            if isinstance(factor, ContinuousFactor) and len(factor.scope()) == 1:
                factor = DiscreteFactor(factor.scope(), [2], [factor.assignment(0), factor.assignment(1)])
            updated_factors.append(factor)
        # step_1 = time.time()
        # delta = step_1 - step
        # print(f"updating factors {delta} time")
        # step = time.time()

        phi = reduce(lambda x, y: x * y, updated_factors)
        # step_1 = time.time()
        # delta = step_1 - step
        # print(f"reducing factors {delta} time")
        # step = time.time()

        variables = phi.scope()
        nonzero = np.nonzero(phi.values)
        assert(len(variables) == len(nonzero))
        viable = []
        for i in range(len(nonzero[0])):
            d = {}
            for j, var in enumerate(variables):
                d[var] = nonzero[j][i]
            viable.append(d)
        # step_1 = time.time()
        # delta = step_1 - step
        # print(f"finding nonzero values {delta} time")
        # step = time.time()
        new_beams = []
        for trace in viable:
            a = []
            for var in phi.scope():
                a.append(trace[var])
            a = tuple(a)
            new_beams.append((trace, phi.values[a]))
        # step_1 = time.time()
        # delta = step_1 - step
        # print(f"finding beams {delta} time")
        # step = time.time()

        norm = 0
        out_beams = []
        for beam, val in self.beam:
            for new_beam, new_val in new_beams:

                overlap = set(beam.keys()).intersection(set(new_beam.keys()))

                if all([beam[var] == new_beam[var] for var in overlap]):
                    out_beam = beam.copy()
                    out_beam.update(new_beam)
                    out_beam.update(evidence)
                    p = val * new_val
                    out_beams.append((out_beam, p))
                    norm += p
        # step_1 = time.time()
        # delta = step_1 - step
        # print(f"combining beams {delta} time")

        out_beams = [(b, p/norm) for b, p in out_beams]
        self.norm = 1
        self.beam = out_beams

    def p(self, var, val):
        try:
            return sum([p for (beam, p) in self.beam if beam[var]==val])/self.norm
        except KeyError:
            return None

    def clamp(self, evidence):

        for var, val in evidence.items():
            self.beam = [(beam, p) for beam, p in self.beam if beam[var] == val]
        self.norm = sum([p for _, p in self.beam])
    #
    # def variable_clamp(self, worlds):
    #     self.beam = [(beam, val) for beam, val in self.beam if check_beam_holds(beam, worlds)]

    def query(self, variables, values=None):
        if values is None:
            values = [1]*len(variables)
        return {var: self.p(var, val) for var, val in zip(variables, values)}

def is_overlap(set1, set2):
    set1 = set(set1)
    set2 = set(set2)

    return len(set1.intersection(set2)) > 0

def is_scope_overlap(model1, model2):
    scope1 = get_scope(model1)
    scope2 = get_scope(model2)
    return is_overlap(scope1, scope2)

class InferenceType(Enum):
    SearchInference = 1
    BeliefPropagation = 2


def combine_models(model1, model2):
    assert(is_scope_overlap(model1, model2))

    for node in model2.factors:
        model1.add_nodes_from(node.scope())
        model1.add_factors(node)
        model1.add_nodes_from([node])

    return model1



class PGMModel(object):

    def __init__(self, inference_type=InferenceType.SearchInference):
        self.known_rules = set()
        self.rule_priors = {}
        self.colours = {}
        self.inference_type = inference_type
        self.reset()

    def get_model_scopes(self):
        scope = set()
        for model in self.models:
            scope = scope.union(get_scope(model))
        return scope

    def get_nodes(self):
        nodes = set()
        for model in self.models:
            nodes = nodes.union(model.nodes())
        return nodes

    def reset(self):
        """ Resets variables which need to be updated for each new scenario
        """
        self.model = FactorGraph()
        if self.inference_type == InferenceType.SearchInference:
            self.search_inference = SearchInference(self.model)
        elif self.inference_type == InferenceType.BeliefPropagation:
            self.search_inference = PGMPYInference(self.model)
        else:
            raise TypeError('Invalid Inference Type')
        self.observed = {}
        self.colour_variables = []

        self.models = []

    def add_cm(self, cm, block):



        red_rgb = f'F({block})'
        red_o1 = f'{cm.name}({block})'
        self.colours[cm.name] = cm
        self.colour_variables.append(red_o1)
        if 't' in block:
            self.model.add_nodes_from([red_o1])
            return red_o1
        if red_o1 not in self.model.nodes():
            red_distribution = lambda rgb, c: cm.p(c, rgb)
            red_evidence = [red_rgb, red_o1]
            red_cm_factor = ContinuousFactor(red_evidence, red_distribution)
            self.add_factor(red_evidence + [red_cm_factor], red_cm_factor)
        return red_o1

    def add_factor(self, nodes, factor):

        if self.inference_type != InferenceType.SearchInference:
            new_model = FactorGraph()
            new_model.add_nodes_from(nodes + [factor])
            new_model.add_factor(factor)
            combined = False
            for model in self.models:
                if is_overlap(get_scope(model), get_scope(new_model)):
                    combine_models(model, new_model)
                    combined = True
            if not combined:
                self.models.append(new_model)
        else:
            node_filter = lambda x: filter(lambda y: y not in self.model.nodes(), x)
            self.model.add_nodes_from(node_filter(set(nodes + [factor])))
            self.model.add_factors(factor)

    def get_rule_prior(self, rule):
        try:
            return self.rule_priors[rule]
        except KeyError:
            return 0.1

    def add_prior(self, rule):
        if rule not in self.known_rules:
            rule_prior_value = self.get_rule_prior(rule)

            rule_prior = DiscreteFactor([rule], [2], [1-rule_prior_value, rule_prior_value])
            self.add_factor([rule], rule_prior)

    def add_no_correction(self, args, time, rules):
        if not rules:
            return
        o1, o2 = args[:2]

        violations = []

        for rule in rules:
            print("no correction", args, time, rules)
            if isinstance(rule, RedOnBlueRule):
                red = rule.c1
                blue = rule.c2
                colour_models = self.add_cms(self.colours[red], self.colours[blue], [o1, o2],
                                             correction_type=CorrectionType.TOWER)

                Vrule = self.add_violation_factor(rule, time, colour_models, correction_type=CorrectionType.TOWER)

                violations.append(Vrule)
            elif isinstance(rule, ColourCountRule):
                return []
            else:
                raise ValueError(f"Unexpected rule type {type(rule)}")

        self.add_correction_factor(violations, time)

        return violations

    def add_no_table_correciton(self, args, time):
        if not self.known_rules:
            return

        violations = []

        for rule in self.known_rules:
            red = rule.c1
            blue = rule.c2
            violations.append(self.add_table_violation(rule, self.colours[red], self.colours[blue], args, time))

        self.add_correction_factor(violations, time)

        return violations

    def add_violation_factor(self, rule, time, colours, correction_type=CorrectionType.TABLE, table_empty=False):
        """ Adds a factor representing when the rule is violated

        :param colours:
        :param correction_type:
        :param table_empty:
        :param rule: Rule object representing the rule being violated
        :param time: the time step at which the violation happened
        :param red_o1: factor name of form colour(o1) as returned by add_cm()
        :param blue_o2: factor name of form colour(o2) as returned by add_cm()
        :return: the factor name in the form V_time(rule)
        """
        violated_rule_factor_name = f"V_{time}({rule})"

        evidence = colours + [rule]

        violated_rule_cpd = rule.generateCPD(correction_type=correction_type, len_evidence=len(evidence),
                                             table_empty=table_empty)

        try:

            rule_violated_factor = TabularCPD(violated_rule_factor_name, 2, violated_rule_cpd,
                                              evidence=evidence, evidence_card=[2]*len(evidence))
        except ValueError as e:
            print("rule", rule)
            print("time", time)
            print("correction_type", correction_type)
            print("table empty", table_empty)
            print("evidence", evidence)
            print("len evidence", str(len(evidence)))
            print("cpd_shape", np.array(violated_rule_cpd).shape)
            raise e
        rule_violated_factor = rule_violated_factor.to_factor()

        self.add_factor([violated_rule_factor_name, rule, rule_violated_factor], rule_violated_factor)
        return violated_rule_factor_name

    def add_correction_factor(self, violations, time):
        """
        :param violations:
        :param time:
        :return:
        """
        corr = f'corr_{time}'
        correction_table = variable_or_CPD(len(violations))
        correction = TabularCPD(corr, 2, correction_table, evidence=violations, evidence_card=[2]*len(violations))
        correction_factor = correction.to_factor()
        self.add_factor([corr, correction_factor], correction_factor)
        return corr

    def add_cc_and_rob(self, colour_count: ColourCountRule, red_on_blue_options: list, red_cm: KDEColourModel,
                       blue_cm: KDEColourModel, objects_in_tower: list, top_object:str, time:int):
        objects_in_tower = objects_in_tower.copy()
        self.known_rules.add(colour_count)
        red_on_blue1, red_on_blue2 = red_on_blue_options
        self.known_rules.add(red_on_blue1)
        self.known_rules.add(red_on_blue2)

        objects_in_tower.remove(top_object)
        colours_in_tower = [self.add_cm(red_cm, obj) for obj in objects_in_tower] + [self.add_cm(blue_cm, top_object)]
        violations = self.add_joint_violations(colour_count, red_on_blue_options, time, colours_in_tower)
        self.add_correction_factor(violations, time)
        return violations

    def add_joint_violations(self, colour_count: ColourCountRule, red_on_blue_options: list, time: int,
                             colours_in_tower: list):
        cpds = [colour_count.generateCPD(num_blocks_in_tower=len(colours_in_tower), correction_type=CorrectionType.TABLE)
                for rule in red_on_blue_options]
        violated_rule_factor_name1 = f"V_{time}({colour_count} && {red_on_blue_options[0]})"
        violated_rule_factor_name2 = f"V_{time}({colour_count} && {red_on_blue_options[1]})"

        evidence1 = colours_in_tower + [red_on_blue_options[0], colour_count]
        evidence2 = colours_in_tower + [red_on_blue_options[1], colour_count]

        cpd1, cpd2 = cpds

        rule_violated_factor1 = TabularCPD(violated_rule_factor_name1, 2, cpd1,
                                          evidence=evidence1, evidence_card=[2] * len(evidence1))
        rule_violated_factor1 = rule_violated_factor1.to_factor()

        rule_violated_factor2 = TabularCPD(violated_rule_factor_name2, 2, cpd1,
                                           evidence=evidence2, evidence_card=[2] * len(evidence2))
        rule_violated_factor2 = rule_violated_factor2.to_factor()

        self.add_factor([violated_rule_factor_name1, rule_violated_factor1, red_on_blue_options[0], colour_count], rule_violated_factor1)
        self.add_factor([violated_rule_factor_name2, rule_violated_factor2, red_on_blue_options[1], colour_count], rule_violated_factor2)
        return [violated_rule_factor_name1, violated_rule_factor_name2]

    def add_colour_count_correction(self, rule: ColourCountRule, cm: KDEColourModel, objects_in_tower: list, time: int):
        self.known_rules.add(rule)
        colour_variables = []
        for obj in objects_in_tower:
            colour_variables.append(self.add_cm(cm, obj))
        violations = [self.add_violation_factor(rule, time, colour_variables)]
        self.add_correction_factor(violations, time)
        return violations

    def extend_model(self, rules, red_cm, blue_cm, args, time, correction_type, table_empty=False):

        colour_models = self.add_cms(red_cm, blue_cm, args, correction_type=correction_type)

        if table_empty:
            colour_models.append(self.add_cm(blue_cm, args[0]))

        r1, r2 = rules

        self.add_prior(r1)
        self.add_prior(r2)

        self.known_rules.add(r1)
        self.known_rules.add(r2)

        violated_r1 = self.add_violation_factor(r1, time, colour_models,
                                                correction_type=correction_type, table_empty=table_empty)
        violated_r2 = self.add_violation_factor(r2, time, colour_models,
                                                correction_type=correction_type, table_empty=table_empty)
        self.add_correction_factor([violated_r1, violated_r2], time)

        return violated_r1, violated_r2

    def add_cms(self, red_cm, blue_cm, args, correction_type=CorrectionType.TABLE):
        o1, o2 = args[:2]
        red_o1 = self.add_cm(red_cm, o1)
        blue_o2 = self.add_cm(blue_cm, o2)

        if correction_type == CorrectionType.TABLE:
            o3 = args[2]
            red_o3 = self.add_cm(red_cm, o3)
            blue_o3 = self.add_cm(blue_cm, o3)
            return [red_o1, blue_o2, red_o3, blue_o3]
        else:
            return [red_o1, blue_o2]

    def add_same_reason(self, current_violations, previous_violations):
        # print("adding same reason")
        # print(current_violations)
        # print(previous_violations)
        evidence = []
        for c, p in zip(current_violations, previous_violations):
            evidence.extend([c, p])

        # evidence = list(current_violations) + list(previous_violations)
        t = equals_CPD(current_violations, previous_violations)

        # print(t)
        f = DiscreteFactor(evidence, [2]*len(evidence), t)
        self.add_factor(evidence, f)

    def observe(self, observable={}):

        self.search_inference.infer(self.observed, observable)
        self.observed.update(observable)

    # def observe_uncertain(self, worlds):
    #     self.search_inference.variable_clamp(worlds)

    # def infer(self):
    #     # for beam in self.search_inference.beam:
    #     #     print(beam)
    #
    #
    #
    #     # for beam in self.search_inference.beam:
    #     #     print(beam)

    def query(self, variables, values=None):
        return self.search_inference.query(variables, values=values)

    def get_rule_probs(self, update_prior=False):
        rules = [rule for rule in self.known_rules]
        q = self.search_inference.query(rules)
        for rule, p in q.items():
            if p is None:
                q[rule] = self.rule_priors[rule]

        if update_prior:
            for rule in rules:
                self.rule_priors[rule] = q[rule]
        return q

    def get_colour_predictions(self):
        q = self.search_inference.query(self.colour_variables)

        return q

    def create_negative_model(self, rule, red_cm, blue_cm, args, time):
        o1, o2 = args[:2]

        red_o1 = self.add_cm(red_cm, o1)
        blue_o2 = self.add_cm(blue_cm, o2)

        self.add_prior(rule)
        self.known_rules.add(rule)
        rule_violation_CPD = rule.generate_CPD()

        correction_table = variable_or_CPD(1)

        Vrule = f'V_{time}({rule})'
        corr = f'corr_{time}'

        rule_violated = TabularCPD(Vrule, 2, rule_violation_CPD, evidence=[red_o1, blue_o2, rule], evidence_card=[2, 2, 2])
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

        self.known_rules.add(rule)

        rule_cpd = generate_neg_table_cpd()
        correction_table = variable_or_CPD(1)

        Vrule = f'V_{time}({rule})'
        corr = f'corr_{time}'

        rule_evidence = [rule, blue_o1, red_o3]
        rule_violated = TabularCPD(Vrule, 2, rule_cpd, evidence=rule_evidence, evidence_card=[2,2,2,2,2])
        self.add_factor(rule_evidence + [Vrule], rule_violated.to_factor())

        correction = TabularCPD(corr, 2, correction_table, evidence=[Vrule], evidence_card=[2])
        self.add_factor([Vrule, corr], correction.to_factor())

        return [Vrule]

    def create_uncertain_table_model(self, rules, red_cm, blue_cm, args,
                                     objects_in_tower, time):

        self.known_rules = self.known_rules.union(rules)
        rule1, rule2 = rules

        reds = [self.add_cm(red_cm, obj) for obj in objects_in_tower]
        blues = [self.add_cm(blue_cm, obj) for obj in objects_in_tower[1:]]
        red_o3 = self.add_cm(red_cm, args[-1])
        blue_o3 = self.add_cm(blue_cm, args[-1])

        colour_variables = reds + blues + [red_o3, blue_o3]

        violations = [self.add_violation_factor(rule1, time, colour_variables, correction_type=CorrectionType.UNCERTAIN_TABLE),
                      self.add_violation_factor(rule2, time, colour_variables, correction_type=CorrectionType.UNCERTAIN_TABLE)]
        self.add_correction_factor(violations, time)
        return violations


