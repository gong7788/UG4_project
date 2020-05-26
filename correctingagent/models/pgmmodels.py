import copy
import heapq
import time
from _operator import mul
from collections import defaultdict
from enum import Enum
from functools import reduce

from pgmpy.factors.discrete import TabularCPD, DiscreteFactor, State
from pgmpy.inference import BeliefPropagation
from pgmpy.models import FactorGraph, BayesianModel
from pgmpy.factors.continuous import ContinuousFactor
import numpy as np
from pgmpy.sampling import GibbsSampling, BayesianModelSampling

from correctingagent.models.prob_model import KDEColourModel
from correctingagent.util.CPD_generation import variable_or_CPD, generate_neg_table_cpd, equals_CPD
from correctingagent.world.rules import ColourCountRule, RedOnBlueRule, CorrectionType


def check_beam_holds(beam, worlds):
    if not worlds:
        return True
    return any([all([beam[var] == w[var] if var in beam.keys() else False for var in w.keys()]) for w in worlds])


def to_CPD(factor):
    factor0_str = str(factor.variables[0])
    if 'corr' in factor0_str:
        corr_factor = factor.variables[0]
        violations = factor.variables[1:]
        num_violations = len(violations)
        values = factor.values.reshape(2, 2**num_violations)
        return TabularCPD(corr_factor, 2, values, evidence=violations, evidence_card=[2]*num_violations)
    if len(factor.variables) == 1 and 'all' in factor0_str:
        return TabularCPD(factor.variables[0], 2, [factor.values])
    if "V_" in factor0_str and "V_" not in str(factor.variables[1]):
        violation_factor = factor.variables[0]
        evidence = factor.variables[1:]
        values = factor.values.reshape(2, 2**len(evidence))
        return TabularCPD(violation_factor, 2, values, evidence=evidence, evidence_card=[2]*len(evidence))
    elif "V_" in factor0_str and len(factor.variables) == 4:
        factor_name = f"same_reason_{factor0_str.replace('V_', '')[0]}"
        evidence = factor.variables
        values = [1-factor.values.reshape(2**4), factor.values.reshape(2**4)]
        return TabularCPD(factor_name, 2, values, evidence=evidence, evidence_card=[2]*len(evidence))

    elif len(factor.variables) == 1:
        colour_variable = factor.variables[0]

        return TabularCPD(colour_variable, 2, factor.values.reshape((1,2)))


class InferenceType(Enum):
    SearchInference = 1
    BeliefPropagation = 2
    GibbsSampling = 3
    BayesianModelSampler = 4


class SamplingType(Enum):
    Rejection = 1
    LikelihoodWeighted = 2


class PGMPYInference(object):

    def __init__(self, model, inference_type=InferenceType.BeliefPropagation, num_samples=1000,
                 sampling_type=SamplingType.LikelihoodWeighted):
        self.model = copy.deepcopy(model)
        self.evidence = {}
        self.inference = None
        self.scope = set()
        self.inference_type = inference_type
        self.num_samples = num_samples
        self.sampling_type = sampling_type

    def reduce_model(self, evidence):
        model = copy.deepcopy(self.model)
        continuous_factors = [factor for factor in model.factors if isinstance(factor, ContinuousFactor)]

        for var, val in evidence.items():
            for factor in continuous_factors:
                if var in factor.scope() and "F(" in var:  # make sure that we only reduce at this stage for continuous values, let the inference algorithm deal with reducing for binary variables
                    factor.reduce([(var, val)])

        new_model = BayesianModel()

        additional_evidence = {}

        for node in model.factors:
            if isinstance(node, ContinuousFactor):
                if len(node.scope()) == 1:
                    node = TabularCPD(str(node.scope()[0]), 2, [[node.assignment(0), node.assignment(1)]])
            else:
                node = to_CPD(node)

            var = node.variable
            for v in node.scope():
                if var != v:
                    new_model.add_edge(str(v), str(var))

            if "same_reason" in var:
                additional_evidence[var] = 1

            new_model.add_nodes_from([str(n) for n in node.scope()])
            new_model.add_cpds(node)
        return new_model, additional_evidence

    def infer(self, evidence, new_evidence):

        evidence.update(new_evidence)

        new_model, additional_evidence = self.reduce_model(evidence)

        try:
            if self.inference_type == InferenceType.BeliefPropagation:
                inference = BeliefPropagation(new_model)
            elif self.inference_type == InferenceType.GibbsSampling:
                inference = GibbsSampling(new_model)
            elif self.inference_type == InferenceType.BayesianModelSampler:
                inference = BayesianModelSampling(new_model)
        except Exception as e:
            # for factor in new_model.factors:
            #     print(factor)
            raise e

        self.evidence = {var: val for (var, val) in evidence.items() if "F(" not in var}
        self.evidence.update(additional_evidence)
        self.inference = inference
        self.scope = get_scope(new_model)

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
                pass
                # output[variable] = None
            else:
                vars_to_query.append(variable)
                vals_to_query.append(val)

        if self.inference_type == InferenceType.BayesianModelSampler:
            evidence = [State(var=key, state=val) for key, val in self.evidence.items() if key in get_scope(self.model)]
        else:
            evidence = {key: val for key, val in self.evidence.items() if key in get_scope(self.model)}

        if self.inference_type == InferenceType.BeliefPropagation:
            q = self.inference.query(variables=vars_to_query, evidence=evidence)
            output.update({var: q[var].values[val] for var, val in zip(vars_to_query, vals_to_query)})
        else:
            if self.sampling_type == SamplingType.Rejection:
                sample = self.inference.rejection_sample(evidence=evidence, size=self.num_samples)
                q = sample.sum(axis=0)/self.num_samples

            elif self.sampling_type == SamplingType.LikelihoodWeighted:
                sample = self.inference.likelihood_weighted_sample(evidence=evidence, size=self.num_samples)

                filtered_sample = sample[sample._weight > 0]
                i = 1
                while len(filtered_sample) < 100 and i < 5:
                    sample = self.inference.likelihood_weighted_sample(evidence=evidence, size=self.num_samples*10**min(i, 3))

                    filtered_sample = sample[sample._weight > 0]
                    i += 1

                q = filtered_sample.sum(axis=0)/len(filtered_sample)

            for var, val in zip(vars_to_query, vals_to_query):
                output[var] = q[var] if val == 1 else 1-q[var]

        return output


def get_scope(model):
    scope = set()
    if isinstance(model, FactorGraph):
        for variable in model.factors:
            scope = scope.union(variable.scope())
    else:
        for cpd in model.cpds:
            scope = scope.union(cpd.scope())
    return scope


def reduce_model(model, evidence):
    model = copy.deepcopy(model)
    # continuous_factors = [factor for factor in model.factors if isinstance(factor, ContinuousFactor)]

    for var, val in evidence.items():
        for factor in model.factors:
            if var in factor.scope():  # and "F(" in var:  # make sure that we only reduce at this stage for continuous values, let the inference algorithm deal with reducing for binary variables
                factor.reduce([(var, val)])

    new_model = FactorGraph()

    additional_evidence = {}

    for node in model.factors:
        if isinstance(node, ContinuousFactor):
            if len(node.scope()) == 1:
                node = TabularCPD(str(node.scope()[0]), 2, [[node.assignment(0), node.assignment(1)]]).to_factor()
        if len(node.scope()) == 0:
            continue

        #         try:
        #             var = node.variable
        #         except:
        #             print(node.scope())
        #         for v in node.scope():
        #             if var != v:
        #                 new_model.add_edge(str(v), str(var))

        #         if "same_reason" in var:
        #             additional_evidence[var] = 1

        new_model.add_nodes_from([str(n) for n in node.scope()])
        new_model.add_factors(node)
    return new_model, additional_evidence


def get_non_zero_states(model, data):
    new_model, additional_evidence = reduce_model(model, data)
    data.update(additional_evidence)
    return _get_non_zero_states(new_model)


def _get_non_zero_states(model):
    factors = model.cpds if isinstance(model, BayesianModel) else model.factors
    phi = reduce(lambda x, y: x * y, factors)

    # find all non-zero probability states from the relevant factors
    variables = phi.scope()
    nonzero = np.nonzero(phi.values)  # the indexes of non-zero values in phi
    assert (len(variables) == len(nonzero))
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
        new_beams.append((phi.values[a], trace))

    return new_beams


class SortableDict(dict):

    def __lt__(self, other):
        return True


def normalise_beam(beams):
    norm_val = sum([val for val, _ in beams])
    return [(val / norm_val, b) for val, b in beams]


def evaluate_factor(data, factor):
    factor_data = {v: data[v] for v in factor.scope()}
    return factor.p(factor_data)


def product(x):
    return reduce(mul, x, 1)


def evaluate_model(data, model):
    return product([evaluate_factor(data, factor) for factor in model.factors])


def beams_compatible(beam1, beam2):
    intersecting_variables = set(beam1.keys()).intersection(beam2.keys())
    return all([beam1[var] == beam2[var] for var in intersecting_variables])


class ApproximateSearchInference(object):

    def __init__(self, beam_size, models):
        self.beam_size = beam_size
        self.beams = [(1, SortableDict())]
        self.models = models
        self.previous_inference_time = 0

    def update_beams(self, new_beams, data, models):
        model = build_combined_model(models)

        q = []

        for v1, old_beam in self.beams:
            for v2, new_beam in new_beams:

                if beams_compatible(old_beam, new_beam):
                    combined_beam = SortableDict()
                    combined_beam.update(old_beam)
                    combined_beam.update(new_beam)
                    combined_beam.update(data)

                    beam_value = evaluate_model(combined_beam, model)
                    heapq.heappush(q, (beam_value, combined_beam))
                    if len(q) > self.beam_size > 0:
                        heapq.heappop(q)
        self.beams = normalise_beam(q)

    def clamp(self, evidence):
        for var, val in evidence.items():
            self.beams = normalise_beam([(p, beam) for p, beam in self.beams if beam[var] == val])

    def infer(self, evidence, new_evidence):

        if self.previous_inference_time == len(self.models):
            self.clamp(new_evidence)
        else:
            evidence.update(new_evidence)
            beams = get_non_zero_states(self.models[-1], evidence)
            self.update_beams(beams, evidence, self.models[:-1])
            self.previous_inference_time = len(self.models)

    def p(self, var, val):
        try:
            return sum([p for (p, beam) in self.beams if beam[var] == val])
        except KeyError:
            return None

    def query(self, variables, values=None):
        if values is None:
            values = [1] * len(variables)
        return {var: self.p(var, val) for var, val in zip(variables, values)}


class SearchInference(object):

    def __init__(self, model, beam_size=-1):
        self.model = model
        self.beam = [({}, 1)]
        self.beam_size = beam_size
        self.norm = 0

    def infer(self, evidence, new_evidence):

        overlapping_vars = False
        if self.beam:
            remains = set(new_evidence.keys()) - set(evidence.keys())
            overlapping_vars = remains.intersection(set(self.beam[0][0].keys()))

        if overlapping_vars:
            self.clamp({o: new_evidence[o] for o in overlapping_vars})

        evidence.update(new_evidence)

        variables = set(self.model.nodes()) - set(evidence.keys()) - set(self.model.factors) - set(self.beam[0][0].keys())
        if len(variables) == 0:
            return
        # add factor to set of factors if a at least one variable is included in the factors scope (i.e. ignore irrelevant factors
        factors = [factor for factor in self.model.factors if len(set(factor.scope()) - variables) < len(factor.scope())]

        # Reduce factors givent he evidence
        for var, val in evidence.items():
            for factor in factors:
                if var in factor.scope():
                    factor.reduce([(var, val)])

        # Make continuous factors discrete
        updated_factors = []
        for factor in factors:
            if isinstance(factor, ContinuousFactor) and len(factor.scope()) == 1:
                factor = DiscreteFactor(factor.scope(), [2], [factor.assignment(0), factor.assignment(1)])
            updated_factors.append(factor)

        # Combine everything into one big factor
        phi = reduce(lambda x, y: x * y, updated_factors)

        # find all non-zero probability states from the relevant factors
        variables = phi.scope()
        nonzero = np.nonzero(phi.values)  # the indexes of non-zero values in phi
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

                overlap = set(beam.keys()).intersection(set(new_beam.keys()))

                if all([beam[var] == new_beam[var] for var in overlap]):
                    out_beam = beam.copy()
                    out_beam.update(new_beam)
                    out_beam.update(evidence)
                    p = val * new_val
                    out_beams.append((out_beam, p))
                    norm += p

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
    scope1 = [x for x in get_scope(model1) if "F(" not in str(x)]
    scope2 = [x for x in get_scope(model2) if "F(" not in str(x)]
    return is_overlap(scope1, scope2)


def factors_are_equal(factor1, factor2):
    if type(factor1) != type(factor2):
        return False
    scope1 = factor1.scope()
    scope2 = factor2.scope()

    return len(scope1) == len(scope2) and all([n1 == n2 for n1, n2 in zip(scope1, scope2)])


def is_relevant(model, other_models):
    if model in other_models:
        return False
    return any([is_scope_overlap(model, m) for m in other_models])


def combine_models(model1, model2):
    #assert(is_scope_overlap(model1, model2))

    for node in model2.factors:
        if node not in model1.factors:
            model1.add_nodes_from(node.scope())
            model1.add_factors(node)
            model1.add_nodes_from([node])
    model1.factors = list(set(model1.factors))

    return model1


def create_new_model(nodes, factor):
    new_model = FactorGraph()
    new_model.add_nodes_from([str(n) for n in nodes] + [str(factor)])
    new_model.add_factors(factor)
    return new_model


def build_combined_model(relevant_models):
    new_model = FactorGraph()
    for model in relevant_models:
        combine_models(new_model, model)
    return new_model


class PGMModel(object):

    def __init__(self, inference_type=InferenceType.SearchInference, sampling_type=SamplingType.LikelihoodWeighted, max_inference_size=-1, max_beam_size=0):
        self.inference_type = inference_type
        self.sampling_type = sampling_type
        self.max_inference_size = max_inference_size
        self.max_beam_size = max_beam_size
        self.reset()

    def reset(self):
        """ Resets variables which need to be updated for each new scenario
        """
        self.model = FactorGraph()

        self.ordered_models = []

        if self.inference_type == InferenceType.SearchInference:
            self.inference = ApproximateSearchInference(self.max_beam_size, self.ordered_models)

        # elif self.inference_type == InferenceType.SearchInference:
        #     self.search_inference = SearchInference(self.model)
        else:
            self.inference = PGMPYInference(self.model, inference_type=self.inference_type, sampling_type=self.sampling_type)

        self.observed = {}

        self.models = []
        self.model_nodes = set()

    def add_new_model(self):
        if self.max_inference_size > 0 or isinstance(self.inference, ApproximateSearchInference):
            model = FactorGraph()
            self.ordered_models.append(model)
            return model
        else:
            return None

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

    def test_models(self):
        for model in self.models:
            variable_nodes = set([x for factor in model.factors for x in factor.scope()])
            factor_nodes = set(model.nodes()) - variable_nodes

            assert(len(factor_nodes) == len(model.factors))

    def reduce_models(self):

        for i, model1 in enumerate(self.models):
            for j, model2 in enumerate(self.models):
                if i != j:
                    if is_scope_overlap(model1, model2):
                        new_combined = combine_models(model1, model2)
                        self.models.remove(model1)
                        self.models.remove(model2)
                        new_combined.factors = list(set(new_combined.factors))
                        self.models.append(new_combined)

                        return self.reduce_models()

    def add_factor(self, nodes, factor, model=None):

        if model is not None:
            self.model_nodes.add(factor)
            new_model = create_new_model(nodes, factor)
            combine_models(model, new_model)

        if self.inference_type != InferenceType.SearchInference:

            if any([factors_are_equal(factor, factor2) for factor2 in self.model_nodes]):
                return

            self.model_nodes.add(factor)

            new_model = create_new_model(nodes, factor)
            combined = False

            for model in self.models:
                if is_scope_overlap(model, new_model):
                    updated_model = combine_models(model, new_model)
                    model.factors = list(set(model.factors))
                    self.reduce_models()
                    combined = True
            if not combined:
                self.models.append(new_model)
        else:
            node_filter = lambda x: filter(lambda y: y not in self.model.nodes(), x)
            self.model.add_nodes_from(node_filter(set([str(n) for n in nodes] + [str(factor)])))
            self.model.add_factors(factor)

    def observe(self, observable=None):

        if self.inference_type == InferenceType.SearchInference:
            start = time.time()
            self.inference.infer(self.observed, observable)
            self.observed.update(observable)
            end = time.time()
            return end - start
        else:
            self.observed.update(observable)

    def query(self, variables, values=None):

        if self.inference_type == InferenceType.SearchInference:
            return self.inference.query(variables, values=values)

        elif self.max_inference_size > 0:

            reverse_models = defaultdict(list)
            for variable in variables:
                latest_relevant_model = [model for model in self.ordered_models if variable in get_scope(model)][-1]

                reverse_models[latest_relevant_model].append(variable)

            output = {}

            for model, variables in reverse_models.items():
                relevant_models = [model]
                for i in range(self.max_inference_size - 1):
                    try:
                        next_relevant_model = [m for m in self.ordered_models if is_relevant(m, relevant_models)][-1]
                        relevant_models.append(next_relevant_model)
                    except IndexError:
                        break

                inference_model = build_combined_model(relevant_models)
                inference = PGMPYInference(inference_model, inference_type=self.inference_type,
                                           sampling_type=self.sampling_type)
                inference.infer({}, self.observed)

                output.update(inference.query(variables, values=values))
            return output

        else:
            output = {}
            for model in self.models:
                if any([variable in get_scope(model) for variable in variables]):
                    inference = PGMPYInference(model, inference_type=self.inference_type,
                                               sampling_type=self.sampling_type)
                    inference.infer({}, self.observed)

                    output.update(inference.query(variables, values=values))

            return output


class CorrectionPGMModel(PGMModel):

    def __init__(self, **kwargs):
        super(CorrectionPGMModel, self).__init__(**kwargs)
        self.known_rules = set()
        self.rule_priors = {}
        self.colours = {}
        # self.reset()

    def reset(self):
        super(CorrectionPGMModel, self).reset()
        self.colour_variables = []

    def add_cm(self, cm, block, model=None):

        red_rgb = f'F({block})'
        red_o1 = f'{cm.name}({block})'
        self.colours[cm.name] = cm
        self.colour_variables.append(red_o1)
        if 't' in block:
            self.model.add_nodes_from([red_o1])
            red_cm_factor = TabularCPD(red_o1, 2, [[1, 0]])
            self.add_factor([red_o1], red_cm_factor, model)
            return red_o1
        if red_o1 not in self.model.nodes():
            red_distribution = lambda rgb, c: cm.p(c, rgb)
            red_evidence = [red_rgb, red_o1]
            red_cm_factor = ContinuousFactor(red_evidence, red_distribution)
            self.add_factor(red_evidence + [red_cm_factor], red_cm_factor, model)
        return red_o1

    def get_rule_prior(self, rule):
        try:
            return self.rule_priors[rule]
        except KeyError:
            return 0.1

    def add_prior(self, rule, model=None):

        rule_prior_value = self.get_rule_prior(rule)

        rule_prior = DiscreteFactor([str(rule)], [2], [1-rule_prior_value, rule_prior_value])
        self.add_factor([str(rule)], rule_prior, model)

    def add_no_correction(self, args, time, rules, model=None):
        if not rules:
            return
        o1, o2 = args[:2]

        violations = []

        print("no correction", args, time, rules)

        for rule in rules:

            if isinstance(rule, RedOnBlueRule):
                red = rule.c1
                blue = rule.c2
                colour_models = self.add_cms(self.colours[red], self.colours[blue], [o1, o2],
                                             correction_type=CorrectionType.TOWER, model=model)

                Vrule = self.add_violation_factor(rule, time, colour_models, correction_type=CorrectionType.TOWER, model=model)

                violations.append(Vrule)
            elif isinstance(rule, ColourCountRule):
                return []
            else:
                raise ValueError(f"Unexpected rule type {type(rule)}")

        self.add_correction_factor(violations, time, model=model)

        return violations

    # def add_no_table_correction(self, args, time):
    #     if not self.known_rules:
    #         return
    #
    #     model = self.add_new_model()
    #
    #     violations = []
    #
    #     for rule in self.known_rules:
    #         red = rule.c1
    #         blue = rule.c2
    #         violations.append(self.add_table_violation(rule, self.colours[red], self.colours[blue], args, time, model=model))
    #
    #     self.add_correction_factor(violations, time)
    #
    #     return violations

    def add_violation_factor(self, rule, time, colours, correction_type=CorrectionType.TABLE, table_empty=False, model=None):
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

        evidence = colours + [str(rule)]

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

        self.add_factor([violated_rule_factor_name, rule, rule_violated_factor] + evidence, rule_violated_factor, model=model)
        return violated_rule_factor_name

    def add_correction_factor(self, violations, time, model=None):
        """
        :param violations:
        :param time:
        :return:
        """
        corr = f'corr_{time}'
        correction_table = variable_or_CPD(len(violations))
        correction = TabularCPD(corr, 2, correction_table, evidence=violations, evidence_card=[2]*len(violations))
        correction_factor = correction.to_factor()
        self.add_factor([corr, correction_factor] + violations, correction_factor, model=model)
        return corr

    def add_cc_and_rob(self, colour_count: ColourCountRule, red_on_blue_options: list, red_cm: KDEColourModel,
                       blue_cm: KDEColourModel, objects_in_tower: list, top_object:str, time:int):

        model = self.add_new_model()

        objects_in_tower = objects_in_tower.copy()
        self.known_rules.add(colour_count)
        red_on_blue1, red_on_blue2 = red_on_blue_options
        self.known_rules.add(red_on_blue1)
        self.known_rules.add(red_on_blue2)

        objects_in_tower.remove(top_object)
        colours_in_tower = [self.add_cm(red_cm, obj, model) for obj in objects_in_tower] + [self.add_cm(blue_cm, top_object, model)]
        violations = self.add_joint_violations(colour_count, red_on_blue_options, time, colours_in_tower, model)
        self.add_correction_factor(violations, time, model)
        return violations

    def add_joint_violations(self, colour_count: ColourCountRule, red_on_blue_options: list, time: int,
                             colours_in_tower: list, model: FactorGraph):
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

        self.add_factor([violated_rule_factor_name1, rule_violated_factor1, red_on_blue_options[0], colour_count], rule_violated_factor1, model)
        self.add_factor([violated_rule_factor_name2, rule_violated_factor2, red_on_blue_options[1], colour_count], rule_violated_factor2, model)
        return [violated_rule_factor_name1, violated_rule_factor_name2]

    def add_colour_count_correction(self, rule: ColourCountRule, cm: KDEColourModel, objects_in_tower: list, time: int, model: FactorGraph):
        self.known_rules.add(rule)
        colour_variables = []
        for obj in objects_in_tower:
            colour_variables.append(self.add_cm(cm, obj))
        violations = [self.add_violation_factor(rule, time, colour_variables)]
        self.add_correction_factor(violations, time, model)
        return violations

    def extend_model(self, rules, red_cm, blue_cm, args, time, correction_type, table_empty=False):

        model = self.add_new_model()

        colour_models = self.add_cms(red_cm, blue_cm, args, correction_type=correction_type, model=model)

        if table_empty:
            colour_models.append(self.add_cm(blue_cm, args[0], model))

        r1, r2 = rules

        self.add_prior(r1, model)
        self.add_prior(r2, model)

        self.known_rules.add(r1)
        self.known_rules.add(r2)

        violated_r1 = self.add_violation_factor(r1, time, colour_models,
                                                correction_type=correction_type, table_empty=table_empty, model=model)
        violated_r2 = self.add_violation_factor(r2, time, colour_models,
                                                correction_type=correction_type, table_empty=table_empty, model=model)
        self.add_correction_factor([violated_r1, violated_r2], time, model)

        return violated_r1, violated_r2

    def add_cms(self, red_cm, blue_cm, args, correction_type=CorrectionType.TABLE, model=None):
        o1, o2 = args[:2]
        red_o1 = self.add_cm(red_cm, o1, model)
        blue_o2 = self.add_cm(blue_cm, o2, model)

        if correction_type == CorrectionType.TABLE:
            o3 = args[2]
            red_o3 = self.add_cm(red_cm, o3, model)
            blue_o3 = self.add_cm(blue_cm, o3, model)
            return [red_o1, blue_o2, red_o3, blue_o3]
        else:
            return [red_o1, blue_o2]

    def add_same_reason(self, current_violations, previous_violations, model=None):

        evidence = []
        for c, p in zip(current_violations, previous_violations):
            evidence.extend([c, p])

        # evidence = list(current_violations) + list(previous_violations)
        t = equals_CPD(current_violations, previous_violations)

        f = DiscreteFactor(evidence, [2]*len(evidence), t)
        self.add_factor(evidence, f, model)

    def get_rule_probs(self, update_prior=False):
        rules = [str(rule) for rule in self.known_rules]
        q = self.query(rules)
        # for rule, p in q.items():
        #     if p is None:
        #         q[rule] = self.rule_priors[rule]

        if update_prior:
            for rule in rules:
                try:
                    self.rule_priors[rule] = q[rule]
                except KeyError:
                    q[rule] = self.rule_priors[rule]
        return q

    def get_colour_predictions(self):
        q = self.query(self.colour_variables)

        return q

    def create_negative_model(self, rule, red_cm, blue_cm, args, time):
        o1, o2 = args[:2]

        model = self.add_new_model()

        red_o1 = self.add_cm(red_cm, o1, model)
        blue_o2 = self.add_cm(blue_cm, o2, model)

        self.add_prior(rule, model)
        self.known_rules.add(rule)
        rule_violation_CPD = rule.generate_CPD()

        correction_table = variable_or_CPD(1)

        Vrule = f'V_{time}({rule})'
        corr = f'corr_{time}'

        rule_violated = TabularCPD(Vrule, 2, rule_violation_CPD, evidence=[red_o1, blue_o2, rule], evidence_card=[2, 2, 2])
        rule_violated_factor = rule_violated.to_factor()
        self.add_factor([Vrule, rule, rule_violated_factor], rule_violated_factor, model)

        correction = TabularCPD(corr, 2, correction_table, evidence=[Vrule], evidence_card=[2])
        correction_factor = correction.to_factor()
        self.add_factor([corr, correction_factor], correction_factor, model)

        return [Vrule]

    def create_table_neg_model(self, rule, red_cm, blue_cm, args, time):
        model = self.add_new_model()

        o1, o2, o3 = args[:3]
        blue_o1 = self.add_cm(blue_cm, o1, model)
        red_o3 = self.add_cm(red_cm, o3, model)

        self.add_prior(rule, model)

        self.known_rules.add(rule)

        rule_cpd = generate_neg_table_cpd()
        correction_table = variable_or_CPD(1)

        Vrule = f'V_{time}({rule})'
        corr = f'corr_{time}'

        rule_evidence = [rule, blue_o1, red_o3]
        rule_violated = TabularCPD(Vrule, 2, rule_cpd, evidence=rule_evidence, evidence_card=[2,2,2,2,2])
        self.add_factor(rule_evidence + [Vrule], rule_violated.to_factor(), model)

        correction = TabularCPD(corr, 2, correction_table, evidence=[Vrule], evidence_card=[2])
        self.add_factor([Vrule, corr], correction.to_factor(), model)

        return [Vrule]

    def create_uncertain_table_model(self, rules, red_cm, blue_cm, args,
                                     objects_in_tower, time):

        model = self.add_new_model()

        self.known_rules = self.known_rules.union(rules)
        rule1, rule2 = rules
        self.add_prior(rule1, model)
        self.add_prior(rule2, model)

        reds = [self.add_cm(red_cm, obj, model) for obj in objects_in_tower]
        blues = [self.add_cm(blue_cm, obj, model) for obj in objects_in_tower[1:]]
        red_o3 = self.add_cm(red_cm, args[-1], model)
        blue_o3 = self.add_cm(blue_cm, args[-1], model)

        colour_variables = reds + blues + [red_o3, blue_o3]

        violations = [self.add_violation_factor(rule1, time, colour_variables, correction_type=CorrectionType.UNCERTAIN_TABLE, model=model),
                      self.add_violation_factor(rule2, time, colour_variables, correction_type=CorrectionType.UNCERTAIN_TABLE, model=model)]
        self.add_correction_factor(violations, time, model)
        return violations


