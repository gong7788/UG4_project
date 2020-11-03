import random

from gently.environment import RuleGroup, ConceptGroup
from gently.util import get_subsets


def basic_correction(left_side, right_side):
    return f"no, when you see {' and '.join(left_side.labels[0])} you should do it {' and '.join(right_side.labels[0])}"


def individual_failures_correction(failures):
    subsets = get_subsets(failures)
    for subset in subsets:
        yield f"no, {' and '.join(subset)}".strip(" ,")


def contrast_correction(overlap, left_side):
    if len(overlap) >= 1:
        subsets = get_subsets(list(overlap))
        for subset in subsets:
            if subset != []:
                yield f"no, do it like when you see {' and '.join(left_side.labels[0])} but {' and '.join(subset)}"


class TeacherConcept(object):

    def __init__(self, name, dimension, lower_bound, upper_bound):
        self.name = name
        self.dimension = dimension
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def check_behaviour(self, behaviour):
        val = behaviour.get_value(self.dimension)
        return self.upper_bound >= val >= self.lower_bound


class TeacherAgent(object):

    def __init__(self, rules, concepts):
        self.rule_group = RuleGroup(rules)
        self.concepts = {concept.name: concept for concept in concepts}
        self.corrected_rules = set()

    def get_feedback(self, behaviour_curve, context):

        # concepts = self.rule_group.get_behaviour_concepts(context)

        faults = []

        for rule in self.rule_group.rules:
            if rule.evaluate(context):
                faultless = all \
                    ([self.concepts[concept].check_behaviour(behaviour_curve) for concept in rule.right_side.labels[0]])
                if not faultless:

                    faults.append(rule)

        corrections = []
        for fault in faults:
            corrections += self.get_correction(fault, context, behaviour_curve)

        print(corrections)

        return self.select_correction(corrections)

    def select_correction(self, corrections):
        try:
            correction_sentence, rule = random.choice(corrections)
        except IndexError:
            return "yes"

        self.corrected_rules.add(rule)
        return correction_sentence

    def get_correction(self, rule, context, behaviour_curve):

        for group in rule.left_side.labels:
            concept_group = ConceptGroup([group])
            if concept_group.evaluate(context):
                yield (basic_correction(concept_group, rule.right_side), rule)


                if rule in self.corrected_rules:

                    failures = [label for label in rule.right_side.labels[0] if not self.concepts[label].check_behaviour(behaviour_curve)]
                    for sentence in individual_failures_correction(failures):
                        yield (sentence, rule)

        for rule2 in self.corrected_rules:
            r1_concepts = rule.right_side.labels[0]
            r2_concepts = rule2.right_side.labels[0]

            if rule2 == rule:
                continue


            overlap = set(r1_concepts).intersection(r2_concepts)

            for sentence in contrast_correction(overlap, rule2.left_side):
                yield (sentence, rule)
