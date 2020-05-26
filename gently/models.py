from functools import reduce

from pgmpy.factors.discrete import TabularCPD

from correctingagent.models.pgmmodels import PGMModel
from correctingagent.util.CPD_generation import binary_flip
from operator import mul
import numpy as np

def product(x):
    return reduce(mul, x, 1)

def check_concept_in_list(concept, correction_content_concepts):
    return concept.name in [c.name for c in correction_content_concepts]

def get_value(concept, truth_value, correction_content_concepts, p=0.7):
    concept_in_list = check_concept_in_list(concept, correction_content_concepts)
    if concept_in_list and truth_value is True:
        return 0
    elif concept_in_list and truth_value is False:
        return 1
    elif truth_value is True:
        return p
    else:
        return 1-p


def generate_correction_cpd(correction_content_concepts, concept_variables, p=0.7):
    flips = binary_flip(len(concept_variables))
    positive_output = []
    negative_output = []
    for flip in flips:
        value = product([get_value(concept, truth_value, correction_content_concepts, p) for truth_value, concept in zip(flip, concept_variables)])
        positive_output.append(value)
        negative_value = int(all(flip))
        negative_output.append(negative_value)
    return np.array([negative_output, positive_output])

class GentlyPGMModel(PGMModel):

    def __init__(self, p=0.7, **kwargs):
        super(GentlyPGMModel, self).__init__(**kwargs)
        self.p = p

    def add_correction(self, correction_content_concepts, concept_variables, time):

        model = self.add_new_model()

        cpd = generate_correction_cpd(correction_content_concepts, concept_variables, p=self.p)
        evidence = [c.name for c in concept_variables]
        cpd_factor = TabularCPD(f"corr_{time}", 2, cpd, evidence=evidence, evidence_card=len(concept_variables))
        self.add_factor(evidence, cpd_factor, model)


