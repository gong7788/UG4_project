
import numpy as np
from scipy.special import erf
from scipy.stats import norm

from gently.BezierCurves import BehaviourCurve


def get_percentile(mu, sig, x):
    return 2 * erf((x - mu) / (2 ** (1 / 2) * sig)) - 2


def update_mean(m_0, n_0, x):
    m_1 = n_0 * m_0 + x
    m_1 = m_1 / (n_0 + 1)
    return m_1, (n_0 + 1)


def negative_mean(m_0, n_0, x, var=1):
    m_1 = (n_0 + 1) * m_0 - (x - m_0) / abs((x - m_0 + 1e-10) / var ) ** 2
    m_1 = m_1 / (n_0 + 1)
    return m_1, n_0


def negative_variance(sig_0, v_0, n_0, x, m_0):
    sig_0 = sig_0 ** 2
    sig_1 = (v_0 + 1) * sig_0 - n_0 / (n_0 + 1) * 1 / ((m_0 - x) / sig_0) ** 2
    if sig_1 <= 0:
        return sig_0, v_0
    else:
        return sig_1 ** (1/2) / (v_0 + 1), v_0


def negative_variance2(sig_0, v_0, n_0, x, m_0):
    percentile = norm.cdf(x, m_0, sig_0)
    p = 2 * abs(percentile - 0.5)
    return sig_0 * p, v_0


def update_variance(sig_0, v_0, n_0, x, m_0):
    n_1 = n_0 + 1
    v_1 = v_0 + 1
    vsig = v_0 * sig_0 + n_0 / n_1 * (m_0 - x) ** 2
    return vsig / v_1, v_1


def update_params(m_0, n_0, sig_0, v_0, x):
    m_1, n_1 = update_mean(m_0, n_0, x)
    sig_1, v_1 = update_variance(sig_0, v_0, n_0, x, m_0)

    return m_1, n_1, sig_1, v_1


def update_params_negative(m_0, n_0, sig_0, v_0, x, use_var=True, default_var=1):
    m_1, n_1 = negative_mean(m_0, n_0, x, var=sig_0 if use_var else default_var)
    sig_1, v_1 = negative_variance2(sig_0, v_0, n_0, x, m_0)
    return m_1, n_1, sig_1, v_1


def test_good_sample(sample, min_val, max_val):
    min_holds = min_val is None or sample > min_val
    max_holds = max_val is None or sample < max_val
    return min_holds and max_holds



concepts = {
    "left": "curviness",
    "right": "curviness",
    "quickly": "speed",
    "slowly": "speed",
    "gently": "energy",
    "firmly": "energy",
    "speed": "speed",
    "energy": "energy",
    "curviness": "curviness",
}


class Concept(object):

    def __init__(self, name, mean, variance, distribution_type="normal"):

        min_max_vals = {"curviness": (None, None),
                        "speed": (0, 1.5),
                        "energy": (0, 1)}

        self.min_val, self.max_val = min_max_vals[concepts[name]]

        self.concept_type = concepts[name]
        self.name = name
        self.mean = mean
        self.variance = variance
        self.n = 1
        self.v = 1
        self.distribution_type = distribution_type
        if name in ["curviness", "speed", "energy"]:
            self.default_concept = True

    def generate(self):

        if self.distribution_type == "normal":
            v = np.random.normal(loc=self.mean, scale=self.variance)

            while not test_good_sample(v, self.min_val, self.max_val):
                v = np.random.normal(loc=self.mean, scale=self.variance)
            return v
        elif self.distribution_type == "uniform":
            v = np.random.uniform(self.min_val, self.max_val)
            return v

    def update_positive(self, datapoint):

        self.mean, self.n, self.variance, self.v = update_params(
            self.mean, self.n, self.variance, self.v, datapoint)

    def update_negative(self, datapoint):

        self.mean, self.n, self.variance, self.v = update_params_negative(
            self.mean, self.n, self.variance, self.v, datapoint)

    def update(self, datapoint, positive=True):
        if positive:
            self.update_positive(datapoint)
        else:
            self.update_negative(datapoint)

    @staticmethod
    def default_speed_concept():
        return Concept("speed", 0, 100, distribution_type="uniform")

    @staticmethod
    def default_curve_concept():
        return Concept("curviness", 0, 1)

    @staticmethod
    def default_energy_concept():
        return Concept("energy", 0, 100, distribution_type="uniform")

def get_concept_of_type(concepts, concept_type):
    specific_conepts = [concept for concept in concepts if concept.concept_type == concept_type]
    assert (len(specific_conepts) <= 1)
    chosen_concept = specific_conepts[0] if len(specific_conepts) == 1 else None
    return chosen_concept

class Behaviour(object):

    def __init__(self, name, curve_concept=None, speed_concept=None, energy_concept=None):
        self.name = name
        self.curve_concept = curve_concept if curve_concept is not None else Concept.default_curve_concept()
        self.speed_concept = speed_concept if speed_concept is not None else Concept.default_speed_concept()
        self.energy_concept = energy_concept if energy_concept is not None else Concept.default_energy_concept()

    @staticmethod
    def from_list(name, concepts):

        curve_concept = get_concept_of_type(concepts, "curviness")
        energy_concept = get_concept_of_type(concepts, "energy")
        speed_concept = get_concept_of_type(concepts, "speed")
        return Behaviour(name, curve_concept=curve_concept, energy_concept=energy_concept, speed_concept=speed_concept)

    def update_concept(self, behaviour: BehaviourCurve, concept, positive=True):

        if self.curve_concept.name == concept or concept is None:
            self.curve_concept.update(behaviour.curviness, positive)
        if self.speed_concept.name == concept or concept is None:
            self.speed_concept.update(behaviour.speed, positive)
        if self.energy_concept.name == concept or concept is None:
            self.energy_concept.update(behaviour.energy, positive)

    def generate_behaviour(self):
        curviness = self.curve_concept.generate()
        speed = self.speed_concept.generate()
        energy = self.energy_concept.generate()

        return BehaviourCurve(curviness, speed, energy)


