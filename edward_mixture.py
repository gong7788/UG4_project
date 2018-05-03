from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.cm as cm
import numpy as np
import six
import tensorflow as tf

from edward.models import (
    Categorical, Dirichlet, Empirical, InverseGamma,
    MultivariateNormalDiag, Normal, ParamMixture, Beta, Bernoulli, Mixture, MixtureSameFamily, PointMass)


import data
import random

data_dict = {c:data.generate_data(c, 2) for c in ['red', 'green', 'blue', 'yellow']}
objs = {}
n = 2
for j, c in enumerate(['red', 'green', 'blue', 'yellow']):
    for i in range(n):
        objs["o{}".format(j*n + i)] = (c, data_dict[c][i])


def check_rule(o1, o2):
    c1, d1 = o1
    c2, d2 = o2

    if c1 == 'red':
         if c2 != 'blue':
            return 'correction'

    return 'no correction'

def correction_to_int(corr_string):
    return int(corr_string == 'correction')

def colour_to_int(corr_string, colour):
    return int(corr_string == colour)

def my_sample(objs, n=10):
    cs = []
    red = []
    blue = []

    for i in range(n):
        o1 = random.sample(objs.keys(), k=1)[0]
        o2 = random.sample(objs.keys(), k=1)[0]
        while o1 == o2:
            o2 = random.sample(objs.keys(), k=1)[0]
        c = correction_to_int(check_rule(objs[o1], objs[o2]))
        f_red = objs[o1][1]
        f_blue = objs[o2][1]
        cs.append(c)
        red.append(f_red)
        blue.append(f_blue)
    return cs, red, blue

def my_colour_sample(objs, n=10):
    cs = []
    red = []

    for i in range(n):
        o1 = random.sample(objs.keys(), k=1)[0]
        c = objs[o1][0]
        f_red = objs[o1][1]
        cs.append(c)
        red.append(f_red)
    return cs, red

colours, c_data = my_colour_sample(objs)
colours = list(map(lambda x: colour_to_int(x, 'blue'), colours))
colours = np.array(colours)
c_data = np.array(c_data)


# Define the probability Model
blue_prior = Dirichlet(tf.ones(2))
D=3

blue_mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=2)
blue_sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=2)
f_blue = ParamMixture(blue_prior, {'loc': blue_mu, 'scale_diag': tf.sqrt(blue_sigmasq)},
                 MultivariateNormalDiag, sample_shape=10)
blue = f_blue.cat

print(f_blue.)
# Inference Time
# Define the proposal distributions for KLqp learning
qblue_prior = Dirichlet(tf.get_variable("qblue_prior/params", [2]))#, tf.get_variable("qblue_prior2/params", [1]))
qblue_mu = Normal(tf.get_variable("qblue_mu/loc", [2, D]), tf.nn.softplus(tf.get_variable("qblue_mu/scale", [2, D])))
qblue_sigmasq = InverseGamma(tf.get_variable("qblue_sigm/loc", [2, D]), tf.get_variable("qblue_sigm/scale", [2, D]))

#Create the inference object
inference = ed.KLqp({blue_prior: qblue_prior, blue_mu:qblue_mu, blue_sigmasq: qblue_sigmasq},
                     data={blue: colours, f_blue: c_data})
inference.initialize()
sess = ed.get_session()
tf.global_variables_initializer().run()
inference.run(n_iter=500, n_print=100, n_samples=10)
