
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
#import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import six
import tensorflow as tf
import data
import random
from edward.models import (
    Categorical, Dirichlet, Empirical, InverseGamma,
    MultivariateNormalDiag, Normal, ParamMixture, Beta, Bernoulli, Mixture, MixtureSameFamily, PointMass)

#plt.style.use('ggplot')



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

colours

T = 500
N = 10
D=3
#bb = Dirichlet(tf.ones(2))
blue_prior = Beta(1.,1.)
#blue = Categorical(probs=[bb[0], bb[0]-1])

blue = Bernoulli(blue_prior)

blue_mu = Normal(tf.zeros(D), tf.ones(D))
blue_sigmasq = InverseGamma(tf.ones(D), tf.ones(D))


blue_c1 = MultivariateNormalDiag(tf.ones(D)*0.5, tf.ones(D)*100)
blue_c2 = MultivariateNormalDiag(blue_mu, blue_sigmasq)

#f_blue = ParamMixture(bb, {'loc': blue_mu, 'scale_diag': tf.sqrt(blue_sigmasq)},
#                 MultivariateNormalDiag,
#                 sample_shape=N)
#blue = f_blue.cat


f_blue = tf.cast(blue, tf.float32) * blue_c2 + tf.cast((1 - blue), tf.float32) * blue_c1

with tf.variable_scope("test2", reuse=tf.AUTO_REUSE):
    qblue_prior = Beta(tf.get_variable("qblue_prior1/params", []), tf.get_variable("qblue_prior2/params", []))
    qblue_mu = Normal(tf.get_variable("qblue_mu/loc", [D]), tf.nn.softplus(tf.get_variable("qblue_mu/scale", [D])))
    qblue_sigmasq = Normal(tf.get_variable("qblue_sigmasq/loc", [D]), tf.nn.softplus(tf.get_variable("qblue_sigmasq/scale", [D])))

#Empirical(tf.get_variable(
        #"qblue_prior/params", [T, 2],
        #initializer=tf.constant_initializer(1)))
#qblue = Empirical(tf.get_variable(
#        "qblue/params", [T, N],
#        initializer=tf.zeros_initializer(), dtype=tf.int32))
#qblue_mu = Empirical(tf.get_variable("qblue_mu/params", [T, 2, D],
#        initializer=tf.zeros_initializer()))
#qblue_sigmasq = Empirical(tf.get_variable("qblue_sigmasq/params", [T, 2, D],
#        initializer=tf.ones_initializer()))

colour_placeholder = tf.placeholder(tf.int32, [])
data_placeholder = tf.placeholder(tf.float32, [D])

inference = ed.KLqp({blue_prior: qblue_prior, blue_mu:qblue_mu, blue_sigmasq: qblue_sigmasq},
                     data={blue: colour_placeholder, f_blue: data_placeholder})
inference.initialize(n_iter=10*500)

sess = ed.get_session()
tf.global_variables_initializer().run()

for i in range(500):
    for c, d in zip(colours, c_data):
        info_dict = inference.update({colour_placeholder: c, data_placeholder:d})
#inference.run(n_iters=500, n_print=100, n_samples=10)
        inference.print_progress(info_dict)


test_c, test_data = my_colour_sample(objs)
test_c = list(map(lambda x: colour_to_int(x, 'blue'), test_c))
test_c = np.array(test_c)
test_data = np.array(test_data, dtype=np.float32)

test_c
print(colours)
print(test_c)
