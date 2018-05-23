from collections import namedtuple
import numpy as np
from scipy.stats import norm
import copy

ColourModel = namedtuple('ColourModel', ['name', 'mu', 'sigma'])
rule_belief = (0.5, 0.5)



class ColourModel(object):

    def __init__(self, name, mu0=np.array([0.5, 0.5, 0.5]),
                 alpha0=np.array([1., 1., 1.]),
                 beta0=np.array([1., 1., 1.]),
                 gamma=np.array(10),
                 p_c=np.array([0.5, 0.5]),
                 mu1=np.array([0.5, 0.5, 0.5]),
                 alpha1=np.array([1., 1., 1.]),
                 beta1=np.array([1., 1., 1.])):

        self.name = name
        self.mu0 = mu0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.mu1 = mu0
        self.alpha1 = alpha0
        self.beta1 = beta0
        self.sigma0 = beta0/(alpha0 + 3/2)
        self.sigma1 = beta1/(alpha1 + 3/2)
        self.p_c = p_c
        self.gamma = gamma


    def p(self, c, fx):
        p1 = self.p_c[1] * np.sum(norm.pdf(fx, loc=self.mu0, scale=self.sigma0))
        p0 = self.p_c[0] * np.sum(norm.pdf(fx, loc=self.mu1, scale=self.sigma1))
        return [p0, p1][c]/(p1 + p0)

    def update(self, fx, w):
        asquigle = 1/self.gamma + 1*w
        bsquigle = self.mu0/self.gamma + fx*w
        csquigle = self.mu0**2/self.gamma + (fx*w)**2
        mu_post = bsquigle/asquigle
        alpha = self.alpha0 + 1/2
        beta = self.beta0 +  0.5*(csquigle - bsquigle**2/asquigle)
        self.mu0 = mu_post
        self.alpha0 = alpha
        self.beta0 = beta
        return mu_post, alpha, beta
        #updated_mu = (w*fx*self.sigma_prior + self.mu * self.sigma)/(w*self.sigma_prior + self.sigma)
        #return updated_mu

# class ColourModel(object):
#
#     def __init__(self, name, mu=np.array([0.5, 0.5, 0.5]),
#                  sigma=np.array([0.1, 0.1, 0.1]),
#                  p_c=np.array([0.5, 0.5]),
#                  mu_nill = np.array([0.5, 0.5, 0.5]),
#                  sigma_nill = np.array([10., 10., 10.])):
#         self.name = name
#         self.mu = mu
#         self.sigma = sigma
#         self.mu_nill = mu_nill
#         self.sigma_nill = sigma_nill
#         self.p_c = p_c
#         self.sigma_prior = 1.
#
#
#     def p(self, c, fx):
#         p1 = self.p_c[1] * np.sum(norm.pdf(fx, loc=self.mu, scale=self.sigma))
#         p0 = self.p_c[0] * np.sum(norm.pdf(fx, loc=self.mu_nill, scale=self.sigma_nill))
#         return [p0, p1][c]/(p1 + p0)
#
#     def update_mu(self, fx, w):
#         fx = np.array(fx)
#         w = np.array(w)
#         updated_mu = (w*fx*self.sigma_prior + self.mu * self.sigma)/(w*self.sigma_prior + self.sigma)
#         self.mu = updated_mu
          #return updated_mu
#



class CorrectionModel(object):
    def __init__(self, rules, c1, c2, rule_belief=(0.5, 0.5)):
        self.rules = rules
        self.c1 = c1
        self.c2 = c2
        self.rule_belief = rule_belief
        self.variables = [c1.name, c2.name, 'r']

    # data and hidden seem to be some kind of dictionary while hidden seems to be a list of keys
    def p(self, data, visible):
        hidden = set(self.variables) - visible.keys()
        if not hidden:
            #print('reached here with hidden={}'.format(hidden))
            return (self.rule_belief[visible['r']] *
                    self.c1.p(visible[self.c1.name], data[self.c1.name]) *
                    self.c2.p(visible[self.c2.name], data[self.c2.name]) *
                    self.evaluate_correction(visible))
        else:
            h = hidden.pop()
            #print('adding {} to visible'.format(h))
            visible[h] = 0
            v0 = self.p(data, copy.copy(visible))
            #print('finished recursion for {}=0 with value {}'.format(h, v0))
            visible[h] = 1
            v1 = self.p(data, copy.copy(visible))
            #print('finished recursion for {}=1 with value {}'.format(h, v1))
            return v0 + v1

    def p_no_corr(self, data, visible):
        hidden = set(self.variables) - visible.keys()
        if not hidden:
            return (self.rule_belief[visible['r']] *
                    self.c1.p(visible[self.c1.name], data[self.c1.name]) *
                    self.c2.p(visible[self.c2.name], data[self.c2.name]) *
                    (1 - self.evaluate_correction(visible)))
        else:
            h = hidden.pop()
            #print('adding {} to visible'.format(h))
            visible[h] = 0
            v0 = self.p_no_corr(data, copy.copy(visible))
            #print('finished recursion for {}=0 with value {}'.format(h, v0))
            visible[h] = 1
            v1 = self.p_no_corr(data, copy.copy(visible))
            #print('finished recursion for {}=1 with value {}'.format(h, v1))
            return v0 + v1



    def p_r(self, r, data, visible={}):
        v0 = copy.copy(visible)
        v0.update({'r':0})
        r0 = self.p(data, visible=v0)
        v1 = copy.copy(visible)
        v1.update({'r':1})
        r1 = self.p(data, visible=v1)
        eta = r0 + r1
        return [r0, r1][r]/eta

    def update_belief_r(self, data,visible={}):
        r0 = self.p_r(0, data, visible=visible.copy())
        r1 = self.p_r(1, data, visible=visible.copy())
        self.rule_belief = (r0, r1)
        return self.rule_belief

    def update_c(self, data):
        self.c1.update(data[self.c1.name], self.rule_belief[0])
        self.c2.update(data[self.c2.name], self.rule_belief[1])

    def update_c_no_corr(self, data):
        w1 = self.p_no_corr(data, visible={self.c1.name:1})
        w2 = self.p_no_corr(data, visible={self.c2.name:1})
        self.c1.update(data[self.c1.name], w1)
        self.c2.update(data[self.c2.name], w2)

    def update_model(self, data):
        self.update_belief_r(data)
        self.update_c(data)

    def evaluate_correction(self, visible):
        # r=0: \forall x. y. c1(x) & on(x,y) -> c2(y). => on(x,y) c1(x) -c2(y)
        # r=1 \forall x.y. c2(y) & on(x,y) -> c1(x). => on(x, y) -c1(x) c2(y)
        rule0 = visible['r'] == 0 and visible[self.c1.name] == 1 and visible[self.c2.name] == 0
        rule1 = visible['r'] == 1 and visible[self.c1.name] == 0 and visible[self.c2.name] == 1
        return float(rule0 or rule1)


class TableCorrectionModel(CorrectionModel):
    def __init__(self, rules, c1, c2, rule_belief=(0.5, 0.5)):
        self.rules = rules
        self.c1 = c1
        self.c2 = c2
        self.c3 = ColourModel('{}/{}'.format(c1.name, c2.name),
                              mu = c1.mu, sigma=c1.sigma,
                              mu_nill=c2.mu, sigma_nill=c2.sigma)
        self.rule_belief = rule_belief
        self.variables = ['r', c1.name, c2.name, self.c3.name]

    # data and hidden seem to be some kind of dictionary while hidden seems to be a list of keys
    def p(self, data, visible):
        hidden = set(self.variables) - visible.keys()
        if not hidden:
            #print('reached here with hidden={}'.format(hidden))
            return (self.rule_belief[visible['r']] *
                    self.c1.p(visible[self.c1.name], data[self.c1.name]) *
                    self.c2.p(visible[self.c2.name], data[self.c2.name]) *
                    self.evaluate_correction(visible) *
                    self.c3.p(visible[self.c3.name], data[self.c3.name])
                   )
        else:
            h = hidden.pop()
            #print('adding {} to visible'.format(h))
            visible[h] = 0
            v0 = self.p(data, copy.copy(visible))
            #print('finished recursion for {}=0 with value {}'.format(h, v0))
            visible[h] = 1
            v1 = self.p(data, copy.copy(visible))
            #print('finished recursion for {}=1 with value {}'.format(h, v1))
            return v0 + v1

    def p_r(self, r, data, visible={}):
        return super().p_r(r, data.copy(), visible.copy())


    def evaluate_correction(self, visible):
        # r=0: \forall x. y. c1(x) & on(x,y) -> c2(y). => on(x,y) -c1(x) c2(y) c1(z)
        # r=1 \forall x.y. c2(y) & on(x,y) -> c1(x). => on(x, y) c1(x) -c2(y) c2(z)
        rule0 = visible['r'] == 0 and visible[self.c1.name] == 0 and visible[self.c2.name] == 1 and visible[self.c3.name] == 0
        rule1 = visible['r'] == 1 and visible[self.c1.name] == 1 and visible[self.c2.name] == 0 and visible[self.c3.name] == 1
        return float(rule0 or rule1)
