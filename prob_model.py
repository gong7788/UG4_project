from collections import namedtuple
import numpy as np
from scipy.stats import norm
import copy
import matplotlib.pyplot as plt


ColourModel = namedtuple('ColourModel', ['name', 'mu', 'sigma'])
#rule_belief = (0.5, 0.5)





class RuleBelief(object):

    def __init__(self, colours, rule1, rule2, prior=0.001):
        # [[(r1,r2), (r1,-r2)],[(-r1,r2), (-r1, -r2)]]
        self.colours = colours
        self.rule1 = rule1
        self.rule2 = rule2
        self.belief = np.array([[prior*prior, prior*(1-prior)], [(1-prior)*prior, (1-prior)**2]])
    
    def p_r1(self):
        return np.sum(self.belief, axis=1)[0]
    
    def p_r2(self):
        return np.sum(self.belief, axis=0)[0]
    
    def update(self, message_probs):
        # message_probs: [P(m=r1|x), P(m=r2|x)]
        m_r1, m_r2 = message_probs
        self.belief = np.array(
        [[1*self.p_r2()*m_r1 + 1*self.p_r1()*m_r2,     1*(1-self.p_r2())*m_r1 + 0*self.p_r1()*m_r2],
         [0*self.p_r2()*m_r1 + 1*(1-self.p_r1())*m_r2, 0*(1-self.p_r2())*m_r1 + 0*(1-self.p_r1())*m_r2]])

    def get_as_priors(self):
        r1 = self.p_r1()
        r2 = self.p_r2()
        return np.array([r1, r2])/(r1+r2)
        
    def get_best_rules(self):
        highest_belief = np.argmax(self.belief)
        rule_positions = np.array([[True, True], [True, False], [False, True], [False, False]]) # [[[r1, r2],[r1,-r2]], [[-r1, r2], [-r1, -r2]]]
        return np.array([self.rule1, self.rule2])[rule_positions[highest_belief]]

class ColourModel(object):

    def __init__(self, name, mu0=np.array([0.5, 0.5, 0.5]), sigma0=np.array([1, 1, 1]),
                 mu1=np.array([0.5, 0.5, 0.5]), sigma1 = np.array([1,1,1])
                 ):

        self.name = name
        self.mu0 = mu0
        
        self.mu1 = mu1
        self.sigma0 = sigma0
        self.sigma1 = sigma1

        self.n0 = 1
        self.v0 = 1
        self.n1 = 1
        self.v1 = 1

    def p(self, c, fx, p_c=0.5):
        if fx is None:
            return [1, 0][c]
        p_c_0 = 1-p_c
        p1 = p_c * np.prod(norm.pdf(fx, loc=self.mu0, scale=self.sigma0))
        p0 = p_c_0 * np.prod(norm.pdf(fx, loc=self.mu1, scale=self.sigma1))
        return [p0, p1][c]/(p1 + p0)


    # def p(self, c, fx, p_c=0.5):
    #     if fx is None:
    #         return [1, 0][c]
    #     p_c_0 = 1 - p_c
    #     log_p_c = np.log(p_c)
    #     log_p_c0 = np.log(p_c_0)
    #     log_p1 = log_p_c + np.sum(np.log(norm.pdf(fx, loc=self.mu0, scale=self.sigma0)))
    #     log_p0 = log_p_c0 + np.sum(np.log(norm.pdf(fx, loc=self.mu1, scale=self.sigma1)))
    #     norm =

    def update(self, fx, w):
        if fx is None:
            return

        new_mu0 = (self.n0 * self.mu0 + w*fx)/(self.n0+w)
        v_times_sigma = self.v0 * self.sigma0 + (self.n0 * w)/(self.n0 + w) * (self.mu0 - fx)**2
        self.v0 += w
        self.n0 += w

        self.mu0 = new_mu0
        self.sigma0 = v_times_sigma/self.v0

    def update_negative(self, fx, w):
        if fx is None:
            return

        new_mu1 = (self.n1 * self.mu1 + w*fx)/(self.n1+w)
        v_times_sigma = self.v1 * self.sigma1 + (self.n1 * w)/(self.n1 + w) * (self.mu1 - fx)**2
        self.v1 += w
        self.n1 += w

        self.mu1 = new_mu1
        self.sigma1 = v_times_sigma/self.v1

    # def update(self, fx, w):
    #     if fx is None:
    #         return self.mu0, self.alpha0, self.beta0
    #     asquigle = 1/self.gamma + 1*w
    #     bsquigle = self.mu0/self.gamma + fx*w
    #     csquigle = self.mu0**2/self.gamma + (fx*w)**2
    #     mu_post = bsquigle/asquigle
    #     alpha = self.alpha0 + 1/2
    #     beta = self.beta0 +  0.5*(csquigle - bsquigle**2/asquigle)
    #     self.mu0 = mu_post
    #     self.alpha0 = alpha
    #     self.beta0 = beta
    #     self.sigma0 = beta/(alpha + 3/2)
    #     return mu_post, self.sigma0, alpha, beta
    #     #updated_mu = (w*fx*self.sigma_prior + self.mu * self.sigma)/(w*self.sigma_prior + self.sigma)
    #     #return updated_mu


    def draw(self, show=False, save_location_basename=None):
        x = np.linspace(0, 1, 100)
        mu_r,mu_g,mu_b = self.mu0
        mu2_r, mu2_g, mu2_b = self.mu1
        sigma_r, sigma_g, sigma_b = self.sigma0
        sigma2_r, sigma2_g, sigma2_b = self.sigma1
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(x,norm.pdf(x, loc=mu_r, scale=sigma_r), color='red', label='r')
        ax1.plot(x,norm.pdf(x, loc=mu_g, scale=sigma_g), color='green', label='g')
        ax1.plot(x,norm.pdf(x, loc=mu_b, scale=sigma_b), color='blue', label='b')
        ax2.plot(x, norm.pdf(x, loc=mu2_r, scale=sigma2_r), color='red', label='r')
        ax2.plot(x, norm.pdf(x, loc=mu2_g, scale=sigma2_g), color='green', label='g')
        ax2.plot(x, norm.pdf(x, loc=mu2_b, scale=sigma2_b), color='blue', label='b')
        plt.title(self.name)
        plt.legend()
        if show:
            plt.show()
        else:
            if save_location_basename==None:
                save_location = 'results/colours/{}.png'.format(self.name)
            else:
                save_location = 'results/colours/plots/' + save_location_basename + '_' + self.name + '.png'
            plt.savefig(save_location)


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
    def __init__(self, rule_names, rules, c1, c2, rule_belief=(0.5, 0.5)):
        self.rules = rules
        self.rule_names = rule_names
        self.c1 = c1
        self.c2 = c2
        self.rule_belief = RuleBelief((c1, c2), rules[0], rules[1])
        self.rule_prior = self.rule_belief.get_as_priors()
        self.variables = [c1.name, c2.name, 'r']

    # data and hidden seem to be some kind of dictionary while hidden seems to be a list of keys
    def p(self, data, visible, priors=(0.5,0.5,0.5)):
        hidden = set(self.variables) - visible.keys()
        if not hidden:
            #print('reached here with hidden={}'.format(hidden))
            prior_c1 = priors[0] #if visible[self.c1.name] == 1 else 1-priors[0]
            prior_c2 = priors[1] #if visible[self.c2.name] == 1 else 1-priors[1]

            return (self.rule_prior[visible['r']] *
                    self.c1.p(visible[self.c1.name], data[self.c1.name], p_c=prior_c1) *
                    self.c2.p(visible[self.c2.name], data[self.c2.name], p_c=prior_c2) *
                    self.evaluate_correction(visible))
        else:
            h = hidden.pop()
            #print('adding {} to visible'.format(h))
            visible[h] = 0
            v0 = self.p(data, copy.copy(visible), priors=priors)
            #print('finished recursion for {}=0 with value {}'.format(h, v0))
            visible[h] = 1
            v1 = self.p(data, copy.copy(visible), priors=priors)
            #print('finished recursion for {}=1 with value {}'.format(h, v1))
            return v0 + v1

    def p_no_corr(self, data, visible, priors=(0.5,0.5,0.5)):
        hidden = set(self.variables) - visible.keys()
        if not hidden:
            if isinstance(self, TableCorrectionModel):
                corr = CorrectionModel.evaluate_correction(self, visible)
            else:
                corr = self.evaluate_correction(visible)
            return (self.rule_prior[visible['r']] *
                    self.c1.p(visible[self.c1.name], data[self.c1.name], p_c=priors[0]) *
                    self.c2.p(visible[self.c2.name], data[self.c2.name], p_c=priors[1]) *
                    (1 - corr))
        else:
            h = hidden.pop()
            #print('adding {} to visible'.format(h))
            visible[h] = 0
            v0 = self.p_no_corr(data, copy.copy(visible), priors=priors)
            #print('finished recursion for {}=0 with value {}'.format(h, v0))
            visible[h] = 1
            v1 = self.p_no_corr(data, copy.copy(visible), priors=priors)
            #print('finished recursion for {}=1 with value {}'.format(h, v1))
            return v0 + v1



    def p_r(self, r, data, visible={}, priors=(0.5,0.5,0.5)):
        v0 = copy.copy(visible)
        v1 = copy.copy(visible)
        v0.update({'r':0})
        v1.update({'r':1})
        r0 = self.p(data, visible=v0, priors=priors)
        r1 = self.p(data, visible=v1, priors=priors)
        eta = r0 + r1
        if np.isnan([r0, r1][r]/eta):
            print('r0, r1', r0, r1)
            print('visible', visible)
            print('priors', priors)
            print('v0 v1', v0, v1)
            print('eta', eta)
            import pdb; pdb.set_trace()
            #raise ValueError('NAN')
        
        return [r0, r1][r]/eta


    def get_message_probs(self, data, visible={}, priors=(0.5,0.5,0.5)):
        r0 = self.p_r(0, data, visible=copy.copy(visible), priors=priors)
        r1 = self.p_r(1, data, visible=copy.copy(visible), priors=priors)
        return (r0, r1)

    def update_belief_r(self, r0, r1):
        #r0 = self.p_r(0, data, visible=visible.copy())
        #r1 = self.p_r(1, data, visible=visible.copy())
        self.rule_belief.update((r0, r1))
        self.rule_prior = self.rule_belief.get_as_priors()
        return (r0, r1)

    def update_c(self, data, priors=(0.5,0.5,0.5), visible={}):
        p_c1 = self.p_c(self.c1.name, data, priors=priors, visible=visible)
        p_c2 = self.p_c(self.c2.name, data, priors=priors, visible=visible)


        self.c1.update(data[self.c1.name], p_c1)
        self.c2.update(data[self.c2.name], p_c2)
        self.c1.update_negative(data[self.c1.name], (1-p_c1))
        self.c2.update_negative(data[self.c2.name], (1-p_c2))

    def update_c_no_corr(self, data, priors=(0.5, 0.5, 0.5)):
        c1_pos = self.p_no_corr(data, visible={self.c1.name:1}, priors=priors)
        c1_neg = self.p_no_corr(data, visible={self.c1.name:0}, priors=priors)
        w1 = c1_pos/(c1_pos+c1_neg)
        c2_pos = self.p_no_corr(data, visible={self.c2.name:1}, priors=priors)
        c2_neg = self.p_no_corr(data, visible={self.c2.name:0}, priors=priors)
        w2 = c2_pos/(c2_pos+c2_neg)
        print(w1, w2)
        #print(self.c1.p(1, data[self.c1.name]))
        r1, r2 = self.rule_prior
        if (r1 > r2) and w1 > 0.5 or (r2 > r1) and w2 > 0.5:
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

    def p_c_no_corr(self, c, data, priors=(0.5, 0.5, 0.5)):
        c1_pos = self.p_no_corr(data, visible={c:1}, priors=priors)
        c1_neg = self.p_no_corr(data, visible={c:0}, priors=priors)
        w1 = c1_pos/(c1_pos+c1_neg)
        return w1

    def p_c(self, c, data, priors=(0.5, 0.5, 0.5), visible={}):

        if c in visible:
            if visible[c] == 1:
                return 1.0
            if visible[c] == 0:
                return 0.0

        vis1 = copy.copy(visible)
        vis0 = copy.copy(visible)
    
        vis1[c] = 1
        vis0[c] = 0


        #visible[c] = 1
        c_pos = self.p(data, visible=vis1, priors=priors)
        #print(c_pos)
        #c_pos = self.p(data, visible={c:1}, priors=priors)
        #visible[c] = 0
        c_neg = self.p(data, visible=vis0, priors=priors)
        #print(c_neg)
        #c_neg = self.p(data, visible={c:0}, priors=priors)
        p_c = c_pos/(c_pos+c_neg)
        return p_c

    def updated_object_priors(self, data, objs, priors, visible={}):
        return {objs[0]: {self.c1.name:self.p_c(self.c1.name, data, priors=priors, visible=copy.copy(visible))},
                objs[1]: {self.c2.name:self.p_c(self.c2.name, data, priors=priors, visible=copy.copy(visible))}}

class TableCorrectionModel(CorrectionModel):
    def __init__(self, rule_names, rules, c1, c2, rule_belief=(0.5, 0.5)):
        
        super().__init__(rule_names, rules, c1, c2, rule_belief = rule_belief)
        #self.rules = rules
        #self.c1 = c1
        #self.c2 = c2
        # self.c3 = ColourModel('{}/{}'.format(c1.name, c2.name),
        #                       mu0 = c1.mu0, beta0=c1.beta0, alpha0=c1.alpha0,
        #                       mu1=c2.mu0, beta1=c2.beta0, alpha1=c2.alpha0)
        self.c3 = ColourModel('{}/{}'.format(c1.name, c2.name),
                                mu0=c1.mu0, sigma0=c1.sigma0,
                                mu1=c2.mu0, sigma1=c2.sigma0)
        #self.rule_prior = rule_belief
        #self.variables = ['r', c1.name, c2.name, self.c3.name]
        self.variables.append(self.c3.name)

    # data and hidden seem to be some kind of dictionary while hidden seems to be a list of keys
    def p(self, data, visible, priors=(0.5, 0.5, 0.5)):
        hidden = set(self.variables) - visible.keys()
        if not hidden:
            #print('reached here with hidden={}'.format(hidden))
            prior_c1 = priors[0] #if visible[self.c1.name] == 1 else 1-priors[0]
            prior_c2 = priors[1] #if visible[self.c2.name] == 1 else 1-priors[1]
            prior_c3 = priors[2] #if visible[self.c3.name] == 1 else 1-priors[2]
            return (self.rule_prior[visible['r']] *
                    self.c1.p(visible[self.c1.name], data[self.c1.name], p_c=prior_c1) *
                    self.c2.p(visible[self.c2.name], data[self.c2.name], p_c=prior_c2) *
                    self.evaluate_correction(visible) *
                    self.c3.p(visible[self.c3.name], data[self.c3.name], p_c=prior_c3)
                   )
        else:
            h = hidden.pop()
            #print('adding {} to visible'.format(h))
            visible[h] = 0
            v0 = self.p(data, copy.copy(visible), priors=priors)
            #print('finished recursion for {}=0 with value {}'.format(h, v0))
            visible[h] = 1
            v1 = self.p(data, copy.copy(visible), priors=priors)
            #print('finished recursion for {}=1 with value {}'.format(h, v1))
            return v0 + v1

    #def p_r(self, r, data, visible={}):
    #    return super().p_r(r, data.copy(), visible.copy())


    def evaluate_correction(self, visible):
        # r=0: \forall x. y. c1(x) & on(x,y) -> c2(y). => on(x,y) -c1(x) c2(y) c1(z)
        # r=1 \forall x.y. c2(y) & on(x,y) -> c1(x). => on(x, y) c1(x) -c2(y) c2(z)
        rule0 = visible['r'] == 0 and visible[self.c1.name] == 0 and visible[self.c2.name] == 1 and visible[self.c3.name] == 1
        rule1 = visible['r'] == 1 and visible[self.c1.name] == 1 and visible[self.c2.name] == 0 and visible[self.c3.name] == 0
        return float(rule0 or rule1)

    def update_c(self, data, priors=(0.5, 0.5, 0.5), visible={}):
        prior_dict = self.updated_object_priors(data, ['o1', 'o2', 'o3'], priors, visible=visible)

        w1 = prior_dict['o1'][self.c1.name]
        w2 = prior_dict['o2'][self.c2.name]
        w3 = prior_dict['o3'][self.c1.name]
        w4 = prior_dict['o3'][self.c2.name]
        # w1 = self.p_c(self.c1.name, data, priors=priors)
        # w2 = self.p_c(self.c2.name, data, priors=priors)
        # w4 = self.p_c(self.c3.name, data, priors=priors)
        # w3 = 1-w4 # these are in the other direction because 0 for c3 means it is equivalent to c1 and p_c returns probability that c=1
        self.c1.update(data[self.c1.name], w1)
        self.c2.update(data[self.c2.name], w2)
        self.c1.update(data[self.c3.name], w3)
        self.c2.update(data[self.c3.name], w4)

    def updated_object_priors(self, data, objs, priors, visible={}):
        obj_dict = super().updated_object_priors(data, objs, priors, visible=copy.copy(visible))
        p3 = self.p_c(self.c3.name, data, priors=priors, visible=copy.copy(visible))
        obj_dict[objs[2]] = {self.c1.name: p3, self.c2.name:1-p3}
        return obj_dict
        
