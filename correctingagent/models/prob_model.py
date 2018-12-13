import numpy as np
from scipy.stats import norm
import copy
import matplotlib.pyplot as plt
import torch
import logging
from KDEpy import NaiveKDE
#ColourModel = namedtuple('ColourModel', ['name', 'mu', 'sigma'])
#rule_belief = (0.5, 0.5)

default_update_negative=False

logger = logging.getLogger('agent')


class RuleBelief(object):

    def __init__(self, colours, rule1, rule2, prior=0.001):
        # [[(r1,r2), (r1,-r2)],[(-r1,r2), (-r1, -r2)]]
        self.no_learning = False
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
        if self.no_learning:
            return np.array([0.5, 0.5])
        else:
            r1 = self.p_r1()
            r2 = self.p_r2()
            return np.array([r1, r2])/(r1+r2)
        #return np.array([0.5, 0.5])


    def get_best_rules(self):
        highest_belief = np.argmax(self.belief)
        rule_positions = np.array([[True, True], [True, False], [False, True], [False, False]]) # [[[r1, r2],[r1,-r2]], [[-r1, r2], [-r1, -r2]]]
        return np.array([self.rule1, self.rule2])[rule_positions[highest_belief]]



class NeuralColourModel(object):

    def __init__(self, name, lr=0.1, H=10, momentum=0, dampening=0, weight_decay=0, nesterov=False, optimiser='Adam'):
        self.name = name
        D_in = 3
        H = H
        D_out = 1
        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(), # torch.nn.Tanh(),
            torch.nn.Linear(H, D_out),
            torch.nn.Sigmoid(),
        )

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.lr = lr
        if optimiser == 'SGD':
            self.optim =  torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        elif optimiser == 'Adam':
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR()

    def p(self, c, fx, p_c=None):

        if fx is None:
            return torch.tensor([1, 0][c], dtype=torch.float)

        fx = torch.tensor(fx, dtype=torch.float)
        p_c1 = self.model(fx)
        p_c0 = 1-p_c1
        return [p_c0, p_c1][c]


    def update(self, fx, w):
        if fx is None:
            return
        w = torch.tensor(w, dtype=torch.float)

        y_pred = self.p(1, fx)
        loss = self.loss_fn(y_pred, w)
        #print(loss)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def update_negative(self, fx, w):
        pass


class ComboModel(object):
    def __init__(self, name, cm1, cm2):
        self.name = name
        self.cm1 = cm1
        self.cm2 = cm2

    def p(self, c, fx, p_c=None):
        p1 = self.cm1.p(c, fx)
        p2 = self.cm2.p(c, fx)
        n = p1 + p2
        return [p1/n, p2/n][c]

class ColourModel(object):

    def __init__(self, name, mu0=np.array([0.5, 0.5, 0.5]), sigma0=np.array([1, 1, 1]),
                 mu1=np.array([0.5, 0.5, 0.5]), sigma1 = np.array([1,1,1])
                 ):

        self.name = name
        self.mu0 = mu0
        self.mu1 = mu1


        self.sigma0_times_v0 = np.array([4.,4.,4.])
        self.sigma1_times_v1 = np.array([4.,4.,4.])

        self.n0 = 1
        self.v0 = 2
        self.n1 = 1
        self.v1 = 2

        self.sigma1 = (self.sigma1_times_v1/2)/(self.v1/2 + 1)
        self.sigma0 = (self.sigma0_times_v0/2)/(self.v0/2 + 1)

        self.alpha0 = np.array([1, 1, 1])
        self.beta0 = np.array([1, 1, 1])
        self.gamma = 0.005
        #self.sigma0 = self.beta0/(self.alpha0 + 3/2)

    def p(self, c, fx, p_c=0.5):
        if fx is None:
            return [1, 0][c]
        p_c_0 = 1-p_c
        p1 = p_c * np.prod(norm.pdf(fx, loc=self.mu0, scale=self.sigma0))
        p0 = p_c_0 * np.prod(norm.pdf(fx, loc=self.mu1, scale=self.sigma1))
        return [p0, p1][c]/(p1 + p0)


    def update(self, fx, w):
        if fx is None:
            return

        new_mu0 = (self.n0 * self.mu0 + w*fx)/(self.n0+w)
        self.sigma0_times_v0 = self.sigma0_times_v0 + (self.n0 * w)/(self.n0 + w) * (self.mu0 - fx)**2
        self.v0 += w
        self.n0 += w

        self.mu0 = new_mu0
        #self.sigma0 = v_times_sigma/self.v0 # I believe this is wrong
        self.sigma0 = (self.sigma0_times_v0/2)/(self.v0/2 + 1)
        #print(self.sigma0)



    def update_negative(self, fx, w):
        if fx is None:
            return

        new_mu1 = (self.n1 * self.mu1 + w*fx)/(self.n1+w)
        self.sigma1_times_v1 = self.sigma1_times_v1 + (self.n1 * w)/(self.n1 + w) * (self.mu1 - fx)**2
        self.v1 += w
        self.n1 += w

        self.mu1 = new_mu1
        self.sigma1 = (self.sigma1_times_v1/2)/(self.v1/2 + 1)

    def update2(self, fx, w):
        if fx is None:
            return self.mu0, self.alpha0, self.beta0
        asquigle = 1/self.gamma + 1*w
        bsquigle = self.mu0/self.gamma + fx*w
        csquigle = self.mu0**2/self.gamma + (fx*w)**2
        mu_post = bsquigle/asquigle
        alpha = self.alpha0 + 1/2
        beta = self.beta0 +  0.5*(csquigle - bsquigle**2/asquigle)
        self.mu0 = mu_post
        self.alpha0 = alpha
        self.beta0 = beta
        self.sigma0 = self.beta0/(self.alpha0 + 3/2)
        return mu_post, self.sigma0, alpha, beta
        #updated_mu = (w*fx*self.sigma_prior + self.mu * self.sigma)/(w*self.sigma_prior + self.sigma)
        #return updated_mu


    def draw(self, show=False, save_location_basename=None, draw_both=False):
        x = np.linspace(0, 1, 100)
        mu_r,mu_g,mu_b = self.mu0
        mu2_r, mu2_g, mu2_b = self.mu1
        sigma_r, sigma_g, sigma_b = self.sigma0
        sigma2_r, sigma2_g, sigma2_b = self.sigma1

        if draw_both:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.set_title('P(F(x)|{}(x)=1)'.format(self.name), fontsize=14)
            ax1.plot(x, norm.pdf(x, loc=mu_r, scale=sigma_r), color='red', label='r')
            ax1.plot(x, norm.pdf(x, loc=mu_g, scale=sigma_g), color='green', label='g')
            ax1.plot(x, norm.pdf(x, loc=mu_b, scale=sigma_b), color='blue', label='b')

            ax2.set_title('P(F(x)|{}(x)=0)'.format(self.name), fontsize=14)
            ax2.plot(x, norm.pdf(x, loc=mu2_r, scale=sigma2_r), color='red', label='r')
            ax2.plot(x, norm.pdf(x, loc=mu2_g, scale=sigma2_g), color='green', label='g')
            ax2.plot(x, norm.pdf(x, loc=mu2_b, scale=sigma2_b), color='blue', label='b')
        else:
            fig, ax1 = plt.subplots(1, 1)
            ax1.set_title('P(F(x)|{}(x)=1)'.format(self.name), fontsize=14)
            ax1.plot(x, norm.pdf(x, loc=mu_r, scale=sigma_r), color='red', label='r')
            ax1.plot(x, norm.pdf(x, loc=mu_g, scale=sigma_g), color='green', label='g')
            ax1.plot(x, norm.pdf(x, loc=mu_b, scale=sigma_b), color='blue', label='b')

        plt.legend(prop={'size': 10})

        if show:
            plt.show()
        else:
            if save_location_basename==None:
                save_location = 'results/colours/{}.png'.format(self.name)
            else:
                save_location = 'results/colours/plots/' + save_location_basename + '_' + self.name + '.png'
            plt.savefig(save_location)


class MLEColourModel(ColourModel):
    def __init__(self, *args, **kwargs):
        self.data = []
        self.data_negative = []
        super().__init__(*args, **kwargs)

    # def update(self, fx, w):
    #     self.data.append((w, fx))
    #     total = np.array([0,0,0])
    #     norm = 0
    #     for w, fx in self.data:
    #         total += w*fx
    #         norm += w
    #     self.mu0 = total/norm
    #     for w, fx in self.data:

    def update(self, fx, w):
        if fx is None:
            return

        total = np.array([0,0,0])
        norm = 0
        for w, fx in self.data:
            total += w*fx
            norm += w
        sample_mean = total/norm
        new_mu = (self.n0*self.mu0 + norm*sample_mean)/(self.n0 + norm)
        var = np.array([0,0,0])
        for w, fx in self.data:
            var += w*(fx-sample_mean)**2
        new_sigma_times_v0 = self.sigma0_times_v0 + var + (self.n0 * norm)/(self.n0 + norm) * (self.mu0 - sample_mean)**2

        v0_prime = self.v0 + norm


        self.mu0 = new_mu0

        self.sigma0 = (new_sigma_times_v0/2)/(v0_prime/2 + 1)


def get_bw_value(data):
    n = len(data)
    if n == 0:
        return 0.5
    bw = max(0.5/n, 0.1)
    return bw

class KDEColourModel(ColourModel):

    def __init__(self, name, bw=0.15, data = None, weights=np.array([]), data_neg=None, weights_neg=np.array([]), kernel='gaussian', fix_bw=False, use_3d=False, norm=2):
        self.name = name
        self.data = data
        self.weights = weights
        self.model = None
        self.model_neg = None
        self.data_neg = data_neg
        self.weights_neg = weights_neg
        self.use_3d = use_3d
        if fix_bw:
            self.bw = lambda x: bw
        else:
            self.bw = get_bw_value
        self.kernel = kernel
        if data is not None:
            self.model = self.fit_model(self.data, self.weights)
        if data_neg is not None:
            self.model_neg = self.fit_model(self.data_neg, self.weights_neg)


    def update(self, fx, w):
        if fx is None or w == 0:
            return
        if self.data is None:
            self.data = np.array([fx])
        else:
            self.data = np.append(self.data, np.array([fx]), axis=0)
        self.weights = np.append(self.weights, w)
        self.model = self.fit_model(self.data, self.weights)

    def update_negative(self, fx, w):
        if fx is None or w == 0:
            return
        if self.data_neg is None:
            self.data_neg = np.array([fx])
        else:
            self.data_neg = np.append(self.data_neg, np.array([fx]), axis=0)
        self.weights_neg = np.append(self.weights_neg, w)
        self.model_neg = self.fit_model(self.data_neg, self.weights_neg)

    def fit_model(self, data, weights):
        #print('data', data)
        if not self.use_3d:
            train_data = np.concatenate([data, -data, 2-data])
            #print('train data', train_data)
            r = train_data[:,0]
            g = train_data[:,1]
            b = train_data[:,2]
            #print('r', r)
            weights = np.concatenate([weights, weights, weights])
            bw = self.bw(data)
            r_model = NaiveKDE(kernel=self.kernel, bw=bw).fit(r, weights=weights)
            g_model = NaiveKDE(kernel=self.kernel, bw=bw).fit(g, weights=weights)
            b_model = NaiveKDE(kernel=self.kernel, bw=bw).fit(b, weights=weights)
            model = (r_model, g_model, b_model)
            return model
        else:
            assert(data.shape[1] == 3)
            bw = self.bw(data)
            model = NaiveKDE(kernel=self.kernel, bw=bw, norm=2).fit(data)
            return model

    def evaluate_model(self, model, fx, split=False):
        if not self.use_3d:
            r, g, b = fx
            r_model, g_model, b_model = model
            try:
                p_r = r_model.evaluate(np.array(r))
                p_g = g_model.evaluate(np.array(g))
                p_b = b_model.evaluate(np.array(b))
            except ValueError:
                p_r = r_model.evaluate(np.array([r]))[0]
                p_g = g_model.evaluate(np.array([g]))[0]
                p_b = b_model.evaluate(np.array([b]))[0]
            if not split:
                return 3*p_r * 3*p_g * 3*p_b
            else:
                return 3*p_r, 3*p_g, 3*p_b
        else:
            fx = np.array(fx)
            if len(fx.shape) == 2:
                p = model.evaluate(np.array(fx))
            elif len(fx.shape) == 1:
                p = model.evaluate(np.array([fx]))[0]
            else:
                raise ValueError('features are the wrong shape, expected either (3,) or (n, 3) but got {}'.format(fx.shape))
            return p


    def p(self, c, fx, p_c=0.5):
        if fx is None:
            return [1, 0][c]
        p_c_0 = 1-p_c
        p_f_c1 = self.evaluate_model(self.model, fx) if self.model else 1
        p_f_c0 = self.evaluate_model(self.model_neg, fx) if self.model_neg else 1
        p1 = p_c * p_f_c1
        p0 = p_c_0 * p_f_c0
        return [p0, p1][c]/(p1 + p0)

    def draw(self, show=False, save_location_basename=None, draw_both=False):
        x = np.linspace(0, 1, 100)
        r_model, g_model, b_model = self.model
        rs = 3*r_model.evaluate(x)
        gs = 3*g_model.evaluate(x)
        bs = 3*b_model.evaluate(x)


        if draw_both:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.set_title('P(F(x)|{}(x)=1)'.format(self.name), fontsize=14)
            ax1.plot(x, rs, color='red', label='r')
            ax1.plot(x, gs, color='green', label='g')
            ax1.plot(x, bs, color='blue', label='b')

            r_model_neg, g_model_neg, b_model_neg = self.model_neg

            rs_neg = 3*r_model_neg.evaluate(x)
            gs_neg = 3*g_model_neg.evaluate(x)
            bs_neg = 3*b_model_neg.evaluate(x)

            ax2.set_title('P(F(x)|{}(x)=0)'.format(self.name), fontsize=14)
            ax2.plot(x, rs_neg, color='red', label='r')
            ax2.plot(x, gs_neg, color='green', label='g')
            ax2.plot(x, bs_neg, color='blue', label='b')
        else:
            fig, ax1 = plt.subplots(1, 1)
            ax1.set_title('P(F(x)|{}(x)=1)'.format(self.name), fontsize=14)
            ax1.plot(x, rs, color='red', label='r')
            ax1.plot(x, gs, color='green', label='g')
            ax1.plot(x, bs, color='blue', label='b')

        plt.legend(prop={'size': 10})

        if show:
            plt.show()
        else:
            if save_location_basename==None:
                save_location = 'results/colours/{}.png'.format(self.name)
            else:
                save_location = 'results/colours/plots/' + save_location_basename + '_' + self.name + '.png'
            plt.savefig(save_location)



class CorrectionModel(object):

    def __init__(self, rule_names, rules, c1, c2, rule_belief=None):
        self.rules = rules
        self.rule_names = rule_names
        self.c1 = c1
        self.c2 = c2
        if rule_belief is None:
            self.rule_belief = RuleBelief((c1.name, c2.name), rules[0], rules[1])
        else:
            self.rule_belief = rule_belief
        self.rule_prior = self.rule_belief.get_as_priors()
        self.variables = [c1.name, c2.name, 'r']

    # data and hidden seem to be some kind of dictionary while hidden seems to be a list of keys
    def p(self, data, visible, priors=(0.5,0.5,0.5)):
        hidden = set(self.variables) - visible.keys()
        if not hidden:
            prior_c1 = priors[0] #if visible[self.c1.name] == 1 else 1-priors[0]
            prior_c2 = priors[1] #if visible[self.c2.name] == 1 else 1-priors[1]

            try:
                rp = self.rule_prior[visible['r']]
                c1p = self.c1.p(visible[self.c1.name], data[self.c1.name], p_c=prior_c1)
                c2p = self.c2.p(visible[self.c2.name], data[self.c2.name], p_c=prior_c2)
                correction_eval = self.evaluate_correction(visible)
                # logger.debug(visible)
                # logger.debug('rule prior: ' + str(rp))
                # logger.debug('P({}=1): '.format(self.c1.name) + str(c1p))
                # logger.debug('P({}=1): '.format(self.c2.name) + str(c2p))
                return (rp * c1p * c2p * correction_eval)
            except RuntimeError:
                return (self.rule_prior[visible['r']] *
                    self.c1.p(visible[self.c1.name], data[self.c1.name], p_c=prior_c1).detach().numpy() *
                    self.c2.p(visible[self.c2.name], data[self.c2.name], p_c=prior_c2).detach().numpy() *
                    self.evaluate_correction(visible))
        else:
            h = hidden.pop()
            visible[h] = 0
            v0 = self.p(data, copy.copy(visible), priors=priors)
            visible[h] = 1
            v1 = self.p(data, copy.copy(visible), priors=priors)
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
            visible[h] = 0
            v0 = self.p_no_corr(data, copy.copy(visible), priors=priors)
            visible[h] = 1
            v1 = self.p_no_corr(data, copy.copy(visible), priors=priors)
            return v0 + v1

    def p_r(self, r, data, visible={}, priors=(0.5,0.5,0.5)):
        v0 = copy.copy(visible)
        v1 = copy.copy(visible)
        v0.update({'r':0})
        v1.update({'r':1})
        r0 = self.p(data, visible=v0, priors=priors)
        r1 = self.p(data, visible=v1, priors=priors)
        # logger.debug((r0, r1))
        eta = r0 + r1
        if np.isnan([r0, r1][r]/eta):
            print('r0, r1', r0, r1)
            print('visible', visible)
            print('priors', priors)
            print('v0 v1', v0, v1)
            print('eta', eta)
            import pdb; pdb.set_trace()

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

    def update_c(self, data, priors=(0.5,0.5,0.5), visible={}, update_negative=default_update_negative, which_to_update=(1,1,1)):

        p_c1 = self.p_c(self.c1.name, data, priors=priors, visible=visible)
        p_c2 = self.p_c(self.c2.name, data, priors=priors, visible=visible)


        # logger.debug('predicted P({}=1) = {}'.format(self.c1.name, p_c1))
        # logger.debug('predicted P({}=1) = {}'.format(self.c2.name, p_c2))
        if which_to_update[0]:
            self.c1.update(data[self.c1.name], p_c1)
        if which_to_update[1]:
            self.c2.update(data[self.c2.name], p_c2)
        if update_negative:
            if which_to_update[0]:
                self.c1.update_negative(data[self.c1.name], (1-p_c1))
            if which_to_update[1]:
                self.c2.update_negative(data[self.c2.name], (1-p_c2))

    def update_c_no_corr(self, data, priors=(0.5, 0.5, 0.5)):
        c1_pos = self.p_no_corr(data, visible={self.c1.name:1}, priors=priors)
        c1_neg = self.p_no_corr(data, visible={self.c1.name:0}, priors=priors)
        w1 = c1_pos/(c1_pos+c1_neg)
        c2_pos = self.p_no_corr(data, visible={self.c2.name:1}, priors=priors)
        c2_neg = self.p_no_corr(data, visible={self.c2.name:0}, priors=priors)
        w2 = c2_pos/(c2_pos+c2_neg)
        #print(w1, w2)
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
    def __init__(self, rule_names, rules, c1, c2, rule_belief=None):

        super().__init__(rule_names, rules, c1, c2, rule_belief=rule_belief)
        #self.rules = rules
        #self.c1 = c1
        #self.c2 = c2
        # self.c3 = ColourModel('{}/{}'.format(c1.name, c2.name),
        #                       mu0 = c1.mu0, beta0=c1.beta0, alpha0=c1.alpha0,
        #                       mu1=c2.mu0, beta1=c2.beta0, alpha1=c2.alpha0)

        if isinstance(c1, NeuralColourModel):
            self.c3 = ComboModel('{}/{}'.format(c1.name, c2.name), cm1=c1, cm2=c2)
        elif isinstance(c1, KDEColourModel):
            self.c3 = KDEColourModel('{}/{}'.format(c1.name, c2.name),
                                     data=c1.data, weights=c1.weights,
                                     data_neg=c2.data, weights_neg=c2.weights)
        else:
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
            rp = self.rule_prior[visible['r']]
            c1p = self.c1.p(visible[self.c1.name], data[self.c1.name], p_c=prior_c1)
            c2p = self.c2.p(visible[self.c2.name], data[self.c2.name], p_c=prior_c2)
            c3p = self.c3.p(visible[self.c3.name], data[self.c3.name], p_c=prior_c3)
            correction_eval = self.evaluate_correction(visible)
            # if correction_eval == 1:
            #     logger.debug(visible)
            #     logger.debug('rule prior: ' + str(rp))
            #     logger.debug('P({}=1): '.format(self.c1.name) + str(c1p))
            #     logger.debug('P({}=1): '.format(self.c2.name) + str(c2p))
            #     logger.debug('P({}=1): '.format(self.c3.name) + str(c3p))


            try:
                return (rp * c1p * c2p * c3p *correction_eval)
            except RuntimeError:
                return (self.rule_prior[visible['r']] *
                        self.c1.p(visible[self.c1.name], data[self.c1.name], p_c=prior_c1).detach().numpy() *
                        self.c2.p(visible[self.c2.name], data[self.c2.name], p_c=prior_c2).detach().numpy() *
                        self.evaluate_correction(visible) *
                        self.c3.p(visible[self.c3.name], data[self.c3.name], p_c=prior_c3).detach().numpy()
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

    def update_c(self, data, priors=(0.5, 0.5, 0.5), visible={}, update_negative=default_update_negative, which_to_update=(1,1,1)):
        prior_dict = self.updated_object_priors(data, ['o1', 'o2', 'o3'], priors, visible=visible)

        w1 = prior_dict['o1'][self.c1.name]
        w2 = prior_dict['o2'][self.c2.name]
        w3 = prior_dict['o3'][self.c1.name]
        w4 = prior_dict['o3'][self.c2.name]
        # w1 = self.p_c(self.c1.name, data, priors=priors)
        # w2 = self.p_c(self.c2.name, data, priors=priors)
        # w4 = self.p_c(self.c3.name, data, priors=priors)
        # w3 = 1-w4 # these are in the other direction because 0 for c3 means it is equivalent to c1 and p_c returns probability that c=1
        # logger.debug('predicted P({}=1) = {}'.format(self.c1.name, w1))
        # logger.debug('predicted P({}=1) = {}'.format(self.c2.name, w2))
        # logger.debug('predicted P({}=1) = {}'.format(self.c1.name, w3))
        # logger.debug('predicted P({}=1) = {}'.format(self.c2.name, w4))
        if which_to_update[0]:
            self.c1.update(data[self.c1.name], w1)
        if which_to_update[1]:
            self.c2.update(data[self.c2.name], w2)
        if which_to_update[2]:
            self.c1.update(data[self.c3.name], w3)
            self.c2.update(data[self.c3.name], w4)
        if update_negative:
            if which_to_update[0]:
                self.c1.update_negative(data[self.c1.name], 1-w1)
            if which_to_update[1]:
                self.c2.update_negative(data[self.c2.name], 1-w2)


    def updated_object_priors(self, data, objs, priors, visible={}):
        obj_dict = super().updated_object_priors(data, objs, priors, visible=copy.copy(visible))
        p3 = self.p_c(self.c3.name, data, priors=priors, visible=copy.copy(visible))
        obj_dict[objs[2]] = {self.c1.name: p3, self.c2.name:1-p3}
        return obj_dict
