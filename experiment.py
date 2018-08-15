import world
import agents
from ff import Solved
import ff
import pddl_functions
from teacher import TeacherAgent
import os
import numpy as np
from scipy.stats import norm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd
import gc
import pickle
from evaluation import plot_cumsum, test_colour_model, ResultsFile, get_agent
import configparser
import logging
from collections import defaultdict
import prob_model


handler = logging.StreamHandler()

agent_logger = logging.getLogger('agent')
agent_logger.setLevel(logging.INFO)
agent_logger.addHandler(handler)

logger = logging.getLogger('dialogue')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

DEBUG = True


class Debug(object):

    def __init__(self, config):
        suite = config['scenario_suite']
        agent = config['agent']
        threshold = config['threshold']
        dir_ = 'debug/{}/{}/{}'.format(suite, agent, threshold)
        os.makedirs(dir_, exist_ok=True)
        self.dir_ = dir_
        self.nr = len(os.listdir(dir_))

        self.cm_results = defaultdict(list)
        self.cm_params = defaultdict(list)


    def cm_confusion(self, agent):
        for cm in agent.colour_models.values():
            confusion = test_colour_model(cm)
            for key, value in confusion.items():
                self.cm_results[(cm.name, key)].append(value)

    def save_confusion(self):
        try:
            df = pd.DataFrame(self.cm_results)
        except ValueError:
            longest_list = max(map(len, self.cm_results.values()))
            for key in self.cm_results.keys():
                n = len(self.cm_results[key])
                if  n < longest_list:
                    diff = longest_list - n
                    for i in range(diff):
                        self.cm_results[key].insert(0, None)
            df = pd.DataFrame(self.cm_results)

        df.to_pickle(os.path.join(self.dir_, 'cm_results{}.pickle'.format(self.nr)))

    def update_cm_params(self, agent):
        for cm in agent.colour_models.values():
            if isinstance(cm, prob_model.NeuralColourModel):
                return
            mean = cm.mu0
            std_dev = cm.sigma0
            self.cm_params[(cm.name, 'mu')].append(mean)
            self.cm_params[(cm.name, 'sigma')].append(std_dev)
            self.cm_params[(cm.name, 'mu1')].append(cm.mu1)
            self.cm_params[(cm.name, 'sigma1')].append(cm.sigma1)

    def save_params(self):
        try:
            df = pd.DataFrame(self.cm_params)
        except ValueError:
            longest_list = max(map(len, self.cm_params.values()))
            for key in self.cm_params.keys():
                n = len(self.cm_params[key])
                if  n < longest_list:
                    diff = longest_list - n
                    for i in range(diff):
                        self.cm_params[key].insert(0, None)
            df = pd.DataFrame(self.cm_params)

        df.to_pickle(os.path.join(self.dir_, 'cm_params{}.pickle'.format(self.nr)))

def run_experiment(config_name='DEFAULT'):


    config = configparser.ConfigParser()
    config.read('config/experiments.ini')
    config = config[config_name]
    problem_dir = config['scenario_suite']
    threshold = config.getfloat('threshold')
    Agent = get_agent(config)
    vis = config.getboolean('visualise')
    neural = config.getboolean('neural')

    if DEBUG and not 'Random' in config['agent']:
        debugger = Debug(config)

    results_file = ResultsFile(config=config)

    total_reward = 0
    problems = os.listdir(problem_dir)
    w = world.PDDLWorld('blocks-domain.pddl', '{}/{}'.format(problem_dir, problems[0]))
    teacher = TeacherAgent()
    agent = Agent(w, teacher=teacher, threshold=threshold)
    #results_file = 'results/{}_{}_{}{}.out'.format(agent.name, problem_dir, threshold, file_modifiers)
    #with open(results_file, 'w') as f:
    #    f.write('Results for {}\n'.format(problem_dir))

    results_file.write('Results for {}\n'.format(problem_dir))
    for problem in problems:
        w = world.PDDLWorld('blocks-domain.pddl', '{}/{}'.format(problem_dir, problem))
        agent.new_world(w)
        while not w.test_success():
            plan = agent.plan()
            for a, args in plan:
                if a == 'reach-goal':
                    break
                w.update(a, args)
                if vis:
                    w.draw()
                correction = agent.teacher.correction(w)
                if correction:
                    logger.info("T: " + correction)
                    agent.get_correction(correction, a, args)
                    if vis:
                        w.draw()
                    break
        if DEBUG and not 'Random' in config['agent']:
            debugger.cm_confusion(agent)
            debugger.update_cm_params(agent)


                #else:
                #    agent.no_correction(a, args)
        # if np.isnan(np.sum(agent.colour_models[agent.colour_models.ke].mu0)):
        #     raise ValueError('NAN NAN NAN')
        clear_output()
        total_reward += w.reward
        print('{} reward: {}'.format(problem, w.reward))
        #gc.collect()
        #with open(results_file, 'a') as f:
        results_file.write('{} reward: {}\n'.format(problem, w.reward))
        results_file.write('{} cumulative reward: {}\n'.format(problem, total_reward))

    #print('total reward: {}'.format(total_reward))
    #with open(results_file, 'a') as f:
    results_file.write('total reward: {}\n'.format(total_reward))

    results_file.save_agent(agent)
    if DEBUG and not 'Random' in config['agent']:
        debugger.save_confusion()
        debugger.save_params()

    # with open('results/agents/{}_{}_{}{}.pickle'.format(agent.name, problem_dir, threshold, file_modifiers), 'wb') as f:
    #     try:
    #         agent.priors.to_dict()
    #     except AttributeError:
    #         pass
    #     pickle.dump(agent, f)

    #plot_cumsum(results_file, discount=True, save_loc='results/plots/{}_{}_{}.png'.format(agent.name, problem_dir, threshold))
