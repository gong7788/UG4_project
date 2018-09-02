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
import sqlalchemy

handler = logging.StreamHandler()

agent_logger = logging.getLogger('agent')
agent_logger.setLevel(logging.INFO)
agent_logger.addHandler(handler)

logger = logging.getLogger('dialogue')
logger.setLevel(logging.INFO)
logger.addHandler(handler)



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

def get_neural_config(name):
    config = configparser.ConfigParser()
    config.read('config/neural.ini')
    config_dict = {}
    config = config[name]
    config_dict['lr'] = config.getfloat('lr')
    config_dict['H'] = config.getint('H')
    config_dict['momentum'] = config.getfloat('momentum')
    config_dict['dampening'] = config.getfloat('dampening')
    config_dict['weight_decay'] = config.getfloat('weight_decay')
    config_dict['nesterov'] = config.getboolean('nesterov')
    config_dict['optimiser'] = config['optimiser']
    return config_dict



def run_experiment(config_name='DEFAULT', debug=False, neural_config='DEFAULT'):

    if debug:
        agent_logger.setLevel(logging.DEBUG)

    config = configparser.ConfigParser()
    config.read('config/experiments.ini')
    config = config[config_name]
    problem_dir = config['scenario_suite']
    threshold = config.getfloat('threshold')
    update_negative = config.getboolean('update_negative')
    Agent = get_agent(config)
    vis = config.getboolean('visualise')


    if debug and not 'Random' in config['agent']:
        debugger = Debug(config)

    results_file = ResultsFile(config=config)

    total_reward = 0
    problems = os.listdir(problem_dir)
    w = world.PDDLWorld('blocks-domain.pddl', '{}/{}'.format(problem_dir, problems[0]))
    teacher = TeacherAgent()
    if Agent in [agents.NeuralCorrectingAgent]:
        config_dict = get_neural_config(neural_config)
        agent = Agent(w, teacher=teacher, **config_dict)
    elif Agent in [agents.CorrectingAgent]:
        agent = Agent(w, teacher=teacher, threshold=threshold, update_negative=update_negative)
    else:
        agent = Agent(w, teacher=teacher, threshold=threshold)


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
        if debug and not 'Random' in config['agent']:
            debugger.cm_confusion(agent)
            debugger.update_cm_params(agent)



        total_reward += w.reward
        print('{} reward: {}'.format(problem, w.reward))

        results_file.write('{} reward: {}\n'.format(problem, w.reward))
        results_file.write('{} cumulative reward: {}\n'.format(problem, total_reward))


    results_file.write('total reward: {}\n'.format(total_reward))

    results_file.save_agent(agent)
    if debug and not 'Random' in config['agent']:
        debugger.save_confusion()
        debugger.save_params()

    return results_file.name



def add_experiment(config_name, neural_config, debug=False):
    engine = sqlalchemy.create_engine('sqlite:///db/experiments.db')
    df = pd.read_sql('experiments', index_col='index', con=engine)

    df = df.append({'config_name':config_name, 'neural_config':neural_config, 'status':'running'}, ignore_index=True)
    df.to_sql('experiments', con=engine, if_exists='replace')

    results_file = run_experiment(config_name=config_name, neural_config=neural_config, debug=debug)
    df = pd.read_sql('experiments', index_col='index', con=engine)
    last_label = df.index[-1]
    df.at[last_label, 'experiment_file'] = results_file
    df.at[last_label, 'status'] = 'done'
    df.to_sql('experiments', con=engine, if_exists='replace')
