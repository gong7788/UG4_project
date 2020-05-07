
import pickle
import os
import webcolors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..util.colour_dict import colour_dict
from ..util.util import get_config, config_location
from ..agents import agents, PGMAgent
from ..world.colours import name_to_rgb, name_to_hsv, get_colour
import configparser
from collections import defaultdict
import seaborn as sns
from skimage.color import rgb2hsv, hsv2rgb
from ..world.colours import colour_generators
from ..models import prob_model
sns.set_style('darkgrid')
sns.set_context("paper")
# from experiment_tracking import read_experiments, get_results_file, get_baseline


c = get_config()
results_location = c['results_location']


def get_agent_name(result_file):
    name = result_file.name
    return [c for c in name.split('/') if ('agents.' in c or 'PGM' in c)][0]


class Experiment(object):

    def __init__(self, config_name='DEFAULT'):

        config = configparser.ConfigParser()
        config_file = os.path.join(config_location, 'experiments.ini')
        config.read(config_file)

        config = config[config_name]

        suite = config['scenario_suite']
        self.name = suite
        self.threshold = config['threshold']
        agents = os.listdir(os.path.join(results_location, suite))
        results_files = []
        for agent in agents:

            config['agent'] = agent
            rf = ResultsFile.read(config)
            results_files.append((agent, rf))
        self.results_files = results_files

    @staticmethod
    def from_results_files(results_files, name):
        exp = Experiment()
        rfs = {}
        exp.name = name
        agent_name_counter = defaultdict(int)
        for rf in results_files:
            agent_name = get_agent_name(rf)
            nr = agent_name_counter[agent_name]
            agent_name_counter[agent_name] += 1
            rfs['_'.join([agent_name,str(nr)])] = rf
        exp.results_files = rfs
        return exp

    def to_df(self, discount=True):
        results = {}
        for name, f in self.results_files.items():
            rewards, cum_rewards = f.read_file()
            results[name] = rewards
        df = pd.DataFrame(data=results)
        if discount:
            for column in df.columns:
                df[column] += 10
                df[column] = -1 * df[column]/2
                df['cumsum_{}'.format(column)] = df[column].cumsum()
        return df

    def to_test_df(self, discount=True):
        results = {}
        for name, f in self.results_files.items():
            rewards, cum_rewards = f.read_file()
            results[name] = rewards
        df = pd.DataFrame(data=results)
        if discount:
            for column in df.columns:
                df[column] += 10
                df[column] = -1*df[column]/2
                df['cumsum_{}'.format(column)] = df[column].cumsum()
        return df

    def print_test(self, labels=[]):
        df = self.to_test_df()
        columns = list(filter(lambda x: 'cumsum' in x, df.columns))
        if not labels:
            labels = columns
        for l, c in zip(labels, columns):
            a = df[c].tail(1)
            print(l, a.values[0])

    def show_results(self, labels=[]):
        self.plot(labels=labels)
        self.print_test(labels=labels)

    def plot_df(self, df, labels=[], dataset_label=None):
        locations = get_config()
        experiment = self.name
        columns = filter(lambda x: 'cumsum' in x, df.columns)
        plt.figure()#figsize=(12.8, 9.6))
        if labels:
            for column, name, linestyle in zip(columns, labels, ['-.', '--', ':', '-']):
                df[column].plot(label=name, linestyle=linestyle, linewidth=5)
        else:
            for column, linestyle in zip(columns, ['-', '--', ':', '-.']*3):
                _, name, nr = column.split('_')
                if 'Random' in name:
                    name = 'naive agent' + ' ' + str(nr)
                elif 'Neural' in name:
                    name = 'neural agent' + ' ' + str(nr)
                elif 'Correcting' in name:
                    name = 'lingustic agent' + ' ' + str(nr)
                df[column].plot(label=name, linestyle=linestyle, linewidth=3)

        plt.tick_params(axis='both', which='major', labelsize=13)
        #plt.tick_params(axis='both', which='minor', labelsize=9)
        #plt.xlabel('scenario #', fontsize=20)
        plt.ylabel('regret', fontsize=20)
        title_fontsize = 20
        if dataset_label is None:
            plt.title('Cumulative Regret for {}'.format(experiment), fontsize=title_fontsize)
        else:
            plt.title('Cumulative Regret for {}'.format(dataset_label), fontsize=title_fontsize)
        plt.legend(loc='upper left', fontsize=18)
        plt.savefig('{}/plots/{}'.format(locations['results_location'], experiment + '.png'), dpi=200)
        plt.show()

    def get_agent(self, name):
        files = dict(self.results_files)
        return files[name].load_agent()

    def plot(self, discount=True, labels=[], dataset_label=None):
        df = self.to_df(discount=discount)
        self.plot_df(df, labels=labels, dataset_label=dataset_label)



    def test_colour_models(self, name, colour_thresh=0.5, draw=False):
        agent = self.get_agent(name)
        for cm in agent.colour_models.values():
            if draw:
                cm.draw()
            yield(cm.name, test_colour_model(cm, colour_thresh=colour_thresh))

    def print_colour_models(self, name, colour_thresh=0.5):
        for name, values in self.test_colour_models(name, colour_thresh=colour_thresh):
            print(name, values)


def extract_data(line):
    data = int(line.split(':')[1])
    return data


class ResultsFile(object):
    def __init__(self, config=None, name=None):
        if name is not None:

            self.name = name
            self.dir_ = os.path.split(name)[0]
            self.nr = int(os.path.split(name)[1].strip('experiment').strip('.out'))
            self.test_name = os.path.join(self.dir_, f'test{self.nr}.out')

        elif config is not None:
            suite = config['problem_name']
            agent = config['agent']
            if not isinstance(agent, str):
                agent = str(agent)
            threshold = str(config['threshold'])
            dir_ = ResultsFile.get_dir(suite, agent, threshold)
            # dir_ = 'results/{}/{}/{}'.format(suite, agent, threshold)
            self.dir_ = dir_

            os.makedirs(dir_, exist_ok=True)
            nr = ResultsFile.get_number(dir_)
            # nr = len(list(filter(lambda x: 'experiment' in x, os.listdir(dir_))))
            self.nr = nr
            self.name = os.path.join(dir_, f'experiment{nr}.out')
            self.test_name = os.path.join(self.dir_, f'test{self.nr}.out')
        else:
            raise AttributeError('No config or name given')

    def get_number(dir_):
        return len([x for x in os.listdir(dir_) if 'experiment' in x])


    def get_dir(suite, agent, threshold):
        return os.path.join(results_location, suite, agent, threshold)

    def get(filename):
        return ResultsFile(name=filename)

    def read(config):
        # rf = ResultsFile(name='')
        suite = config['scenario_suite']
        agent = config['agent']
        threshold = config['threshold']
        dir_ = ResultsFile.get_dir(suite, agent, threshold)
        # dir_ = 'results/{}/{}/{}'.format(suite, agent, threshold)

        nr = ResultsFile.get_number(dir_) - 1
        name = os.path.join(dir_, 'experiment{}.out'.format(nr))
        rf = ResultsFile(name=name)
        return rf


    def write_test(self, data):
        with open(self.test_name, 'a') as f:
            f.write(data)

    def write(self, data):
        with open(self.name, 'a') as f:
            f.write(data)

    def save_agent(self, agent):
        rel_path = os.path.relpath(self.dir_, results_location)
        save_dir = os.path.join(results_location, 'agents/results', rel_path)
        os.makedirs(save_dir, exist_ok=True)

        file_name = f'agent{self.nr}.pickle'
        save_location = os.path.join(save_dir, file_name)
        agent_type = type(agent)
        goal = agent.goal
        cms = {}
        for colour, cm in agent.colour_models.items():
            datas = (cm.data, cm.weights, cm.data_neg, cm.weights_neg)
            cms[colour] = datas

        with open(save_location, 'wb') as f:
            pickle.dump((agent_type, goal, cms), f)

        #
        # try:
        #     agent.priors.to_dict()
        # except AttributeError:
        #     pass
        #
        # try:
        #     with open(save_location, 'wb') as f:
        #         pickle.dump(agent, f)
        # except pickle.PicklingError:
        #     with open(save_location, 'wb') as f:
        #         agents.pickle_agent(agent, f)

    def load_agent(self):
        rel_path = os.path.relpath(self.dir_, results_location)
        save_dir = os.path.join(results_location, 'agents/results', rel_path)
        file_name = f'agent{self.nr}.pickle'
        save_location = os.path.join(save_dir, file_name)
        try:
            with open(save_location, 'rb') as f:
                return agents.load_agent(f)
        except (ValueError, TypeError):
            with open(save_location, 'rb') as f:
                return pickle.load(f)

    def read_file(self):
        rewards = []
        cumulative_rewards = []
        inference_times = []
        with open(self.name, 'r') as f:
            data = f.readlines()
        for line in data:
            if 'total reward' in line:
                pass
            elif 'cumulative reward' in line:
                datum = extract_data(line)
                cumulative_rewards.append(datum)
            elif 'reward' in line:
                datum = extract_data(line)
                rewards.append(datum)
            elif 'inference time' in line:
                inference_times = [int(x) for x in line.replace("inference time:", "").strip().split(',')]

        return rewards, cumulative_rewards, inference_times


    def to_df(self, discount=False):
        rewards, _, _ = self.read_file()
        df = pd.DataFrame(data={'rewards': rewards})
        if discount:
            df['rewards'] += 10
            df['rewards'] = df['rewards']/2
        df['cumsum'] = df['rewards'].cumsum()
        return df

    def plot_cumsum(self, discount=False, save_loc='test.png'):
        # plot_cumsum(self.name, discount=discount, save_loc=os.path.join(self.dir_, 'cumsum.png'))
        # df = to_df(self.name, discount=discount)
        # plt.figure()
        # df['cumsum'].plot()
        # plt.savefig(path.join(self.dir_, 'cumsum.png'))
        df = self.to_df(discount=discount)
        plt.figure()
        df['cumsum'].plot()
        plt.savefig(save_loc)


    def get_inference_times(self):
        _, _, inference_times = self.read_file()
        return inference_times
