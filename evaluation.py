
import pickle
import os
import webcolors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from colour_dict import colour_dict, simple_colour_dict


def extract_file(filename):
    with open(filename, 'rb') as f:
        try:
            colour_model = pickle.load(f)
        except UnicodeDecodeError:
            raise TypeError('Wrong file type, expected pickle file got {}'.format(os.path.splitext(filename)))
    return colour_model


def extract_experiment_parameters(filename):
    out = filename.strip('.pickle').split('_')
    if len(out) < 3:
        raise ValueError('unexpected input file format')
    else:
        return out


def name_to_rgb(name):
    rgb = webcolors.name_to_rgb(name)
    return np.array(rgb, dtype=np.float32) / 255


def colour_probs(colour_model, colour_dict, prior=0.5):
    output = {c:{c_i:-1 for c_i in cs} for c, cs in colour_dict.items()}
    for c, cs in colour_dict.items():
        for c_i in cs:
            p_c = colour_model.p(1, name_to_rgb(c_i), p_c=prior)
            output[c][c_i] = p_c
    return output


def colour_confusion(colour, results_dict, threshold=0.5):
    output = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    for c, cs in results_dict.items():
        for p in cs.values():
            if c == colour and p > threshold:
                output['tp'] += 1
            elif c == colour:
                output['fn'] += 1
            elif p > threshold:
                output['fp'] += 1
            else:
                output['tn'] += 1
    return output


def extract_data(line):
    data = int(line.split(':')[1])
    return data


def read_file(results_file):
    rewards = []
    cumulative_rewards = []
    with open(results_file, 'r') as f:
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
    return rewards, cumulative_rewards


def to_df(results_file, discount=False):
    rewards, _ = read_file(results_file)
    df = pd.DataFrame(data={'rewards': rewards})
    if discount:
        df['rewards'] += 10
    df['cumsum'] = df['rewards'].cumsum()
    return df


def plot_cumsum(results_file, discount=False, save_loc='test.png'):
    df = to_df(results_file, discount=discount)
    plt.figure()
    df['cumsum'].plot()
    plt.savefig(save_loc)


def df_experiment(dataset, threshold=0.7, discount=True, file_modifiers=''):
    files = os.listdir('results')
    results_files = filter(lambda x: dataset in x and str(threshold) in x and (file_modifiers in x or 'random' in x), files)
    results = {}
    for f in results_files:
        name = f.split('_')[0]
        rewards, cum_rewards = read_file('results/' + f)
        results[name] = rewards
    df = pd.DataFrame(data=results)
    if discount:
        for column in df.columns:
            df[column] += 10
            df['cumsum_{}'.format(column)] = df[column].cumsum()
    return df


def plot_df(df, experiment, file_modifiers=''):
    columns = filter(lambda x: 'cumsum' in x, df.columns)
    plt.figure()
    for column in columns:
        name = column.split('_')[1]
        if name == 'random':
            name = 'naive agent'
        elif name == 'correcting':
            name = 'lingustic agent'
        df[column].plot(label=name)
    plt.xlabel('scenario #', fontsize=13)
    plt.ylabel('cumulative reward', fontsize=13)
    plt.title('Cumulative reward for the {} dataset'.format(experiment), fontsize=16)
    plt.legend(loc='lower left', prop={'size': 10})
    plt.savefig('results/plots/' + experiment + file_modifiers + '.png')
    plt.show()

def load_agent(dataset, threshold=0.7, file_modifiers=''):
    with open('results/agents/correcting_{}_{}{}.pickle'.format(dataset, threshold, file_modifiers), 'rb') as f:
        agent = pickle.load(f)
    return agent

def test_colour_model(colour_model, colour_dict=colour_dict, colour_thresh=0.5):
    probs = colour_probs(colour_model, colour_dict)
    confusion = colour_confusion(colour_model.name, probs, colour_thresh)
    return confusion

def plot_colours(dataset, threshold=0.7, file_modifiers='', colour_dict=colour_dict, colour_thresh=0.5):
    agent = load_agent(dataset, threshold=threshold, file_modifiers=file_modifiers)
    for cm in agent.colour_models.values():
        probs = colour_probs(cm, colour_dict, prior=0.5)
        confusion = colour_confusion(cm.name, probs, colour_thresh)
        print(cm.name, confusion)
        cm.draw(save_location_basename=dataset)
