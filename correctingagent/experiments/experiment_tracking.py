from pathlib import Path

import sqlalchemy
import pandas as pd
from .evaluation import ResultsFile, Experiment
import configparser
from ..util.util import get_config
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, ttest_rel


def read_experiments():
    config = get_config()
    engine = sqlalchemy.create_engine('sqlite:///{}/experiments.db'.format(config['db_location']))
    df = pd.read_sql('experiments', index_col='index', con=engine)
    return df


def get_results_file(df, index):
    name = df.at[index, 'experiment_file']
    split_name = name.split('/')
    if split_name[0] == 'results':
        locations = get_config()
        name = os.path.join(locations['results_location'], *split_name[1:])
    return ResultsFile(name=name)


def get_baseline(dataset):
    """Returns the Results File of the random agent for this dataset"""
    locations = get_config()
    config_name = dataset + '_random'
    config = configparser.ConfigParser()
    config.read(Path(locations["config_location"]) / 'experiments.ini')

    config = config[config_name]
    return ResultsFile.read(config)


def load_experiments(list_of_experiments, dataset):
    """plot the cumulative reward for a number of experiments listed on the same axis"""
    experiments_df = read_experiments()
    experiments = [get_baseline(dataset)]
    for experiment in list_of_experiments:
        experiments.append(get_results_file(experiments_df, experiment))
    return Experiment.from_results_files(experiments, dataset)


def read_big():
    config = get_config()
    engine = sqlalchemy.create_engine('sqlite:///{}/experiments.db'.format(config['db_location']))
    df = pd.read_sql('big', index_col='index', con=engine)
    return df


def get_experiments(big_id):
    config = get_config()
    engine = sqlalchemy.create_engine(f'sqlite:///{config["db_location"]}/experiments.db')

    rels = pd.read_sql('rels', index_col='index', con=engine)
    experiments = pd.read_sql('experiments', index_col='index', con=engine)

    relevant_ids = rels[rels.big_id == big_id].experiment_id.values
    relevant_experiments = experiments.iloc[relevant_ids]
    experiment_ids = relevant_experiments[relevant_experiments.status == 'done'].index.values
    return experiments.iloc[experiment_ids]


def load_big_experiments(list_of_experiments):
    """plot the cumulative reward for a number of experiments listed on the same axis"""
    if isinstance(list_of_experiments, pd.core.frame.DataFrame):
        list_of_experiments = list_of_experiments.index
    experiments_df = read_experiments()
    # experiments = [get_baseline(dataset)]
    experiments = []
    for experiment in list_of_experiments:
        rf = get_results_file(experiments_df, experiment)
        experiments.append(rf.read_file()[0])
    return np.array(experiments)  # [[experiment-i]]


def get_discounted_data(big_id):
    experiments = get_experiments(big_id)
    raw_data = load_big_experiments(experiments)
    discounted_data = -0.5 * (raw_data + 10)
    return discounted_data


def get_cumsum(big_id):
    discounted_data = get_discounted_data(big_id)
    cumsum = np.cumsum(discounted_data, axis=1)
    return cumsum


def get_mean(big_id):
    cumsum = get_cumsum(big_id)
    mean = np.mean(cumsum, axis=0)

    return mean


def plot_big_experiments(list_of_experiments, labels, title=''):
    for experiment, label, marker in zip(list_of_experiments, labels,
                                         ['--', '-', '-.', ':']):  #::['_', 'x', '+', '|']):
        cumsum = get_mean(experiment)
        plt.plot(range(1, 51), cumsum, label=label, linestyle=marker, linewidth=3)  # marker=marker)
    plt.ylabel('regret', fontsize=13)
    plt.xlabel('scenario', fontsize=13)

    plt.legend(fontsize=12)
    plt.title(title, fontsize=14)
    plt.show()


def do_ttest(old, old_nocorr, new, new_nocorr):
    total_old = get_cumsum(old)[:, -1]
    total_new = get_cumsum(new)[:, -1]
    total_old_no_corr = get_cumsum(old_nocorr)[:, -1]
    total_new_no_corr = get_cumsum(new_nocorr)[:, -1]

    print('old vs new')
    print(ttest_ind(total_old, total_new), np.mean(total_old), np.mean(total_new))
    print(ttest_ind(total_old, total_new, equal_var=False))
    print('old vs old no corr')
    print(ttest_ind(total_old, total_old_no_corr), np.mean(total_old), np.mean(total_old_no_corr))
    print(ttest_ind(total_old, total_old_no_corr, equal_var=False))
    print('old no corr vs new no corr')
    print(ttest_ind(total_old_no_corr, total_new_no_corr), np.mean(total_old_no_corr), np.mean(total_new_no_corr))
    print(ttest_ind(total_old_no_corr, total_new_no_corr, equal_var=False))
    print('new vs new no corr')
    print(ttest_ind(total_new, total_new_no_corr), np.mean(total_new), np.mean(total_new_no_corr))


def do_ttest_rel(old, old_nocorr, new, new_nocorr):
    total_old = get_cumsum(old)[:, -1]
    total_new = get_cumsum(new)[:, -1]
    total_old_no_corr = get_cumsum(old_nocorr)[:, -1]
    total_new_no_corr = get_cumsum(new_nocorr)[:, -1]

    total_old = total_old[:len(total_new)]
    print(total_new_no_corr)
    print(total_new)
    print(np.min(total_old), np.max(total_old), np.min(total_new), np.max(total_new), np.var(total_old_no_corr),
          np.var(total_new_no_corr))

    print('old vs new')
    print(ttest_rel(total_old, total_new), np.mean(total_old), np.mean(total_new))
    print(ttest_rel(total_old, total_new))
    print('old vs old no corr')
    print(ttest_rel(total_old, total_old_no_corr), np.mean(total_old), np.mean(total_old_no_corr))
    print(ttest_rel(total_old, total_old_no_corr))
    print('old no corr vs new no corr')
    print(ttest_rel(total_old_no_corr, total_new_no_corr), np.mean(total_old_no_corr), np.mean(total_new_no_corr))
    print(ttest_rel(total_old_no_corr, total_new_no_corr))
    print('new vs new no corr')
    print(ttest_rel(total_new, total_new_no_corr), np.mean(total_new), np.mean(total_new_no_corr))
