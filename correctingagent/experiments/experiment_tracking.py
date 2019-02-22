import sqlalchemy
import pandas as pd
from .evaluation import ResultsFile, Experiment
import configparser
from ..util.util import get_config
import os

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
    config.read('{}/experiments.ini'.format(locations['config_location']))

    config = config[config_name]
    return ResultsFile.read(config)

def load_experiments(list_of_experiments, dataset):
    """plot the cumulative reward for a number of experiments listed on the same axis"""
    experiments_df = read_experiments()
    experiments = [get_baseline(dataset)]
    for experiment in list_of_experiments:
        experiments.append(get_results_file(experiments_df, experiment))
    return Experiment.from_results_files(experiments, dataset)
