import configparser
import socket

def get_config():
    config = configparser.ConfigParser()
    config.read('config/default.ini')
    if 'inf.ed.ac.uk' in socket.gethostname():
        return config['inf.ed.ac.uk']
    else:
        return config['laptop']

def add_config_option(config_file='config/experiments.ini', category='DEFAULT', option=None, value=None):
    """category (DEFAULT): the name of file to change
    option: which option to add
    value: the value of the option"""
    config = configparser.ConfigParser()
    config.read(config_file)

    try:
        config[category][option] = str(value)
    except KeyError:
        config[category] = {option: str(value)}

    with open(config_file, 'w') as configfile:
        config.write(configfile)

def create_experiment(name, options={}):
    """Creates the config for an experiment

    name: name of experiment in config
    options: the config options for the experiment
    """
    for option, value in options.items():
        add_config_option(config_file='config/experiments.ini', category=name, option=option, value=value)

def create_neural_experiment(name, options={}):
    for option, value in options.items():
        add_config_option(config_file='config/neural.ini', category=name, option=option, value=value)
