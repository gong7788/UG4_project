import configparser
import socket
import os
from os import path


try:
    config_location = os.environ['CONFIG-LOCATION']
except KeyError:
    config_location = '/home/mappelgren/Desktop/correcting-agent/config'


def get_config():
    config = configparser.ConfigParser()
    config.read(path.join(config_location, 'default.ini'))
    hostname = socket.gethostname()
    if 'inf.ed.ac.uk' in hostname:
        return config['inf.ed.ac.uk']
    elif 'mappelgren-HP-EliteDesk-800-G2-SFF' == hostname:
        return config['ubuntu']
    else:
        return config['laptop']


def add_config_option(config_file='experiments.ini', category='DEFAULT', option=None, value=None):
    """category (DEFAULT): the name of file to change
    option: which option to add
    value: the value of the option"""
    config = configparser.ConfigParser()
    config_file = path.join(config_location, config_file)
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
        add_config_option(config_file='experiments.ini', category=name, option=option, value=value)


def create_neural_experiment(name, options={}):
    for option, value in options.items():
        add_config_option(config_file='neural.ini', category=name, option=option, value=value)


def create_kde_experiment(name, options={}):
    for option, value in options.items():
        add_config_option(config_file='kde.ini', category=name, option=option, value=value)





def get_neural_config(name):
    config = configparser.ConfigParser()
    config_file = os.path.join(config_location, 'neural.ini')
    config.read(config_file)
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


def get_kde_config(config_name):
    model_config = configparser.ConfigParser()
    config_file = os.path.join(config_location, 'kde.ini')
    model_config.read(config_file)
    model_config = model_config[config_name]
    config_dict = {}
    config_dict['use_3d'] = model_config.getboolean('use_3d')
    config_dict['fix_bw'] = model_config.getboolean('fix_bw')
    config_dict['bw'] = model_config.getfloat('bw')
    config_dict['norm'] = model_config.getfloat('norm')
    config_dict['use_hsv'] = model_config.getboolean('use_hsv')
    config_dict['num_channels'] = model_config.getint('num_channels')

    return config_dict
