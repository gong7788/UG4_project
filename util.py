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
    config = configparser.ConfigParser()
    config.read(config_file)

    try:
        config[category][option] = value
    except KeyError:
        config[category] = {option: value}

    with open(config_file, 'w') as configfile:
        config.write(configfile)

def create_experiment(name, options={}):
    for option, value in options.items():
        add_config_option(config_file='config/experiments.ini', category=name, option=option, value=value)
