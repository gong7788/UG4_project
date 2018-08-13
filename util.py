import configparser
import socket

def get_config():
    config = configparser.ConfigParser()
    config.read('config/default.ini')
    if 'inf.ed.ac.uk' in socket.gethostname():
        return config['inf.ed.ac.uk']
    else:
        return config['laptop']
