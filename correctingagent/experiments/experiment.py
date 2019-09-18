from ..world import world
from ..agents import agents, PGMAgent
from ..agents.teacher import TeacherAgent, ExtendedTeacherAgent
import os
import pandas as pd
import pickle
from .evaluation import test_colour_model, ResultsFile, get_agent
import configparser
import logging
from collections import defaultdict
from ..models import prob_model
import sqlalchemy
from ..util.util import config_location, get_config, get_neural_config, get_kde_config
from tqdm import tqdm

handler = logging.StreamHandler()

agent_logger = logging.getLogger('agent')
agent_logger.setLevel(logging.INFO)
agent_logger.addHandler(handler)

logger = logging.getLogger('dialogue')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

config = get_config()
data_location = config['data_location']
db_location = config['db_location']

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



def _run_experiment(problem_name, threshold, update_negative, Agent, vis, update_once, colour_model_type, no_correction_update, debug, neural_config, new_teacher, results_file):

    total_reward = 0
    problem_dir = os.path.join(data_location, problem_name)
    problems = os.listdir(problem_dir)
    w = world.PDDLWorld('blocks-domain.pddl', os.path.join(problem_dir, problems[0]))
    if new_teacher:
        teacher = ExtendedTeacherAgent()
    else:
        teacher = TeacherAgent()
    if Agent in [agents.NeuralCorrectingAgent]:
        config_dict = get_neural_config(neural_config)
        agent = Agent(w, teacher=teacher, **config_dict)
    elif Agent in [agents.CorrectingAgent, agents.NoLanguageAgent, PGMAgent.PGMCorrectingAgent]:
        if colour_model_type == 'kde':
            if neural_config is None:
                neural_config = 'DEFAULT'
            model_config = get_kde_config(neural_config)
        else:
            model_config = {}
        agent = Agent(w, teacher=teacher, threshold=threshold, update_negative=update_negative, update_once=update_once, colour_model_type=colour_model_type, model_config=model_config)
    else:
        agent = Agent(w, teacher=teacher, threshold=threshold)


    results_file.write('Results for {}\n'.format(problem_name))
    for problem in problems:
        w = world.PDDLWorld('blocks-domain.pddl', os.path.join(problem_dir, problem))
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
                elif no_correction_update:
                    agent.no_correction(a, args)
        if debug and not 'Random' in config['agent']:
            debugger.cm_confusion(agent)
            debugger.update_cm_params(agent)



        total_reward += w.reward
        print('{} reward: {}'.format(problem, w.reward))

        results_file.write('{} reward: {}\n'.format(problem, w.reward))
        results_file.write('{} cumulative reward: {}\n'.format(problem, total_reward))


    results_file.write('total reward: {}\n'.format(total_reward))



    if debug and not 'Random' in config['agent']:
        debugger.save_confusion()
        debugger.save_params()




    results_file.save_agent(agent)

    return results_file.name


def run_experiment(config_name='DEFAULT', debug=False, neural_config='DEFAULT', new_teacher=False):

    if debug:
        agent_logger.setLevel(logging.DEBUG)

    config = configparser.ConfigParser()
    config_file = os.path.join(config_location, 'experiments.ini')
    config.read(config_file)
    config = config[config_name]
    problem_name = config['scenario_suite']
    threshold = config.getfloat('threshold')
    update_negative = config.getboolean('update_negative')
    Agent = get_agent(config)
    vis = config.getboolean('visualise')
    update_once = config.getboolean('update_once')
    colour_model_type = config['colour_model_type']
    no_correction_update = config.getboolean('no_correction_update')

    results_file = ResultsFile(config=config)

    return _run_experiment(problem_name, threshold, update_negative, Agent, vis, update_once, colour_model_type, no_correction_update, debug, neural_config, new_teacher, results_file)


def add_big_experiment(config_name, neural_config, debug=False):
    #big_db = os.path.join(db_location, 'big.db')
    experiment_db = os.path.join(db_location, 'experiments.db')
    #rels_db = os.path.join(db_location, 'rels.db')

    #big_engine = sqlalchemy.create_engine('sqlite:///' + big_db)
    engine = sqlalchemy.create_engine('sqlite:///' + experiment_db)
    #rels_engine = sqlalchemy.create_engine('sqlite:///' + rels_db)


    df = pd.read_sql('big', index_col='index', con=engine)
    df = df.append({'experiment_name':config_name, 'status':'running'}, ignore_index=True)
    big_id = df.index[-1]
    df.to_sql('big', con=engine, if_exists='replace')





    config = configparser.ConfigParser()
    config_file = os.path.join(config_location, 'experiments.ini')
    config.read(config_file)
    config = config[config_name]
    problem_name = config['scenario_suite']
    threshold = config.getfloat('threshold')
    update_negative = config.getboolean('update_negative')
    Agent = get_agent(config)
    vis = config.getboolean('visualise')
    update_once = config.getboolean('update_once')
    colour_model_type = config['colour_model_type']
    no_correction_update = config.getboolean('no_correction_update')
    new_teacher = config.getboolean('new_teacher')

    data_directories = os.listdir(os.path.join(data_location, problem_name))

    for data_dir in tqdm(data_directories):

        experiment_name = os.path.join(problem_name, data_dir)
        config['scenario_suite'] = experiment_name
        results_file = ResultsFile(config=config)

        experiments = pd.read_sql('experiments', index_col='index', con=engine)
        experiments = experiments.append({'config_name':experiment_name, 'neural_config':neural_config, 'status':'running'}, ignore_index=True)
        experiment_id = experiments.index[-1]
        experiments.to_sql('experiments', con=engine, if_exists='replace')
        #print(experiments)

        rels = pd.read_sql('rels', index_col='index', con=engine)
        rels = rels.append({'big_id':big_id, 'experiment_id':experiment_id}, ignore_index=True)
        rels.to_sql('rels', con=engine, if_exists='replace')
        #print(rels)
        try:
            results_file = _run_experiment(experiment_name, threshold, update_negative, Agent, vis, update_once, colour_model_type, no_correction_update, debug, neural_config, new_teacher, results_file)
        except Exception as e:
            experiments = pd.read_sql('experiments', index_col='index', con=engine)
            experiments.at[experiment_id, 'experiment_file'] = None
            experiments.at[experiment_id, 'status'] = 'ERROR'
            experiments.to_sql('experiments', con=engine, if_exists='replace')
            df = pd.read_sql('big', index_col='index', con=engine)
            df.at[big_id, 'status'] = 'ERROR'
            df.to_sql('big', con=engine, if_exists='replace')
            raise e
        else:
            experiments = pd.read_sql('experiments', index_col='index', con=engine)
            experiments.at[experiment_id, 'experiment_file'] = results_file
            experiments.at[experiment_id, 'status'] = 'done'
            experiments.to_sql('experiments', con=engine, if_exists='replace')

    df = pd.read_sql('big', index_col='index', con=engine)
    df.at[big_id, 'status'] = 'done'
    df.to_sql('big', con=engine, if_exists='replace')






def continue_big_experiment(big_id, debug=False):
    experiment_db = os.path.join(db_location, 'experiments.db')

    engine = sqlalchemy.create_engine('sqlite:///' + experiment_db)

    big = pd.read_sql('big', index_col='index', con=engine)
    experiments = pd.read_sql('experiments', index_col='index', con=engine)
    rels = pd.read_sql('rels', index_col='index', con=engine)

    config = configparser.ConfigParser()
    config_file = os.path.join(config_location, 'experiments.ini')


    config_name = big.iloc[big_id].values[0]
    config.read(config_file)
    config = config[config_name]
    problem_name = config['scenario_suite']
    threshold = config.getfloat('threshold')
    update_negative = config.getboolean('update_negative')
    Agent = get_agent(config)
    vis = config.getboolean('visualise')
    update_once = config.getboolean('update_once')
    colour_model_type = config['colour_model_type']
    no_correction_update = config.getboolean('no_correction_update')
    new_teacher = config.getboolean('new_teacher')


    data_directories = os.listdir(os.path.join(data_location, problem_name))

    experiment_ids = rels[rels.big_id == big_id].experiment_id.values
    relevant_experiments = experiments.iloc[experiment_ids]
    done_experiments = relevant_experiments[relevant_experiments.status == 'done']

    neural_config = done_experiments.neural_config.values[0]
    done_experiment_dirs = done_experiments.config_name.values
    experiments_to_run = [data_loc for data_loc in data_directories if "{}/{}".format(problem_name, data_loc) not in done_experiment_dirs]

    for data_dir in tqdm(experiments_to_run):

        experiment_name = os.path.join(problem_name, data_dir)
        config['scenario_suite'] = experiment_name
        results_file = ResultsFile(config=config)

        experiments = pd.read_sql('experiments', index_col='index', con=engine)
        experiments = experiments.append({'config_name':experiment_name, 'neural_config':neural_config, 'status':'running'}, ignore_index=True)
        experiment_id = experiments.index[-1]
        experiments.to_sql('experiments', con=engine, if_exists='replace')
        #print(experiments)

        rels = pd.read_sql('rels', index_col='index', con=engine)
        rels = rels.append({'big_id':big_id, 'experiment_id':experiment_id}, ignore_index=True)
        rels.to_sql('rels', con=engine, if_exists='replace')
        #print(rels)
        try:
            results_file = _run_experiment(experiment_name, threshold, update_negative, Agent, vis, update_once, colour_model_type, no_correction_update, debug, neural_config, new_teacher, results_file)
        except Exception as e:
            experiments = pd.read_sql('experiments', index_col='index', con=engine)
            experiments.at[experiment_id, 'experiment_file'] = None
            experiments.at[experiment_id, 'status'] = 'ERROR'
            experiments.to_sql('experiments', con=engine, if_exists='replace')
            df = pd.read_sql('big', index_col='index', con=engine)
            df.at[big_id, 'status'] = 'ERROR'
            df.to_sql('big', con=engine, if_exists='replace')
            raise e
        else:
            experiments = pd.read_sql('experiments', index_col='index', con=engine)
            experiments.at[experiment_id, 'experiment_file'] = results_file
            experiments.at[experiment_id, 'status'] = 'done'
            experiments.to_sql('experiments', con=engine, if_exists='replace')

    df = pd.read_sql('big', index_col='index', con=engine)
    df.at[big_id, 'status'] = 'done'
    df.to_sql('big', con=engine, if_exists='replace')

def add_experiment(config_name, neural_config, debug=False, new_teacher=False):
    experiment_db = os.path.join(db_location, 'experiments.db')

    engine = sqlalchemy.create_engine('sqlite:///' + experiment_db)
    df = pd.read_sql('experiments', index_col='index', con=engine)

    df = df.append({'config_name':config_name, 'neural_config':neural_config, 'status':'running'}, ignore_index=True)
    df.to_sql('experiments', con=engine, if_exists='replace')
    try:
        results_file = run_experiment(config_name=config_name, neural_config=neural_config, debug=debug, new_teacher=new_teacher)
    except Exception as e:
        df = pd.read_sql('experiments', index_col='index', con=engine)
        last_label = df.index[-1]
        df.at[last_label, 'experiment_file'] = None
        df.at[last_label, 'status'] = 'ERROR'
        df.to_sql('experiments', con=engine, if_exists='replace')
        raise e
    else:
        df = pd.read_sql('experiments', index_col='index', con=engine)
        last_label = df.index[-1]
        df.at[last_label, 'experiment_file'] = results_file
        df.at[last_label, 'status'] = 'done'
        df.to_sql('experiments', con=engine, if_exists='replace')
