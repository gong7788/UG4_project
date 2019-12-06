from pathlib import Path

from correctingagent.util.database import BigExperimentDB, ExperimentDB, JoinDB
from ..world import world
from ..agents import agents, PGMAgent
from ..agents.agents import CorrectingAgent, RandomAgent, PerfectColoursAgent, NoLanguageAgent
from ..agents.teacher import TeacherAgent, ExtendedTeacherAgent
import os
import pandas as pd
from .evaluation import ResultsFile
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


def get_agent(config):
    agent_dict = {'agents.CorrectingAgent': CorrectingAgent,
                  'agents.RandomAgent': RandomAgent,
                  'RandomAgent': RandomAgent,
                  'agents.PerfectColourAgent': PerfectColoursAgent,
                  'agents.NoLanguageAgent': NoLanguageAgent,
                  'NoLanguageAgent': NoLanguageAgent,
                  'PGMAgent': PGMAgent.PGMCorrectingAgent,
                  'InitialAdvice': PGMAgent.ClassicalAdviceBaseline}
    return agent_dict[config['agent']]


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
            pass
            # confusion = test_colour_model(cm)
            # for key, value in confusion.items():
            #     self.cm_results[(cm.name, key)].append(value)

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

        df.to_pickle(os.path.join(self.dir_, f'cm_params{self.nr}.pickle'))


def create_agent(agent, colour_model_config_name, colour_model_type, w, teacher,
                 threshold, update_negative, update_once, debug_agent, domain_file, simplified_colour_count=False):
    if agent in [agents.CorrectingAgent, agents.NoLanguageAgent, PGMAgent.PGMCorrectingAgent]:
        if colour_model_type == 'kde':
            if colour_model_config_name is None:
                colour_model_config_name = 'DEFAULT'
            colour_model_config = get_kde_config(colour_model_config_name)
        else:
            colour_model_config = {}
        agent = agent(w, teacher=teacher, threshold=threshold, update_negative=update_negative, update_once=update_once,
                      colour_model_type=colour_model_type, model_config=colour_model_config, debug=debug_agent, domain_file=domain_file, simplified_colour_count=simplified_colour_count)
    else:
        agent = agent(w, teacher=teacher, threshold=threshold)
    print(colour_model_config_name)
    print(colour_model_config)
    return agent


def do_scenario(agent, world_scenario, vis=False, no_correction_update=False, break_on_correction=False):
    if vis:
        world_scenario.draw()
    agent.new_world(world_scenario)
    while not world_scenario.test_success():
        plan = agent.plan()
        for a, args in plan:
            if a == 'reach-goal':
                break
            world_scenario.update(a, args)
            if vis:
                world_scenario.draw()
            correction = agent.teacher.correction(world_scenario, args)
            if correction:
                if break_on_correction:
                    return
                # logger.info("T: " + correction)
                print(f"T: {correction}")
                agent.get_correction(correction, a, args)
                if vis:
                    world_scenario.draw()
                break
            elif no_correction_update:
                agent.no_correction(a, args)


def _run_experiment(problem_name=None, threshold=0.5, update_negative=False, agent=None, vis=False, update_once=True,
                    colour_model_type='KDE', no_correction_update=False, debug=False, colour_model_config_name='DEFAULT',
                    new_teacher=False, results_file=None, world_type='PDDL', use_hsv=False, debug_agent=None,
                    domain_file='blocks-domain.pddl', simplified_colour_count=False, **kwargs):
    config = get_config()
    data_location = Path(config['data_location'])

    total_reward = 0
    problem_dir = data_location / problem_name

    problems = os.listdir(problem_dir)
    num_problems = len(problems)
    if world_type == 'RandomColours':
        num_problems = int(num_problems / 2)

    w = world.get_world(problem_name, 1, world_type=world_type, domain_file=domain_file, use_hsv=use_hsv)

    print(colour_model_config_name)

    if new_teacher:
        print("NEW TEACHER!")
        teacher = ExtendedTeacherAgent()
    else:
        print("old teacher :(")
        teacher = TeacherAgent()

    agent = create_agent(agent, colour_model_config_name, colour_model_type, w, teacher,
                         threshold, update_negative, update_once, debug_agent, domain_file, simplified_colour_count)

    if results_file is not None:
        results_file.write(f'Results for {problem_name}\n')

    for i in range(num_problems):
        w = world.get_world(problem_name, i+1, world_type=world_type, domain_file=domain_file)
        do_scenario(agent, w, vis=vis, no_correction_update=no_correction_update)

        if debug and not 'Random' in config['agent']:
            debugger.cm_confusion(agent)
            debugger.update_cm_params(agent)

        total_reward += w.reward
        print(f'{problem_name} reward: {w.reward}')
        if results_file is not None:
            results_file.write(f'{problem_name} reward: {w.reward}\n')
            results_file.write(f'{problem_name} cumulative reward: {total_reward}\n')
    if results_file is not None:
        results_file.write(f'total reward: {total_reward}\n')

    if debug and not 'Random' in config['agent']:
        debugger.save_confusion()
        debugger.save_params()

    if not isinstance(agent, RandomAgent):
        results_file.save_agent(agent)

    return results_file.name


def run_experiment(config_name='DEFAULT', debug=False, colour_model_config='DEFAULT', debug_agent=None):
    if debug:
        agent_logger.setLevel(logging.DEBUG)
    config_dict = get_experiment_config(config_name, colour_model_config)
    results_file = ResultsFile(config=config)
    return _run_experiment(results_file=results_file, **config_dict)


def get_experiment_config(config_name, colour_model_config):
    config = configparser.ConfigParser()
    config_file = os.path.join(config_location, 'experiments.ini')
    config.read(config_file)
    config = config[config_name]
    colour_model_conf = get_kde_config(colour_model_config)
    config_dict = {
        "problem_name": config['scenario_suite'],
        "threshold": config.getfloat('threshold'),
        "update_negative": config.getboolean('update_negative'),
        "agent": get_agent(config),
        "vis": config.getboolean('visualise'),
        "update_once": config.getboolean('update_once'),
        "colour_model_type": config['colour_model_type'],
        "no_correction_update": config.getboolean('no_correction_update'),
        "new_teacher": config.getboolean('new_teacher'),
        "world_type": config['world_type'],
        "use_hsv": colour_model_conf['use_hsv'],
        "domain_file": config['domain_name'],
        "use_metric_ff": config.getboolean('use_metric_ff'),
        "simplified_colour_count": config.getboolean('simplified_colour_count'),
        "colour_model_config_name": colour_model_config
    }
    return config_dict


def add_big_experiment(config_name, colour_model_config, debug=False, vis=False):

    big_db = BigExperimentDB()

    big_id = big_db.add_experiment(config_name)

    config_dict = get_experiment_config(config_name, colour_model_config)
    config_dict['vis'] = vis
    data_directories = os.listdir(os.path.join(data_location, config_dict['problem_name']))

    experiment_base_name = config_dict['problem_name']

    for data_dir in tqdm(data_directories):

        experiment_name = os.path.join(experiment_base_name, data_dir)

        config_dict['problem_name'] = experiment_name

        results_file = ResultsFile(config=config_dict)

        experiment_id = big_db.experiment_db.add_experiment(experiment_name, colour_model_config)

        big_db.join_db.add_experiment(big_id, experiment_id)
        try:
            results_file = _run_experiment(results_file=results_file, **config_dict)
        except Exception as e:
            big_db.experiment_db.update_entry(experiment_id, experiment_file=None, status='ERROR')
            big_db.update_entry(big_id, status='ERROR')
            raise e
        else:
            big_db.experiment_db.update_entry(experiment_id, experiment_file=results_file, status='done')

    big_db.update_entry(big_id, status='done')


def continue_big_experiment(big_id, debug=False):

    big_db = BigExperimentDB()
    experiment_df = big_db.experiment_db.get_df()
    join_db = JoinDB()

    big_df = big_db.get_df()
    config_name = big_df.loc[big_id].values[0]

    join_df = big_db.join_db.get_df()
    experiment_ids = join_df[join_df.big_id == big_id].experiment_id.values

    relevant_experiments = experiment_df.loc[experiment_ids]
    done_experiments = relevant_experiments[relevant_experiments.status == 'done']

    colour_model_config = done_experiments.neural_config.values[0]
    config_dict = get_experiment_config(config_name, colour_model_config)
    data_directories = os.listdir(os.path.join(data_location, config_dict['problem_name']))

    done_experiment_dirs = done_experiments.config_name.values
    experiments_to_run = [data_loc for data_loc in data_directories if f"{config_dict['problem_name']}/{data_loc}" not in done_experiment_dirs]

    experiment_dir = config_dict['problem_name']

    for data_dir in tqdm(experiments_to_run):

        experiment_name = os.path.join(experiment_dir, data_dir)

        experiment_id = big_db.experiment_db.add_experiment(experiment_name, colour_model_config)
        join_db.add_experiment(big_id, experiment_id)

        config_dict['problem_name'] = experiment_name

        results_file = ResultsFile(config=config_dict)

        try:
            results_file = _run_experiment(results_file=results_file, **config_dict)
        except Exception as e:
            big_db.experiment_db.update_entry(experiment_id, experiment_file=None, status='ERROR')
            big_db.update_entry(big_id, status='ERROR')
            raise e
        else:
            big_db.experiment_db.update_entry(experiment_id, experiment_file=results_file, status='done')

    big_db.update_entry(big_id, status='done')


def add_experiment(config_name, neural_config, debug=False, new_teacher=False):
    experiment_db = ExperimentDB()

    experiment_id = experiment_db.add_experiment(config_name, neural_config)
    try:
        results_file = run_experiment(config_name=config_name, colour_model_config=neural_config, debug=debug, new_teacher=new_teacher)
    except Exception as e:
        experiment_db.update_entry(experiment_id, experiment_file=None, status='Error')
        raise e
    else:
        experiment_db.update_entry(experiment_id, experiment_file=results_file, status='done')

