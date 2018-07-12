import world
import agents
from ff import Solved
import ff
import pddl_functions
from teacher import TeacherAgent
import os
import numpy as np
from scipy.stats import norm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd
import gc



def run_experiment(problem_dir, vis=True, threshold=0.7):
    results_file = 'results/{}_{}.out'.format(problem_dir, threshold)
    with open(results_file, 'w') as f:
        f.write('Results for {}\n'.format(problem_dir))
    total_reward = 0
    problems = os.listdir(problem_dir)
    w = world.PDDLWorld('blocks-domain.pddl', '{}/{}'.format(problem_dir, problems[0]))
    teacher = TeacherAgent()
    agent = agents.CorrectingAgent(w, teacher=teacher, threshold=threshold)
    for problem in problems:
        w = world.PDDLWorld('blocks-domain.pddl', '{}/{}'.format(problem_dir, problem))
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
                    print("T:", correction)
                    agent.get_correction(correction, a, args)
                    if vis:
                        w.draw()
                    break
                #else:
                #    agent.no_correction(a, args)
        if np.isnan(np.sum(agent.colour_models['red'].mu0)):
            raise ValueError('NAN NAN NAN')
        clear_output()
        total_reward += w.reward
        print('{} reward: {}'.format(problem, w.reward))
        gc.collect()
        with open(results_file, 'a') as f:
            f.write('{} reward: {}\n'.format(problem, w.reward))
            f.write('{} cumulative reward: {}\n'.format(problem, total_reward))
        
    print('total reward: {}'.format(total_reward))
    with open(results_file, 'a') as f:
        f.write('total reward: {}\n'.format(total_reward))

    print('red', agent.colour_models['red'].mu0, agent.colour_models['red'].sigma0)
    print('blue', agent.colour_models['blue'].mu0, agent.colour_models['blue'].sigma0)
    agent.colour_models['red'].draw(save_location_basename=problem_dir)
    agent.colour_models['blue'].draw(save_location_basename=problem_dir)


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
    df = pd.DataFrame(data={'rewards':rewards}) 
    if discount:
        df['rewards'] += 10
    df['cumsum'] = df['rewards'].cumsum()
    return df
    
def plot_cumsum(results_file, discount=False, save_loc='test.png'):
    df = to_df(results_file, discount=discount)
    df['cumsum'].plot()
    plt.savefig(save_loc)
    