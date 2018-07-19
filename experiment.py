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
import pickle
from evaluation import plot_cumsum


def run_experiment(problem_dir, vis=True, threshold=0.7, Agent=agents.CorrectingAgent, file_modifiers=''):
    
    
    total_reward = 0
    problems = os.listdir(problem_dir)
    w = world.PDDLWorld('blocks-domain.pddl', '{}/{}'.format(problem_dir, problems[0]))
    teacher = TeacherAgent()
    agent = Agent(w, teacher=teacher, threshold=threshold)
    results_file = 'results/{}_{}_{}{}.out'.format(agent.name, problem_dir, threshold, file_modifiers)
    with open(results_file, 'w') as f:
        f.write('Results for {}\n'.format(problem_dir))
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
        # if np.isnan(np.sum(agent.colour_models[agent.colour_models.ke].mu0)):
        #     raise ValueError('NAN NAN NAN')
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

    # for c, cm in agent.colour_models.items():
    with open('results/agents/{}_{}_{}{}.pickle'.format(agent.name, problem_dir, threshold, file_modifiers), 'wb') as f:
        try:
            agent.priors.to_dict()
        except AttributeError:
            pass
        pickle.dump(agent, f)
        # cm.draw(save_location_basename=problem_dir)

    plot_cumsum(results_file, discount=True, save_loc='results/plots/{}_{}_{}.png'.format(agent.name, problem_dir, threshold))
    # print('red', agent.colour_models['red'].mu0, agent.colour_models['red'].sigma0)
    # print('blue', agent.colour_models['blue'].mu0, agent.colour_models['blue'].sigma0)
    # agent.colour_models['red'].draw(save_location_basename=problem_dir)
    # agent.colour_models['blue'].draw(save_location_basename=problem_dir)
