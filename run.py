from experiment import run_experiment
import logging
from agents import RandomAgent
import argparse

parser = argparse.ArgumentParser(description='Run experiment with given parameters')
parser.add_argument('dataset', type=str,
                    help='what dataset to run experiment on')
parser.add_argument('file_modifier', type=str,
                    help='what to add at the end of result files')
parser.add_argument('visualise', type=bool, nargs='?',
                    help='whether to visualise the actions taken', default=False)

args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

print('Running experiment for dataset: {} with visualisation: {}.'.format(args.dataset, args.visualise))
print('Storing Results in {}'.format(args.file_modifier))

run_experiment(args.dataset, vis=args.visualise, file_modifiers=args.file_modifier)

#run_experiment('onerule', vis=False, file_modifiers='_train_negative3')
#run_experiment('simplecolours', vis=False, file_modifiers='_train_negative3')
#run_experiment('tworules', vis=False, file_modifiers='_train_negative4')
#run_experiment('bijection', vis=False, file_modifiers='_train_negative3')
#run_experiment('maroon2', vis=False, file_modifiers='_train_negative4')

#run_experiment('onerule', vis=False, Agent=RandomAgent)
#run_experiment('simplecolours', vis=False, Agent=RandomAgent)
#run_experiment('tworules', vis=False, Agent=RandomAgent)
#run_experiment('bijection', vis=False, Agent=RandomAgent)
#run_experiment('maroon', vis=False, Agent=RandomAgent)


#run_experiment('maroon2', vis=False)
#run_experiment('maroon2', vis=False, Agent=RandomAgent)

