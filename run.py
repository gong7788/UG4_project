from experiment import run_experiment
import logging
from agents import RandomAgent
logger = logging.getLogger()
logger.setLevel(logging.ERROR)


#run_experiment('onerule', vis=False, file_modifiers='_train_negative3')
#run_experiment('simplecolours', vis=False, file_modifiers='_train_negative3')
run_experiment('tworules', vis=False, file_modifiers='_train_negative4')
#run_experiment('bijection', vis=False, file_modifiers='_train_negative3')
#run_experiment('maroon2', vis=False, file_modifiers='_train_negative4')

#run_experiment('onerule', vis=False, Agent=RandomAgent)
#run_experiment('simplecolours', vis=False, Agent=RandomAgent)
#run_experiment('tworules', vis=False, Agent=RandomAgent)
#run_experiment('bijection', vis=False, Agent=RandomAgent)
#run_experiment('maroon', vis=False, Agent=RandomAgent)


#run_experiment('maroon2', vis=False)
#run_experiment('maroon2', vis=False, Agent=RandomAgent)

