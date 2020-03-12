from correctingagent.experiments.experiment import _run_experiment
from correctingagent.agents.PGMAgent import PGMCorrectingAgent

_run_experiment("big_bijection_random_colours/big_bijection_random_colours12", agent=PGMCorrectingAgent, vis=False, colour_model_type="kde", no_correction_update=True, colour_model_config_name='fixed_bw_0.05', teacher_type=True, world_type='RandomColours')
