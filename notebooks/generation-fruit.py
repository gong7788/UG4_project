from pathlib import Path
import correctingagent.world.world_generation as world_generation
from correctingagent.util.colour_dict import colour_dict
from correctingagent.util.colour_dict import fruit_dict
import random

dataset_name = 'fruit-4'

world_generation.generate_dataset_set_w_colour_count(5, 5, 1,
                                        0, 'fruit-4-test',
                                        colour_dict=fruit_dict, use_random_colours=True,
                                        cc_num=2, cc_exact_num=None, num_towers=2,
                                        domain_name="fruitsworld")