from pathlib import Path
import correctingagent.world.world_generation as world_generation
from correctingagent.util.colour_dict import colour_dict
from correctingagent.util.colour_dict import fruit_dict
import random


name = 'fruit-k5-'
c = 0
r = 1

dataset_name = name + str(c) + 'c' + str(r) + 'r'

world_generation.generate_dataset_set_w_colour_count(5, 5, c, r, dataset_name,
                                        colour_dict=fruit_dict, use_random_colours=True,
                                        cc_num=2, cc_exact_num=None, num_towers=2,
                                        domain_name="fruitsworld")