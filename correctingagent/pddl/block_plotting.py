import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import pddl_functions
import webcolors
from ..util.colour_dict import colour_dict, colour_names
from ..world.colours import get_colour


# colors_ = [color for color in list(six.iteritems(colors.cnames)) if not ':' in color]
# colour_names = set([c for c, _ in colors_])
WIDTH = HEIGHT = 0.2

# colour_names = []
# for v in colour_dict.values():
#     colour_names.extend(v)


def generate_start_position(problem, width=0.2):
    objects = pddl_functions.get_objects(problem)
    tower_locations = [obj for obj in objects if obj[0] == 't' and 'tower' not in obj]
    blocks = [obj for obj in objects if 'b' in obj]

    y_pos = 0
    starting_positions = []
    x_pos = 0.2
    for t in tower_locations:
        starting_positions.append((x_pos, y_pos))
        x_pos += WIDTH*2
    for b in blocks:
        starting_positions.append((x_pos, y_pos))
        x_pos += WIDTH*2
    return starting_positions


def plot_blocks(posns, colours, height = 0.2, width=0.2, object_separation=0.1):
    n_c = len(colours)
    n_p = len(posns)
    posns = posns[n_p-n_c:]
    fig, ax = plt.subplots()

    for (x1, y1), c in zip(posns, colours):
        rectangle = Rectangle((x1, y1), width=width, height=height,
            edgecolor= (0,0,0), facecolor=c)
        ax.add_artist(rectangle)

    plt.xlim([0, 2*(width+object_separation)*len(posns)])
    plt.ylim([0, (height*len(posns)+object_separation)])
    plt.axis('off')

    plt.show()


def place_objects(objects, state, y_start):
    tower_locations = [obj for obj in objects if obj[0] == 't' and 'tower' not in obj]
    blocks = [obj for obj in objects if 'b' in obj]

    objects = tower_locations + blocks

    y_pos = {o:-1 for o in objects}
    x_pos = {o:x for o, (x, y) in zip(objects, y_start)}

    for o in objects:
        if state._predicate_holds(pddl_functions.Predicate('on-table', [o])) or 't' in o:
            y_pos[o] = 0
    while -1 in y_pos.values():
        for o in objects:
            for predicate in state.get_predicates(o):
                if predicate.name == 'on':
                    x, y = predicate.args
                    if y_pos[y] != -1:
                        if 't' in y:
                            y_pos[x] = y_pos[y]
                        else:
                            y_pos[x] = y_pos[y] + HEIGHT
                        x_pos[x] = x_pos[y]
    return [(x_pos[o], y_pos[o]) for o in objects]


