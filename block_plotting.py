import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import pddl_functions, ff
import copy
import random
import data

WIDTH = HEIGHT = 0.2

def generate_start_position(problem, width=0.2):
    objects = pddl_functions.get_objects(problem)
    y_pos = 0
    starting_positions = []
    x_pos = 0.2
    for o in objects:
        starting_positions.append((x_pos, y_pos))
        x_pos += WIDTH*2
    return starting_positions


def plot_blocks(posns, colours):
    fig, ax = plt.subplots()
    resolution = 50

    height = width = 0.2
    colour_dict = {'blue':(0,0,1),
                   'yellow':(1,1,0),
                   'red':(1,0,0),
                   'green':(0, 1, 0)}
    patches = []
    for (x1, y1), c in zip(posns, colours):
        rectangle = Rectangle((x1, y1), width=WIDTH, height=HEIGHT,
            edgecolor= (0,0,0), facecolor=c)
        #patches.append(rectangle)
        ax.add_artist(rectangle)
    #p = PatchCollection(patches, match_original=True)

    #ax.add_collection(p)
    #p.set_array(np.array(colours))
    plt.xlim(0, 2.2)
    plt.ylim(0, 2.2)
    #fig.colorbar(p, ax=ax)
    plt.show()

def get_predicates(objects, state, obscure=False):
    if not(obscure):
        return {o:{p.name:p for p in pddl_functions.get_predicates(o, state)} for o in objects}
    else:
        available_colours = ['red', 'blue', 'green', 'yellow']
        return {o:{p.name:p for p in pddl_functions.get_predicates(o, state) if p.name not in available_colours} for o in objects}


def place_objects(objects, state, y_start):
    y_pos = {o:-1 for o in objects}
    x_pos = {o:x for o, (x, y) in zip(objects, y_start)}
    predicates = get_predicates(objects, state)
    for o in objects:
        if 'on-table' in predicates[o].keys():
            y_pos[o] = 0
    while -1 in y_pos.values():
        for o in objects:
            if 'on' in predicates[o].keys():
                x, y = map(lambda x: x.arg_name, predicates[o]['on'].args.args)
                if y_pos[y] != -1:
                    y_pos[x] = y_pos[y] + HEIGHT
                    x_pos[x] = x_pos[y]
    return [(x_pos[o], y_pos[o]) for o in objects]





def get_colours(objects, state):
    colour_dict = {'blue':(0,0,1),
                   'yellow':(1,1,0),
                   'red':(1,0,0),
                   'green':(0, 1, 0)}
    predicates = get_predicates(objects, state)
    available_colours = ['red', 'blue', 'green', 'yellow']
    colours = {o:list(filter(lambda x: x in available_colours, predicates[o].keys()))[0] for o in objects}
    c = [colour_dict[colours[o]] for o in objects]
    return map(data.colour_model, c)

if __name__ == '__main__':
    height = width = 0.2
    #domain, problem = pddl_functions.parse('blocks-domain.pddl', '../FF-v2.3/blocks1.pddl')
    #result = ff.ff('blocks-domain.pddl', '../FF-v2.3/blocks1.pddl')
    #actions = ff.get_actions(result)

    #domain_actions = pddl_functions.create_action_dict(domain)

    #state = problem.initialstate
    #for action, action_arguments in actions:

    #    state = pddl_functions.apply_action(action_arguments, domain_actions[action], state)

    #colours = 100*np.random.rand(len(pddl_functions.get_objects(problem)))
    #posns = generate_start_position(problem)
    #plot_blocks(posns, [(0,0,0)]*len(pddl_functions.get_objects(problem)))

    #problem2 = copy.deepcopy(problem)
    #problem2.initialstate = state
    #positions = place_objects(pddl_functions.get_objects(problem2), problem2.initialstate, posns)
    #plot_blocks(positions, colours)
    #positions1 = place_objects(pddl_functions.get_objects(problem), problem.initialstate, posns)
    #plot_blocks(positions1, colours)


    domain, problem = pddl_functions.parse('blocks-domain.pddl', 'blocks_problem_colour1.pddl')
    objects = pddl_functions.get_objects(problem)
    state = problem.initialstate
    posns = generate_start_position(problem)
    colours = get_colours(objects, state)
    positions = place_objects(objects, state, posns)
    plot_blocks(positions, colours)
