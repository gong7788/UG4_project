


def get_top(relations):
    stuff = list(filter(lambda x: 'in-tower' in x[1], relations.items()))
    if len(stuff) == 1:
        return stuff[0][0]
    stuff = dict(stuff)
    bottom = 't0'
    while True:
        on = stuff.pop(bottom)['on']
        bottom = on.args.args[0].arg_name
        if len(stuff) == 1:
            return bottom


def put_on_top(obj, w):
    obs = w.sense()
    top = get_top(obs.relations)
    return 'put', [obj, top]


def select_action(action_id, w):
    obs = w.sense()
    objs = obs.objects
    obj = objs[action_id]
    return put_on_top(obj, obs)



if __name__ == '__main__':
    import os
    import world
    problem_dir = 'onerule'
    problems = os.listdir(problem_dir)
    w = world.PDDLWorld('blocks-domain.pddl', '{}/{}'.format(problem_dir, problems[0]))

    # put b0 then b2 on top of the tower
    w.update(*put_on_top('b0', w))
    w.update(*put_on_top('b2', w))
    obs = w.sense()
