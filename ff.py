import subprocess

ff_location = 'ff/ff' #'/afs/inf.ed.ac.uk/user/s12/s1202144/Desktop/phd/FF-v2.3/ff'

class NoPlanError(Exception):
    pass

class FailedParseError(Exception):
    pass

class Solved(Exception):
    pass

def ff(domain, problem):
    out = subprocess.run(
    [ff_location,
    '-o', domain,
    '-f', problem],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE)
    output = out.stdout.decode()
    if "problem proven unsolvable" in out.stdout.decode():
        print(out.stdout.decode())
        raise NoPlanError('No plan could be found')

    if "goal can be simplified to TRUE" in output:
        raise Solved('The state satifies the goal')

    if out.stderr:
        raise FailedParseError('Could not parse domain or problem file' + out.stderr.decode())
    return out.stdout.decode()


def get_actions(ff_result):
    lines = list(filter(lambda x: len(x) > 0, ff_result.split('\n')))
    for i, line in enumerate(lines):
        if "ff: found legal plan as follows" in line:
            out = lines[i+1:]
            break
        if (i+1) == len(lines):
            raise NoPlanError('No plan could be found')
    actions = []
    for line in out:
        line = line.strip('step').strip()
        try:
            nr, action = line.split(':')
            nr = int(nr)
            actions.append(get_action(action.strip()))
        except ValueError:
            break
    return actions

def get_action(action_string):
    action_parts = action_string.lower().split()
    action = action_parts[0]
    arguments = action_parts[1:]
    return action, arguments

def run(domain, problem):
    result = ff(domain, problem)
    return get_actions(result)

if __name__ == '__main__':
    out = ff('../FF-v2.3/blocks.pddl', '../FF-v2.3/blocks1.pddl')
    actions = get_actions(out)
    print(actions)
