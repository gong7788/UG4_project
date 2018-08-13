import subprocess
import time
from util import get_config

#ff_location = #'ff/ff' #'/afs/inf.ed.ac.uk/user/s12/s1202144/Desktop/phd/FF-v2.3/ff'
config = get_config()
ff_location = config['ff_location']

class NoPlanError(Exception):
    pass

class FailedParseError(Exception):
    pass

class Solved(Exception):
    pass

class IDontKnowWhatIsGoingOnError(Exception):
    pass

def ff(domain, problem):
    process = subprocess.Popen([ff_location,
    '-o', domain,
    '-f', problem],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE)
    # out = subprocess.run(
    # [ff_location,
    # '-o', domain,
    # '-f', problem],
    # stdout=subprocess.PIPE,
    # stderr=subprocess.PIPE)

    for i in range(1000):
        code = process.poll()
        if code == 0 or code == 1:
            break
        time.sleep(0.001)
    else:
        process.terminate()
        raise IDontKnowWhatIsGoingOnError('The amount of time taken was too long')

    output = process.stdout.read().decode()
    exitCode = process.returncode
    if "problem proven unsolvable" in output or "goal can be simplified to FALSE" in output:
        raise NoPlanError('No plan could be found')

    elif "goal can be simplified to TRUE" in output:
        raise Solved('The state satifies the goal')

    elif "won't get here: non NOT,OR,AND in goal set relevants" in output:
        raise IDontKnowWhatIsGoingOnError('non Not,Or,And in goal set relevants, what ever that means?!')

    elif exitCode:
        raise FailedParseError('Could not parse domain or problem file' + output)
    return output


def get_actions(ff_result):
    lines = list(filter(lambda x: len(x) > 0, ff_result.split('\n')))
    for i, line in enumerate(lines):
        if "ff: found legal plan as follows" in line:
            out = lines[i+1:]
            break
        if (i+1) == len(lines):
            raise NoPlanError('No plan could be found')
    actions = []
    try:
        for line in out:
            line = line.strip('step').strip()
            try:
                nr, action = line.split(':')
                nr = int(nr)
                actions.append(get_action(action.strip()))
            except ValueError:
                break
    except UnboundLocalError:
        print(ff_result)
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
