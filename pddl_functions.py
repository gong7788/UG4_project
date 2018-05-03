import pythonpddl2.pddl
import copy


class InvalidActionError(Exception):
    pass


def update_args(args, action):
    action = copy.deepcopy(action)
    for i, arg in enumerate(args):
        action.parameters.args[i] = pythonpddl2.pddl.TypedArg(arg)
    return action


def equivalent(p1, p2):
    equal = p1.name == p2.name
    if len(p1.args.args) != len(p2.args.args):
        return False
    else:
        for a1, a2 in zip(p1.args.args, p2.args.args):
            equal = equal and a1.arg_name == a2.arg_name
        return equal


def predicate_holds(predicate, state):
    for f in state:
        ps = f.get_predicates(True)
        for p in ps:
            if equivalent(predicate, p):
                return True
    return False


def update_predicate(predicate, arg_dict):
    new_pred = copy.deepcopy(predicate)
    for i, arg in enumerate(predicate.args.args):
        new_pred.args.args[i].arg_name = arg_dict[arg.arg_name]
    return new_pred

def apply_action(arguments, action, state):
    action_args = action.parameters.args
    pred_dict = {param.arg_name:arg for param, arg in zip(action_args, arguments)}
    #specific_action = update_args(arguments, action)
    pos_pre = [predicate_holds(update_predicate(p, pred_dict), state) for p in action.get_pre(True)]
    neg_pre = [not(predicate_holds(update_predicate(p.get_predicates(1)[0], pred_dict), state)) for p in action.get_pre(False)]
    if not(all(pos_pre) and all(neg_pre)):
        raise InvalidActionError('preconditions do not hold for action {} and parameters {}'.format(action.name, ' '.join(arguments)) )

    new_state = copy.deepcopy(state)
    for eff in action.get_eff(1):
        try:
            if eff.is_condition:
                if not(predicate_holds(update_predicate(eff.condition, pred_dict), state)):
                    continue
        except AttributeError:
            pass
        formula = pythonpddl2.pddl.Formula([update_predicate(eff, pred_dict)])
        new_state.append(formula)

    for eff in action.get_eff(0):
        try:
            if eff.is_condition:
                if not(predicate_holds(update_predicate(eff.condition, pred_dict), state)):
                    continue
        except AttributeError:
            pass
        new_state = list(filter(lambda x: not(
            equivalent(
                update_predicate(eff, pred_dict), x.get_predicates(1)[0])),
              new_state))



    return new_state

def parse(domain, problem):
    return pythonpddl2.pddl.parseDomainAndProblem(domain, problem)


def create_action_dict(domain):
    return {action.name:action for action in domain.actions}

def get_predicates(object_, state):
    predicates = []
    for formula in state:
        predicate = formula.get_predicates(1)[0]
        if object_ in [arg.arg_name for arg in predicate.args.args]:
            predicates.append(predicate)
    return predicates


def get_objects(problem):
    return [arg.arg_name for arg in problem.objects.args]


if __name__ == "__main__":
    domain, problem = pythonpddl2.pddl.parseDomainAndProblem(
    '../FF-v2.3/wumpus-a.pddl', '../FF-v2.3/wumpus-a-1.pddl')
    print("Before action:")
    for predicate in problem.initialstate:
        print(predicate.asPDDL())
    a1 = domain.actions[0]
    new_state = apply_action(('agent', 's-1-1', 's-2-1'), a1, problem.initialstate)
    print("After action:")
    for predicate in new_state:
        print(predicate.asPDDL())
