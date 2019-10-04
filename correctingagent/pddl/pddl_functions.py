import pythonpddl.pddl
import copy
#from pythonpddl.pddl import Predicate, TypedArg, TypedArgList, Formula
import pythonpddl
from skimage.color import rgb2hsv

from correctingagent.util.colour_dict import colour_names, colour_dict
from correctingagent.world.colours import get_colour


class InvalidActionError(Exception):
    pass


def filter_tower_locations(objects, get_locations=True):
    return list(filter(lambda x: ('t' in x) == get_locations, objects))


def make_variable_list(variables):
    """Creates a TypedArgList from a list of strings

    variables: list of strings: ["o1", "o2", "o3"]
    returns TypedArgList: pythonpddl representation of a list of arguments
    """
    var_list = [pythonpddl.pddl.TypedArg(arg) for arg in variables]
    return pythonpddl.pddl.TypedArgList(var_list)


def update_args(args, action):
    """Helper function that fills in arguments in a function

    takes args as list of strings: ["b1", "b2"]
    An action put(?ob, ?underob)
    with precondition (clear ?ob) (clear ?underob)
    would be updated with put(b1, b2)
    (clear b1) (clear b2)

    returns the updated action
    """
    action = copy.deepcopy(action)
    for i, arg in enumerate(args):
        action.parameters.args[i] = pythonpddl.pddl.TypedArg(arg)
    return action


def equivalent(p1, p2):
    """Checks if two predicates are equal

    (on o1 o2) == (on o1 o2)
    (on o1 o2) != (blue o1)
    (on o1 o2) != (on ?o ?a)
    """
    equal = p1.name == p2.name
    if len(p1.args.args) != len(p2.args.args):
        return False
    else:
        for a1, a2 in zip(p1.args.args, p2.args.args):
            equal = equal and a1.arg_name == a2.arg_name
        return equal


def predicate_holds(predicate, state):
    """Check if a predicate holds in a particular state"""
    for f in state:
        ps = f.get_predicates(True)
        for p in ps:
            if equivalent(predicate, p):
                return True
    return False


def update_predicate(predicate, arg_dict):
    """Replaces variables inside a predicate with an argument from arg_dict"""
    new_pred = copy.deepcopy(predicate)
    for i, arg in enumerate(predicate.args.args):
        new_pred.args.args[i].arg_name = arg_dict[arg.arg_name]
    return new_pred


def apply_action(arguments, action, state):
    """update a state by applying action with arguments to the state """
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
        formula = pythonpddl.pddl.Formula([update_predicate(eff, pred_dict)])
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
    """wrapper for the pythonpddl parse function

    creates a domain and problem object
     """
    return pythonpddl.pddl.parseDomainAndProblem(domain, problem)


def create_action_dict(domain):
    """Creates a dictionary of available actions from a domain

    returns {action.name: action}
    """
    return {action.name: action for action in domain.actions}


def get_predicates(object_, state):
    predicates = []
    for formula in state:
        if isinstance(formula, pythonpddl.pddl.FExpression):
            continue
        predicate = formula.get_predicates(1)[0]
        if object_ in [arg.arg_name for arg in predicate.args.args]:
            predicates.append(predicate)
    return predicates


def create_formula(predicate, variables, op=None):
    """Builds a pythonpddl Formula for one predicate

    predicate - string name of predicate: "pred"
    variables - list of strings: ["a1", "a2"]
    op - str or None for example "not"
    returns: pythonpddl of (pred a1 a2) or (not (pred a1 a2)) etc
    """
    variables = make_variable_list(variables)
    predicate = pythonpddl.pddl.Predicate(predicate, variables)
    return pythonpddl.pddl.Formula([predicate], op=op)


def get_objects(problem):
    return [arg.arg_name for arg in problem.objects.args]


def obscure_state(state, obscured_predicates=['green', 'blue', 'red', 'yellow']):
    """removes a set of preicates from a state

    used when observing a domain state but for example colours are unknown
    """
    return [p for p in state if p.get_predicates(1)[0].name not in obscured_predicates]


def get_clear_objs(obs):
    rels = obs.relations
    out = []
    for o, val in rels.items():
        if 'clear' in val:
            out.append(o)
    return out


class Predicate(object):

    def __init__(self, name, args, op=None):
        self.name = name
        self.args = args
        self.valency = not(op == 'not')
        self.op = op

    @staticmethod
    def from_formula(formula):
        if formula.is_condition:
            return Conditional.from_formula(formula)
        pred = formula.subformulas[0]
        return Predicate(pred.name, [arg.arg_name for arg in pred.args.args], op=formula.op)

    def to_formula(self):
        args = make_variable_list(self.args)
        predicate = pythonpddl.pddl.Predicate(self.name, args)
        return pythonpddl.pddl.Formula([predicate], op=self.op)

    def negate(self):
        op = 'not' if self.valency is True else None
        return Predicate(self.name, self.args, op)

    def to_pddl(self):
        return self.to_formula().asPDDL()

    def __str__(self):
        return self.to_pddl()

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.to_pddl() == other.to_pddl()

    def __ne__(self, other):
        return not(self.__eq__(other))


class Action(object):

    def __init__(self, name, preconditions, effects, params):
        self.preconditions = preconditions
        self.effects = effects
        self.name = name
        self.params = params

    @staticmethod
    def from_pddl(pddl_action):
        preconditions = [Predicate.from_formula(pred)
                         for pred in pddl_action.pre.subformulas]
        effects = [Predicate.from_formula(pred)
                   for pred in pddl_action.eff]
        args = [arg.arg_name for arg in pddl_action.parameters.args]
        return Action(pddl_action.name, preconditions, effects, args)

    def to_formula(self):
        params = make_variable_list(self.params)
        precond_formulas = [pre.to_formula() for pre in self.preconditions]
        precond = pythonpddl.pddl.Formula(precond_formulas, op='and')
        effects = [effect.to_formula() for effect in self.effects]
        return pythonpddl.pddl.Action(self.name, params, precond, effects)

    def to_pddl(self):
        return self.to_formula().asPDDL()

    def __str__(self):
        return self.to_pddl()

    def __repr__(self):
        return self.__str__()

    def update_with_params(self, params, items):
        updated_predicates = []
        param_dict = {my_param: their_param for my_param, their_param
                      in zip(self.params, params)}
        for predicate in items:
            predicate = copy.deepcopy(predicate)
            if isinstance(predicate, Conditional):
                predicate.condition.args = [param_dict[arg] for arg in predicate.condition.args]
                if isinstance(predicate.effect, Increase):
                    predicate.effect.tower = param_dict[predicate.effect.tower]
                else:
                    predicate.effect.args = [param_dict[arg] for arg in predicate.effect.args]
            else:
                predicate.args = [param_dict[arg] for arg in predicate.args]
            updated_predicates.append(predicate)
        return updated_predicates

    def preconditions_hold(self, state, params):
        preconditions = self.update_with_params(params, self.preconditions)
        return all([state.predicate_holds(predicate) for predicate
                    in preconditions])

    def apply_action(self, state, params):
        if not self.preconditions_hold(state, params):
            raise InvalidActionError('Preconditions of action do not hold in current state')

        effects = self.update_with_params(params, self.effects)
        for effect in effects:
            if isinstance(effect, Conditional):
                if state.predicate_holds(effect.condition):
                    state.apply_effect(effect.effect)
            else:
                state.apply_effect(effect)


class PDDLState(object):
    def __init__(self, predicates, fexpressions):
        self.predicates = predicates
        self.fexpressions = fexpressions

    def to_formula(self):
        return [predicate.to_formula() for predicate in self.predicates + self.fexpressions]

    @staticmethod
    def from_initialstate(state):
        predicates = []
        fexpressions = []
        for predicate in state:
            if isinstance(predicate, pythonpddl.pddl.Formula):
                predicates.append(Predicate.from_formula(predicate))
            elif isinstance(predicate, pythonpddl.pddl.FExpression):
                fexpressions.append(ColourCount.from_fexpression(predicate))
        return PDDLState(predicates, fexpressions)

    def get_predicates(self, arg):
        return [pred for pred in self.predicates if arg in pred.args]

    def get_clear_objs(self):
        return [pred.args[0] for pred in self.predicates if pred.name == 'clear']

    def predicate_holds(self, predicate):
        return predicate in self.predicates

    def apply_effect(self, predicate):
        if isinstance(predicate, Increase):
            for fluent in self.fexpressions:
                if predicate.compare_colour_count(fluent) is True:
                    fluent.increment(predicate.number)
        else:
            if predicate.valency is False:
                predicate = predicate.negate()
                self.predicates.remove(predicate)
            else:
                self.predicates.append(predicate)

    def get_colour_count(self, colour, tower):
        for fluent in self.fexpressions:
            if fluent.colour == colour and fluent.tower == tower:
                return fluent.number

    def get_colours(self, objects, use_hsv=False):
        objects = [object for object in objects if 'g' not in object and 't' not in object]

        colours = {}
        for obj in objects:
            colours[obj] = [pred.name for pred in self.get_predicates(obj) if pred.name in colour_names]

        for obj, colour_list in colours.items():
            if len(colour_list) > 1:
                for c_i in colour_list:
                    if c_i not in colour_dict.keys():
                        colours[obj] = c_i
            else:
                colours[obj] = colour_list[0]
        if use_hsv:
            return {o:rgb2hsv(get_colour(colours[o])) for o in objects}
        else:
            return {o:get_colour(colours[o]) for o in objects}

class ColourCount(object):

    def __init__(self, colour, tower, number):
        self.colour = colour
        self.tower = tower
        self.number = number

    @staticmethod
    def from_fexpression(fexpression):
        head, number = fexpression.subexps
        colour_count = head.name
        arg = head.args.args[0].arg_name
        colour = colour_count.split('-')[0]
        return ColourCount(colour, arg, number.val)

    def to_fexpression(self):
        args = make_variable_list([self.tower])
        colour_count_head = pythonpddl.pddl.FHead(f'{self.colour}-count', args)
        count_value = pythonpddl.pddl.ConstantNumber(self.number)
        subexpressions = [colour_count_head, count_value]
        return pythonpddl.pddl.FExpression('=', subexpressions)

    def to_pddl(self):
        return self.to_fexpression().asPDDL()

    def increment(self, n=1):
        self.number += n

    def decrement(self, n=1):
        self.number -= n

    def __str__(self):
        return self.to_pddl()

    def __repr__(self):
        return self.__str__()


class Increase(object):
    def __init__(self, colour, tower, number):
        self.colour = colour
        self.number = number
        self.tower = tower

    @staticmethod
    def from_formula(formula):
        assert (formula.op == 'increase')
        head, number = formula.subformulas
        colour_count = head.name
        arg = head.args.args[0].arg_name
        colour = colour_count.split('-')[0]
        return Increase(colour, arg, number.val)

    def to_formula(self):
        args = make_variable_list([self.tower])
        colour_count_head = pythonpddl.pddl.FHead(f'{self.colour}-count', args)
        count_value = pythonpddl.pddl.ConstantNumber(self.number)
        subexpressions = [colour_count_head, count_value]
        return pythonpddl.pddl.Formula(subexpressions, op='increase')

    def compare_colour_count(self, other):
        return self.colour == other.colour and self.tower == other.tower

    def to_pddl(self):
        return self.to_formula().asPDDL()

    def __str__(self):
        return self.to_pddl()

    def __repr__(self):
        return self.__str__()


class Conditional(object):
    def __init__(self, condition, effect):
        self.condition = condition
        self.effect = effect

    @staticmethod
    def from_formula(formula):
        assert (formula.is_condition)
        condition = Predicate.from_formula(formula.condition)
        if formula.op == 'increase':
            effect = Increase.from_formula(formula)
        else:
            effect = Predicate.from_formula(formula)
        return Conditional(condition, effect)

    def to_formula(self):
        formula = self.effect.to_formula()
        formula.is_condition = True
        formula.condition = self.condition.to_formula()
        return formula

    def to_pddl(self):
        return self.to_formula().asPDDL()

    def __str__(self):
        return self.to_pddl()

    def __repr__(self):
        return self.__str__()

if __name__ == "__main__":
    domain, problem = pythonpddl.pddl.parseDomainAndProblem(
    '../FF-v2.3/wumpus-a.domain', '../FF-v2.3/wumpus-a-1.domain')
    print("Before action:")
    for predicate in problem.initialstate:
        print(predicate.asPDDL())
    a1 = domain.actions[0]
    new_state = apply_action(('agent', 's-1-1', 's-2-1'), a1, problem.initialstate)
    print("After action:")
    for predicate in new_state:
        print(predicate.asPDDL())
