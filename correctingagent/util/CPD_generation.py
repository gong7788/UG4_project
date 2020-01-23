
import numpy as np
import re
from functools import reduce




def get_violation_type(violation):
    rule = re.sub(r"V_[0-9]+\(", '', violation)[:-1]
    return get_rule_type(rule)


def get_rule_type(rule):

    if rule[:3] != 'all':
        raise NotImplemented('Not implemented non put red on blue rule')
    elif "count" in rule:
        _, right = rule.split('->')
        colour_count, count = rule.split('>=')
        count = int(count.strip())
        colour_name = colour_count.split('-')[0].strip()
        #f"all t. tower(t) -> {colour_name}-count >= {count}"
       #return ColourCountRule(colour_name, count)
    else:
        red, blue, on = split_rule(rule)
        red_colour, o1 = get_predicate(red)
        blue_colour, o2 = get_predicate(blue)
        x, y = get_predicate_args(on)

        if o1[0] == x:
            pass
            # return red_colour, blue_colour, rule_type
        #    return RedOnBlueRule(red_colour, blue_colour, rule_type=1)
        elif o2[0] == x:
            pass
         #   return RedOnBlueRule(blue_colour, red_colour, rule_type=2)
        else:
            raise ValueError('something went wrong')


def split_rule(rule):
    bits = rule.split('.(')
    red = bits[1].split('->')[0].strip()
    bits2 = bits[1].split(' (')
    blue, on = bits2[1].split('&')
    blue = blue.strip()
    on = on.replace('))', '').strip()
    return [red, blue, on]


def get_predicate_args(predicate):
    args = predicate.split('(')[1].replace(')', '')
    return [arg.strip() for arg in args.split(',')]


def get_predicate(predicate):
    pred = predicate.split('(')[0]
    args = predicate.split('(')[1].replace(')', '')
    args = [arg.strip() for arg in args.split(',')]
    return pred, args


def generate_neg_table_cpd():
    cpd_line_corr0 = []
    cpd_line_corr1 = []
    for r1 in range(2):
        for blueo1 in range(2):
            for redo3 in range(2):
                result = r1 * int(blueo1 and redo3)
                cpd_line_corr1.append(result)
                cpd_line_corr0.append(1-result)

    return [cpd_line_corr0, cpd_line_corr1]


def binary_flip(n):
    out = []
    def helper(n, l):
        if n == 0:
            out.append(l)
        else:
            l1 = l.copy()
            l1.append(0)
            l2 = l.copy()
            l2.append(1)
            helper(n-1, l1)
            helper(n-1, l2)

    helper(n-1, [0])
    helper(n-1, [1])
    return out


def variable_or_CPD(n):
    if n == 0:
        return []
    flippings = binary_flip(n)
    CPD = np.zeros((2, len(flippings)), dtype=np.int32)
    for i, l in enumerate(flippings):
        correction = int(reduce(lambda x, y: x or y, l))
        # correction should happen (C=1)
        CPD[1][i] = correction
        # correction should not happen (C=0)
        CPD[0][i] = 1-correction
    return CPD


def generate_flips(variables, cardinalities=None, evidence={}, output=[]):
    variables = list(variables)
    if cardinalities is None:
        cardinalities = [2]*len(variables)
    if len(variables) == 0:
        output.append(evidence)
    else:
        for i in range(cardinalities[0]):
            new_evidence = evidence.copy()
            new_evidence[variables[0]] = i
            generate_flips(variables[1:], cardinalities=cardinalities[1:], evidence=new_evidence, output=output)


def equals_CPD(seta, setb, carda=None, cardb=None):
    if len(seta) != len(setb):
        raise TypeError('Lengths do not match')
    if carda is None:
        carda = [2]*len(seta)
    if cardb is None:
        cardb = [2]*len(setb)
    results = []
    def helper(settingsa, cardsa, settingsb, cardsb, results):
        if len(cardsa) == 0:
            results.append(int(all([a == b for a,b in zip(settingsa, settingsb)])))
        else:
            for i in range(cardsa[0]):
                for j in range(cardsb[0]):
                    sa = settingsa.copy()
                    sb = settingsb.copy()
                    helper(sa+[i], cardsa[1:], sb+[j], cardsb[1:], results)
    helper([], carda, [], cardb, results)

    return results


# def equals_CPD(seta, setb):
#     flips = binary_flip(len(seta) + len(setb))
#
#     results = []
#     for flip in flips:
#         r = all([flip[i] == flip[len(seta)] + i for i in range(len(seta))])
#         results.append(int(r))
#     return results