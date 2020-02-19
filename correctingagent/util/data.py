#import block_plotting
import numpy as np


def binirise_labels(colour, labels):
    return [int(x == colour) for x in labels]


colour_dict = {'blue':(0,0,1),
                   'yellow':(1,1,0),
                   'red':(1,0,0),
                   'green':(0, 1, 0)}


def colour_model(colour):
    sample = np.absolute(np.random.randn(3)*0.1)
    out = np.zeros(3)
    for i, dim in enumerate(colour):
        if dim == 0:
            out[i] = min(dim + sample[i], 1)
        else:
            out[i] = max(dim - sample[i], 0)
    return tuple(out)


def generate_data(colour ,n):
    colours = ['blue', 'green', 'red', 'yellow']
    data = []
    for i in range(n):

        data.append(colour_model(colour_dict[colour]))
    return data


def generate_data_set(n):
    colours = ['blue', 'green', 'red', 'yellow']
    data = []
    labels = []
    for i in range(n):
        for c in colours:
            data.append(colour_model(colour_dict[c]))
            labels.append(c)
    return data, labels


def make_data_dict(colours, n):
    return {c:generate_data(c, n) for c in colours}

def create_objects(colours, n):
    data_dict = make_data_dict(colours, n)
    objs = {}
    n = 2
    for j, c in enumerate(['red', 'green', 'blue', 'yellow']):
        for i in range(n):
            objs["o{}".format(j*n + i)] = (c, data_dict[c][i])
    return objs


def check_rule(o1, o2):
    c1, d1 = o1
    c2, d2 = o2

    if c1 == 'red':
         if c2 != 'blue':
            return 'correction'

    return 'no correction'

def correction_to_int(corr_string):
    return int(corr_string == 'correction')

def colour_to_int(corr_string, colour):
    return int(corr_string == colour)

def my_sample(objs, n=10):
    cs = []
    red = []
    blue = []

    for i in range(n):
        o1 = random.sample(objs.keys(), k=1)[0]
        o2 = random.sample(objs.keys(), k=1)[0]
        while o1 == o2:
            o2 = random.sample(objs.keys(), k=1)[0]
        c = correction_to_int(check_rule(objs[o1], objs[o2]))
        f_red = [objs[o1][1]]
        f_blue = [objs[o2][1]]
        cs.append(c)
        red.append(f_red)
        blue.append(f_blue)
    return cs, np.array(red), np.array(blue)

def my_colour_sample(objs, n=10):
    cs = []
    red = []

    for i in range(n):
        o1 = random.sample(objs.keys(), k=1)[0]
        c = objs[o1][0]
        f_red = objs[o1][1]
        cs.append(c)
        red.append(f_red)
    return cs, red

# colours, c_data = my_colour_sample(objs)
# colours = list(map(lambda x: colour_to_int(x, 'blue'), colours))
# colours = np.array(colours)
# c_data = np.array(c_data)
#
# colours
#
# cs, red_data, blue_data = my_sample(objs)
#
# cs
