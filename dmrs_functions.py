from delphin.interfaces import ace
from delphin.mrs.simpledmrs import dump, dumps, serialize
from delphin.mrs import convert
from delphin.mrs.dmrx import loads
from delphin.extra.latex import dmrs_tikz_dependency
from collections import defaultdict

def parse_sentence2(sentence, grammar = 'ace-0.9.27/erg-1214-osx-0.9.27.dat'):#'../ace-0.9.26/erg-1214-x86-64-0.9.26.dat'):
    p = ace.parse(grammar, sentence)
    top_result = p['RESULTS'][0]['MRS']
    dmrx = convert(p['RESULTS'][0]['MRS'], 'simplemrs', 'dmrx')
    dmrxx = loads(dmrx, single=True)
    return dmrxx.to_dict()



def make_links(links):
    d = defaultdict(dict)
    for link in links:
        d[link['from']][link['rargname']] = link['to']
    return d

def get_node(node, nodes):
    for n in nodes:
        if n['nodeid'] == node:
            return n
    else:
        return None

def full_parse(sentence):
    p = parse_sentence2(sentence)
    links = make_links(p['links'])
    nodes = p['nodes']
    return links, nodes

def get_pred(string):
    parts = string.strip('_').split('_')
    try:
        return parts[0]
    except IndexError:
        return ''

def get_pos(string):
    parts = string.strip('_').split('_')
    try:
        return parts[1]
    except IndexError:
        return ''

def find_verb(links, nodes):
    root = links[0][None]
    curr_link = root
    while get_pos(get_node(curr_link, nodes)['predicate']) != 'v':
        curr_link = links[curr_link]['ARG1']
    return get_node(curr_link, nodes)

def find_rel(links, nodes):
    verb = find_verb(links, nodes)
    nodeid = verb['nodeid']
    rel = links[nodeid]['ARG3']
    return get_node(rel, nodes)

def find_open_class(links, nodes, arg='ARG1'):
    rel = find_rel(links, nodes)
    obj = []
    root = links[rel['nodeid']][arg]
    obj.append(get_node(root, nodes))
    for key,value in links.items():
        try:
            link = value['ARG1']
        except KeyError:
            continue
        if root == link:
            node = get_node(key, nodes)
            if get_pos(node['predicate']) != 'p':
                obj.append(get_node(key, nodes))

    return obj

def get_tripple(links, nodes):
    rel = find_rel(links, nodes)
    obj1 = find_open_class(links, nodes, arg='ARG1')
    obj2 = find_open_class(links, nodes, arg='ARG2')
    return rel, obj1, obj2


def sent_to_tripple(sentence):
    links, nodes = full_parse(sentence)
    return get_tripple(links, nodes)

def get_adjectives(obj):
    return [get_pred(node['predicate']) for node in obj if get_pos(node['predicate']) == 'a']


if __name__ == "__main__":
    l, n = full_parse('put red blocks on blue blocks')
    print(l)
