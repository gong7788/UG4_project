from enum import Enum

from gently.concepts import Concept, Behaviour
from gently.environment import RuleGroup


class ResponseType(Enum):
    new_command = 1
    correction = 2
    assent = 3

def parse_command(sentence):
    sentence = sentence.replace("to", "").strip()
    name, concepts = sentence.split("do it")
    name = name.strip()

    concepts = concepts.split()
    concepts = [Concept.get_base_concept(concept.strip(', ')) for concept in
                concepts if concept.strip(", ") != 'and']



    return Behaviour.from_list(name, concepts)


def parse_negative(sentence):
    sentence = sentence.replace('no,', "").strip()
    concepts = [c.strip(', ') for c in sentence.split() if c.strip(', ') != 'and']
    return concepts

def parse_sentence(sentence):
    if "to" in sentence:
        return parse_command(sentence), ResponseType.new_command
    elif "no" in sentence:
        return parse_negative(sentence), ResponseType.correction
    elif "yes" in sentence:
        return None, ResponseType.assent

#
# class GentlyAgent(object):
#
#     def __init__(self):
#         self.behaviours = {}
#         self.behaviour_under_discussion = None
#
#     def interpret_command(self, action, sentence):
#         data, response_type = parse_sentence(sentence)
#
#         if response_type == ResponseType.new_command:
#             behaviour = data
#             self.behaviours[behaviour.name] = behaviour
#             self.behaviour_under_discussion = behaviour
#
#         if response_type == ResponseType.correction:
#             concepts = data
#             for concept in concepts:
#                 self.behaviour_under_discussion.update_concept(action, concept, positive=False)
#         elif response_type == ResponseType.assent:
#             self.behaviour_under_discussion.update_concept(action, None, positive=True)
#
#     def generate_behaviour(self):
#         return self.behaviour_under_discussion.generate_behaviour()


class GentlyAgent(object):

    def __init__(self):
        self.rules_group = RuleGroup([])

        self.known_concepts = {}

    def generate_behaviour(self, context):

        concepts = list(self.rules_group.get_behaviour_concepts(context))

        assert (len(concepts) == 1 or len(concepts) == 0)
        # We'll figure out if we need to have more than 1 set of concepts
        # Not currently sure what this would look like

        if len(concepts) == 1:
            concepts = [self.known_concepts[c] if c in self.known_concepts else Concept.get_base_concept(c) for c in
                        concepts[0].labels[0]]
        else:
            concepts = []

        b = Behaviour.from_list("none", concepts)

        return b.generate_behaviour()

    def add_rule(self, rule):
        self.rules_group.add_rule(rule)

    def update(self, feedback, action):
        pass
