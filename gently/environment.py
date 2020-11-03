from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np

from correctingagent.util import util


def load_data():
    config = util.get_config()
    data_path = Path(config["data_location"]) / "dsprites" / "dsprites_filtered.npz"
    data = np.load(data_path)

    repetitions = 1
    groups = {}
    groups[0] = list(range(3))  # RGB??
    groups[1] = list(range(3))  # ["square", "elipse", "hearth"]
    groups[2] = list(range(3))

    class_sizes = [len(x) for key, x in groups.items()]

    label_mapping = {0: ["blue", "green", "red"], 1: ["square", "circle", "heart"], 2: ["mint", "blue", "yellow"]}

    # BRG format
    # colormaps reference - https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    objects_cmap = plt.get_cmap("jet")
    rgba = np.linspace(0, 1, len(groups[0]))
    obj_colors = [list(objects_cmap(rgba[idx])[:3]) for idx in groups[0]]

    bg_cmap = plt.get_cmap("Pastel2")
    bg_colors = [list(bg_cmap(idx / len(groups[2]))[:3]) for idx in groups[2]]

    #         indecies = []
    #         for i, c in enumerate(data['latents_classes']):
    #             if (c[1] in latent_spec['shape'] and
    #                 c[2] in latent_spec['scale'] and
    #                 c[3] in latent_spec['orientation'] and
    #                 c[4] in latent_spec['x'] and
    #                 c[5] in latent_spec['y']):
    #                 indecies.append(i)
    #         imgs = np.take(data['imgs'], indecies)
    #         latent_classes = np.take(data['latent_classes'], indecies)

    imgs = data['imgs']
    latent_classes = data['latent_classes']

    filtered_images = []
    filtered_labels = []
    filtered_labels_orig = []
    contexts = []

    for img, labels in zip(imgs, latent_classes[..., [0, 1]]):
        for obj_color_idx, obj_color in enumerate(obj_colors):
            for bg_color_idx, bg_color in enumerate(bg_colors):
                for _ in range(repetitions):

                    obj_color_aug = obj_color.copy()
                    # obj_color_aug += np.random.uniform(low=-0.1, high=0.1, size=3)
                    # obj_color_aug = np.clip(obj_color_aug, 0., 1.)

                    bg_color_aug = bg_color.copy()
                    # bg_color_aug += np.random.uniform(low=-0.01, high=0.01, size=3)
                    # bg_color_aug = np.clip(bg_color_aug, 0., 1.)

                    # add object color
                    colored_img = np.tile(img[..., None], (1, 1, 3)) * obj_color_aug
                    labels[0] = obj_color_idx

                    # add background color
                    colored_img[(colored_img == [0, 0, 0]).all(axis=2)] = bg_color_aug

                    l = [0, 0, 0]
                    labels_ext = np.concatenate((labels, np.array([bg_color_idx])))

                    for idx in range(len(labels_ext)):
                        l[idx] = label_mapping[idx][labels_ext[idx]]

                    c = Context(l, colored_img)

                    contexts.append(c)
    return contexts


class Context(object):

    def __init__(self, labels, data):
        self.labels = labels
        self.features = np.array(data)

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value


class ConceptGroup(object):

    def __init__(self, labels, is_context=True):
        if isinstance(labels, ConceptGroup):
            self.labels = ConceptGroup.labels
        else:
            self.labels = labels
        self.is_context = is_context


    def evaluate(self, other):
        for a in self.labels:
            if all([c in other.labels for c in a]):
                return True
        return False

    def __eq__(self, other):
        if isinstance(other, ConceptGroup):
            return self.labels == other.labels
        else:
            return self.labels == other

    def __repr__(self):
        return " \\/ ".join([" /\\ ".join(l) for l in self.labels])


class Rule(object):

    def __init__(self, left_side, right_side):
        self.left_side = ConceptGroup(left_side, is_context=True)
        self.right_side = ConceptGroup(right_side, is_context=False)

    def evaluate(self, context):
        return self.left_side.evaluate(context)

    def __repr__(self):
        return f"{self.left_side} -> {self.right_side}"


class RuleGroup(object):

    def __init__(self, rules):
        self.rules = rules

    def get_behaviour_concepts(self, context):
        for rule in self.rules:
            if rule.evaluate(context):
                yield rule.right_side

    def add_rule(self, rule):
        self.rules.append(rule)
