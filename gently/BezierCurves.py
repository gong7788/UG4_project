import numpy as np
import bezier
import matplotlib.pyplot as plt


def curve_location(i):
    x = i + .5
    return (x, 1 - x)


def map_speed_variable(i):
    assert (i > 0)
    x = int(i * 200)

    return x


def generate_curve(x=[0, 0], y=[1, 1], z=[0.5, 0.5]):
    array = np.array([x, z, y])

    nodes = np.asfortranarray(array.T)
    curve = bezier.Curve(nodes, degree=2)
    return curve


class BehaviourCurve(object):

    def __init__(self, curviness, speed, energy=.5):
        self.curviness = curviness
        self.speed = speed
        self.curve = generate_curve(z=curve_location(curviness))
        self.energy = energy

    def plot(self, set_plot_env=True):
        if set_plot_env:
            set_plot_environment()
        cmap = plt.get_cmap('hot')
        s_vals = np.linspace(0, 1, map_speed_variable(self.speed))
        points = self.curve.evaluate_multi(s_vals)
        plt.plot(*points, '.', c=cmap(self.energy))

    def get_data(self):
        return np.array([self.curviness, self.speed, self.energy])

    def get_value(self, dimension):
        if dimension == "curviness":
            return self.curviness
        elif dimension == "speed":
            return self.speed
        elif dimension == "energy":
            return self.energy


def set_plot_environment(size=5):
    fig = plt.figure(figsize=(size, size))
    return fig

