
# import matplotlib
import matplotlib.pyplot as plt
# import numpy
import numpy as np


class Plotable(object):
    """ Interface for a plottable object """
    def plot(self, ax):
        return NotImplementedError()


# *** Basic Plotables ***

class ConfusionMatrix(Plotable):
    """ Class to embedd a confusion matrix in a TorchBoard """

    def __init__(self, classes, name="Confusion", normalize=False):
        # save name and classes
        self.name = name
        self.classes = classes
        # should confusion matrix be normalized before plotting
        self.normalize = normalize
        # initialize confusion matrix
        self.confusion = np.zeros((len(classes), len(classes)))

    def plot(self, ax):

        # normalize confusion if needed
        confusion = self.confusion.astype(np.int32) if not self.normalize else self.confusion.astype(np.float32) / (self.confusion.sum(axis=1, keepdims=True) + 1e-10)

        # configure axes
        ax.set(xticks=np.arange(confusion.shape[1]),
            yticks=np.arange(confusion.shape[0]),
            xticklabels=self.classes, yticklabels=self.classes,
            ylabel='True label',
            xlabel='Predicted label')

        # create confusion matrix and colorbar
        im = ax.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if self.normalize else 'd'
        thresh = confusion.max() / 2.
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                ax.text(j, i, format(confusion[i, j], fmt),
                            ha="center", va="center",
                            color="white" if confusion[i, j] > thresh else "black"
                        )

    def __iadd__(self, confusion):
        # update confusion matrix
        self.confusion = confusion
        return self


# *** Statistics ***

class Statistic(Plotable):
    """ Class to manage a single statistic. """

    def __init__(self, name):
        # save name and initialize value-list
        self.name = name
        self.values = []

    def add_value(self, value):
        # add value to statistic
        self.values.append(value)

    def plot(self, ax):
        ax.plot(list(self), label=self.name)

    def __iadd__(self, value):
        # overload inplace add operation to add value to list
        self.add_value(value)
        return self

    def __iter__(self):
        # make statistic iterable
        return iter(self.values)

    def __str__(self):
        return self.name + ": " + str(self.values)

    def __len__(self):
        return len(self.values)

class AverageStatistic(Statistic):
    """ Average Statistic of given Statistic. """

    def __init__(self, parent_stat, name=None, window=100):
        # save reference to statistic
        self.parent_stat = parent_stat
        # save name and window-size
        self.name = name if name is not None else (parent_stat.name + "_avg")
        self.window = window//2

    def add_value(self, value):
        self.parent_stat.add_value(value)
        return self

    def __iter__(self):
        values = list(self.parent_stat)
        # create generator of avg values
        avg_generator = (np.mean(values[max(0, i-self.window):min(len(self.parent_stat), i+self.window)]) for i in range(len(self.parent_stat)))
        return iter(avg_generator)

    def __len__(self):
        return len(self.parent_stat)



# *** Board ***

class TorchBoard(object):
    """ Class to manage multiple Statistics. """

    def __init__(self, *args):
        # statistics to track
        self.statistics = {stat: Statistic(stat) for stat in args}

    def add_stat(self, stat: Statistic):
        assert stat.name not in self.statistics, "Statistic {0} already defined in board!".format(stat.name)
        # add a statistic to board
        self.statistics[stat.name] = stat

    def add_stat_value(self, stat, value):
        assert stat in self.statistics, "Statistic {0} was not recognized!".format(stat)
        # append value to statistic
        self.statistics[stat].append(value)

    def create_fig(self, layout, **kwargs):

        plt.close("all")
        # create figure
        fig = plt.figure(**kwargs)
        # handle single statistic
        if type(layout) in [str]:
            layout = [layout]

        n_rows = len(layout)
        for i, row in enumerate(layout):
            # handle single statistic in row
            if type(row) in [str]:
                row = [row]

            n_cols = len(row)
            for j, col in enumerate(row):
                # create axis
                ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1)
                ax.set_title(' - '.join([s if type(s) is str else s.name for s in col]))
                # handle single statistic in column
                if type(col) in [str]:
                    col = [col]

                for stat in col:
                    # plot all statistics assigned to current axes
                    self.statistics[stat].plot(ax)

                # add legend if there is more than one graph in axes
                if len(col) > 1:
                    ax.legend()

        fig.tight_layout()
        return fig


    def __getattr__(self, name):
        # check if name is statistic
        if name in self.statistics:
            return self.statistics[name]
        raise AttributeError(f'{name} is not a Statistic in current torchboard')

    def __getitem__(self, name):
        # check if name is statistic
        if name in self.statistics:
            return self.statistics[name]
        raise AttributeError(f'{name} is not a Statistic in current torchboard')
    
    def __setitem__(self, name, value):
        # necessary for item assignment
        self.statistics[name] = value
    
    def __str__(self):
        # convert to string
        return '\n'.join(str(stat) for stat in self.statistics.values())
