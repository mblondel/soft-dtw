import copy

import numpy as np

from chainer import training
from chainer import iterators, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer.datasets import tuple_dataset

from sdtw.dataset import load_ucr
from sdtw.chainer_func import SoftDTWLoss

import matplotlib.pylab as plt
import matplotlib.font_manager as fm
plt.style.use('ggplot')
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15


def split_time_series(X_tr, X_te, proportion=0.6):
    len_ts = X_tr.shape[1]
    len_input=int(round(len_ts * proportion))
    len_output=len_ts - len_input

    return np.float32(X_tr[:, :len_input, 0]), \
           np.float32(X_tr[:, len_input:, 0]), \
           np.float32(X_te[:, :len_input, 0]), \
           np.float32(X_te[:, len_input:, 0])


class MLP(Chain):

    def __init__(self, len_input, len_output, activation="tanh", n_units=50):
        self.activation = activation

        super(MLP, self).__init__(
            mid = L.Linear(len_input, n_units),
            out=L.Linear(n_units, len_output),
        )

    def __call__(self, x):
        # Given the current observation, predict the rest.
        xx = self.mid(x)
        func = getattr(F, self.activation)
        h = func(xx)
        y = self.out(h)
        return y


class Objective(Chain):

    def __init__(self, predictor, loss="euclidean", gamma=1.0):
        self.loss = loss
        self.gamma = gamma
        super(Objective, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)

        if self.loss == "euclidean":
            return F.mean_squared_error(y, t)

        elif self.loss == "sdtw":
            loss = 0
            for i in range(y.shape[0]):
                y_i = F.reshape(y[i], (-1,1))
                t_i = F.reshape(t[i], (-1,1))
                loss += SoftDTWLoss(self.gamma)(y_i, t_i)
            return loss

        else:
            raise ValueError("Unknown loss")


def train(network, loss, X_tr, Y_tr, X_te, Y_te, n_epochs=30, gamma=1):
    model= Objective(network, loss=loss, gamma=gamma)

    #optimizer = optimizers.SGD()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    train = tuple_dataset.TupleDataset(X_tr, Y_tr)
    test = tuple_dataset.TupleDataset(X_te, Y_te)

    train_iter = iterators.SerialIterator(train, batch_size=1, shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size=1, repeat=False,
                                         shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (n_epochs, 'epoch'))

    trainer.run()


if __name__ == '__main__':
    import os
    import sys

    try:
        dbname = sys.argv[1]
    except IndexError:
        dbname = "ECG200"

    X_tr, _, X_te, _ = load_ucr(dbname)

    proportion = 0.6
    n_units = 10
    n_epochs = 30
    gamma = 1
    warm_start = True

    X_te_ = X_te
    X_tr, Y_tr, X_te, Y_te = split_time_series(X_tr, X_te, proportion)

    len_input = X_tr.shape[1]
    len_output = Y_tr.shape[1]

    networks = [MLP(len_input, len_output, n_units=n_units),]
    losses = ["sdtw",]
    labels = ["Soft-DTW loss",]

    for i in range(len(networks)):
        if warm_start and i >= 1:
            # Warm-start with Euclidean-case solution
            networks[i].mid = copy.deepcopy(networks[0].mid)
            networks[i].out = copy.deepcopy(networks[0].out)

        train(networks[i], losses[i], X_tr, Y_tr, X_te, Y_te,
          n_epochs=n_epochs, gamma=gamma)

    max_vals = []
    min_vals = []

    fig = plt.figure(figsize=(10,6))

    pos = 220

    for i in range(min(X_te.shape[0], 4)):
        pos += 1
        ax = fig.add_subplot(pos)

        inputseq = np.array([X_te[i]])  # Need to wrap as minibatch...

        # Plot predictions.
        for idx, label in enumerate(labels):
            output = networks[idx](inputseq)
            output = np.squeeze(np.array(output.data))

            ax.plot(range(len_input + 1,len_input + len(output) + 1),
                     output,
                     alpha=0.75,
                     lw=3,
                     label=label,
                     zorder=10)

            max_vals.append(output.max())
            min_vals.append(output.min())

        # Plot ground-truth time series.
        ground_truth = X_te_[i]
        max_vals.append(ground_truth.max())
        min_vals.append(ground_truth.min())
        ax.plot(ground_truth,
                 c="k",
                 alpha=0.3,
                 lw=3,
                 label='Ground truth',
                 zorder=5)

        # Plot vertical line.
        ax.plot([len_input, len_input],
                 [np.min(min_vals), np.max(max_vals)], lw=3, ls="--", c="k")

        # Legend.
        prop = fm.FontProperties(size=18)
        ax.legend(loc="best", prop=prop)

    fig.set_tight_layout(True)
    plt.show()
