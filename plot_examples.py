# Copyright 2019 The nn_inconsistency Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import numpy as np
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mc_training import CheckingNN
import mc_training
from TrainingSetup import TrainingSetup

def plot_loss():
    n_hidden = 16
    n_parallel = 1
    lr = 1e-3
    n_steps = 201
    n_epochs_per_step = 100

    np.random.seed(4)
    x, y = mc_training.get_standard_dataset()
    #y += 0.5
    (a, b, c, w) = CheckingNN.createRandomWeights(n_parallel, n_hidden)
    x_weights = 1./len(x) * np.ones(len(x))
    train_setups = [TrainingSetup(x, x_weights, y) for i in range(n_parallel)]
    lrs = np.array([lr for i in range(n_parallel)])
    net = CheckingNN(initial_weights=(a, b, c, w), train_setups=train_setups, lrs=lrs)

    vars = net.create_training_vars()
    losses = []
    y_eval = []
    x_eval = []


    for step in range(n_steps):
        pred_diff = net.predict(x, 0) - y
        loss = np.dot(pred_diff, pred_diff) / (2 * len(x))
        losses.append(loss-1.0)

        x_eval_current = np.sort(np.hstack([[-3.0, 3.0], -net.b[0, 0, :] / net.a[0, 0, :]]))
        x_eval.append(x_eval_current)
        y_eval.append(net.predict(x_eval_current, 0))
        for step_epoch in range(n_epochs_per_step):
            net.train_one_epoch(vars)

    plt.figure('Loss', figsize=(3, 2))
    plt.semilogy(np.arange(n_steps) * n_epochs_per_step, losses, 'k')
    plt.xlabel(r'Epoch $k$')
    plt.ylabel(r"$L_D(W_k) - \inf_{k'} L_D(W_{k'})$")
    utils.ensureDir('./plots/')
    plt.tight_layout()
    plt.savefig('./plots/loss.pdf')

    plt.figure('NN training', figsize=(6, 4))
    linewidth = 0.5
    plt.plot(x, y, 'k.')
    plt.plot(x_eval[0], y_eval[0], 'k--', linewidth=linewidth, label='Initial')
    plt.plot(x_eval[10], y_eval[10], '#666666', linewidth=linewidth, label='1000 Epochs')
    plt.plot(x_eval[n_steps-1], y_eval[n_steps-1], 'k', linewidth=linewidth, label='20000 Epochs')
    plt.scatter(x_eval[n_steps-1][1:-1], y_eval[n_steps-1][1:-1], s=1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/training.pdf')

    plt.figure('Kinks', figsize=(3, 2))
    kink_movement = np.array([x_eval[i][1:-1] for i in range(len(x_eval))])
    for i in range(kink_movement.shape[1]):
        plt.plot(np.arange(kink_movement.shape[0]) * n_epochs_per_step, kink_movement[:, i], 'k', linewidth=linewidth)
    plt.xlabel(r'Epoch $k$')
    plt.ylabel(r'$-b_{i, k}/a_{i, k}$')
    plt.tight_layout()
    plt.savefig('./plots/kink_movement.pdf')


def plot_main():
    plot_loss()


if __name__ == '__main__':
    #matplotlib.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.rc('text', usetex=True)

    plot_main()
