# Copyright 2022 The nn_inconsistency Authors. All Rights Reserved.
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
#matplotlib.use('Agg')
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import matplotlib.pyplot as plt
from mc_training import CheckingNN
import mc_training
from TrainingSetup import TrainingSetup


def expand_1d(x):
    return np.expand_dims(x, axis=1)


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
    plt.plot(x_eval[10], y_eval[10], '#AAAAAA', linewidth=linewidth, label='1000 Epochs')
    plt.plot(x_eval[n_steps-1], y_eval[n_steps-1], 'k', linewidth=linewidth, label='20000 Epochs')
    plt.scatter(x_eval[n_steps-1][1:-1], y_eval[n_steps-1][1:-1], s=1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.tight_layout()
    #plt.plot(x_eval[999], y_eval[999], '#888888', linewidth=linewidth)
    plt.savefig('./plots/training.pdf')

    plt.figure('Kinks', figsize=(3, 2))
    kink_movement = np.array([x_eval[i][1:-1] for i in range(len(x_eval))])
    for i in range(kink_movement.shape[1]):
        plt.plot(np.arange(kink_movement.shape[0]) * n_epochs_per_step, kink_movement[:, i], 'k', linewidth=linewidth)
    plt.xlabel(r'Epoch $k$')
    plt.ylabel(r'$-b_{i, k}/a_{i, k}$')
    plt.tight_layout()
    plt.savefig('./plots/kink_movement.pdf')


def target_function(x):
    # \int_1^4 -sin(pi * x) dx = 1/pi [cos(pi*x)]_1^4 = 2/pi
    return -np.sin(np.pi * x) - 2/(3*np.pi)


def get_dataset(f, num_samples):
    x = np.random.uniform(1.0, 4.0, num_samples)
    y_noise = np.random.randn(num_samples)
    return x, f(x) + y_noise


def plot_convergence_samples(num_samples=16, estimator=None, desc=''):
    plt.figure('Estimates')
    plt.xlim(1, 4)
    plt.ylim(-5, 5)
    np.random.seed(12345)
    x_domain = np.linspace(1, 4, 200)
    f = target_function
    if num_samples is not None:
        x, y = get_dataset(f, num_samples)
        plt.plot(x, y, 'x', color='#666666')
        if estimator is not None:
            estimator.fit(expand_1d(x), expand_1d(y))
            larger_x_domain = np.linspace(1, 4, 1000)
            plt.plot(larger_x_domain, estimator.predict(expand_1d(larger_x_domain)).flatten(), 'b')
    plt.plot(x_domain, f(x_domain), '#44FF44')
    plt.savefig('./plots/convergence_samples_{}_{}.pdf'.format(desc, num_samples))
    plt.close()
    pass


def plot_trained_network(x, y, plot_filename, random_seed=0, n_hidden=16, n_epochs=100000, lr=1e-3, use_adam=False,
                         model_filename=None, small_domain=False):
    import show_training
    np.random.seed(random_seed)
    n_parallel = 1
    plotter = show_training.SimpleNNPlotter(x, y, small_plot=True, show_plot=False, thick_lines=True,
                                            small_domain=small_domain)

    (a, b, c, w) = CheckingNN.createRandomWeights(n_parallel, n_hidden)
    x_weights = 1. / len(x) * np.ones(len(x))
    train_setups = [TrainingSetup(x, x_weights, y) for i in range(n_parallel)]
    lrs = np.array([lr for i in range(n_parallel)])
    b = -np.hstack([np.linspace(np.min(x), np.max(x[x < 0]), n_hidden // 2),
                        np.linspace(np.min(x[x > 0]), np.max(x), n_hidden // 2)]) * a

    net = CheckingNN(initial_weights=(a, b, c, w), train_setups=train_setups, lrs=lrs)

    vars = net.create_training_vars(use_adam=use_adam)

    for i in range(n_epochs):
        if i % 10000 == 0:
            print('Epoch', i)
        if use_adam:
            net.adam_step(vars)
        else:
            net.train_one_epoch(vars)
        if use_adam and i < 1000:
            net.b = -np.hstack([np.linspace(np.min(x), np.max(x[x < 0]), n_hidden // 2),
                            np.linspace(np.min(x[x > 0]), np.max(x), n_hidden // 2)]) * net.a

    if model_filename is not None:
        utils.serialize(model_filename, net)

    plotter.add(net, instance=0, plot_kinks=False)
    plt.tight_layout(pad=0.2)
    utils.ensureDir(plot_filename)
    plt.savefig(plot_filename)
    plt.close()


def plot_new():
    import show_training
    np.random.seed(1234)
    # x, y = show_training.get_fancy_dataset(show_training.fancy_function, num_samples=300)
    x, y = show_training.get_fancy_dataset4()

    show_training.show_training(x, y, random_seed=8634587, plot_filename='plots/new/convergence.pdf',
                                plot_kinks=False, small_plot=True, small_domain=True,
                                loss_kink_plot=True, lr=1e-2, n_steps=90)

    plot_trained_network(x, y, random_seed=673888, plot_filename='plots/good_nets/good_net_adam_new_673888_16.pdf',
                        n_epochs=200000, lr=1e-2, n_hidden=16, use_adam=True, small_domain=True,
                        model_filename='trained_models/good_net_adam_new_673888_16.p')

    show_training.show_training(x, y, random_seed=8634587, plot_kinks=False,
                                plot_filename='plots/convergence_fused.pdf', small_plot=True,
                                lr=1e-2, n_steps=105, small_domain=True,
                                reference_net=utils.deserialize('trained_models/good_net_adam_new_673888_16.p'))

    # show_training.plot_opt_regression_lines(x, y, filename='./plots/opt_reg_lines.pdf')

    # show_training.animate_movie(x, y, filename='./plots/animation_8634587_105_medium_latex.mp4', random_seed=8634587, n_steps=105,
    #              plot_kinks=False, medium_plot=True)


if __name__ == '__main__':
    #matplotlib.rc('font', **{'family': 'serif', 'serif': ['Times']})
    #matplotlib.rc('mathtext', **{'fontset': 'cm', 'default': 'it'})
    #matplotlib.rcParams['mathtext.fontset'] = 'cm'

    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{times}'

    plot_new()

