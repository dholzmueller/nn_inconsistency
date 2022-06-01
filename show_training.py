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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import imageio
import utils
import torch

from mc_training import CheckingNN
from TrainingSetup import TrainingSetup

class GIFRenderer(object):
    # from https://ndres.me/post/matplotlib-animated-gifs-easily/
    def __init__(self):
        self.images = []

    def add_image(self, fig):
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.images.append(image)

    def save_gif(self, filename, fps):
        #kwargs_write = {'fps': fps, 'quantizer': 'nq'}
        imageio.mimsave(filename, self.images, fps=fps)

class SimpleNNPlotter(object):
    def __init__(self, x, y, gif_filename=None, show_plot=True, small_plot=False, medium_plot=False, thick_lines=False,
                 small_domain=False):
        self.x = x
        self.y = y
        #plt.figure('NN function', figsize=(12, 9))
        plt.figure('NN function', figsize=(3, 2) if small_plot else (6, 4) if medium_plot else (12, 9))
        plt.plot(x, y, 'x', color='k', markersize=4 if small_plot else 10 if medium_plot else 8)
        plt.grid()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        self.gif_filename = gif_filename
        self.gif_renderer = GIFRenderer()
        self.show_plot = show_plot
        self.small_plot = small_plot
        self.medium_plot = medium_plot
        self.thick_lines = thick_lines
        self.small_domain = small_domain

    def add(self, model, instance, plot_kinks=True):
        ax = plt.gca()
        nn_lines = ax.lines[1::2] if plot_kinks else ax.lines[1:]
        for l in nn_lines:
            l.set_alpha(0.2)

        x_model = np.linspace(-1.4, 1.4, 500) if self.small_domain else np.linspace(-4.0, 4.0, 500)
        y_model = model.predict(x_model, instance=instance)
        color_interpol = np.exp(-0.025*len(nn_lines))
        col = matplotlib.colors.hsv_to_rgb((1.2 - 0.6*color_interpol, 1.0, 1.0))
        plt.plot(x_model, y_model, color=col, alpha=1.0, linewidth=(0.5 if self.small_plot else 0.75 if self.medium_plot else 1.0) * (2.0 if self.thick_lines else 1.0))
        #plt.plot(x_model, y_model, color=[0.8*color_interpol, 0.4*color_interpol, (1.0-color_interpol)**2], alpha=1.0)
        if plot_kinks:
            kinks_model = np.clip(-model.b[instance, 0, :] / model.a[instance, 0, :], -4.0, 4.0)
            plt.plot(kinks_model, model.predict(kinks_model, instance=instance), '.', markersize=4 if self.small_plot else 8)
        if self.gif_filename is not None:
            self.gif_renderer.add_image(plt.gcf())

        if self.show_plot:
            plt.pause(0.05)

    def save_gif(self):
        if self.gif_filename is not None:
            self.gif_renderer.save_gif(self.gif_filename, fps=10)

def animate_movie(x, y, filename, random_seed=0, n_hidden=16, n_steps=105, plot_kinks=True, medium_plot=False, thick_lines=True):
    # taken from https://towardsdatascience.com/how-to-create-animated-graphs-in-python-bb619cc2dec1
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='David Holzmüller'), bitrate=50000)

    np.random.seed(random_seed)
    n_parallel = 1
    lr = 2e-3
    plotter = SimpleNNPlotter(x, y, show_plot=False, medium_plot=medium_plot, thick_lines=thick_lines)

    (a, b, c, w) = CheckingNN.createRandomWeights(n_parallel, n_hidden)
    x_weights = 1. / len(x) * np.ones(len(x))
    train_setups = [TrainingSetup(x, x_weights, y) for i in range(n_parallel)]
    lrs = np.array([lr for i in range(n_parallel)])
    net = CheckingNN(initial_weights=(a, b, c, w), train_setups=train_setups, lrs=lrs)

    vars = net.create_training_vars()

    def animate(step):
        print('Step', step)
        plotter.add(net, instance=0, plot_kinks=plot_kinks)
        for step_epoch in range(int(10 * 1.1 ** step) - int(10 * 1.1 ** (step - 1))):
            net.train_one_epoch(vars)

    ani = animation.FuncAnimation(plt.gcf(), animate, frames=n_steps, repeat=True, init_func=lambda: [])
    ani.save(filename, writer=writer, dpi=200)


def show_training(x, y, random_seed=0, n_hidden=16, gif_filename=None, plot_filename=None, swap_variances=False, lr=1e-3,
                  plot_kinks=True, kink_time_plot=False, small_plot=False, loss_time_plot=False, random_kinks=False,
                  equidistant_kinks=False, sampled_kinks=False, n_steps=105, use_adam=False, loss_kink_plot=False,
                  reference_net=None, small_domain=False):
    np.random.seed(random_seed)
    n_parallel = 1
    plotter = SimpleNNPlotter(x, y, gif_filename=gif_filename, small_plot=small_plot,
                              show_plot=(gif_filename is None and plot_filename is None), small_domain=small_domain)

    (a, b, c, w) = CheckingNN.createRandomWeights(n_parallel, n_hidden, swap_variances=swap_variances)
    x_weights = 1. / len(x) * np.ones(len(x))
    train_setups = [TrainingSetup(x, x_weights, y) for i in range(n_parallel)]
    lrs = np.array([lr for i in range(n_parallel)])
    if random_kinks:
        #b = -np.random.uniform(np.min(x), np.max(x), size=n_hidden) * a
        b = -np.random.choice([-3.0, -2.3, -1.6, -1.0, 1.0, 1.6, 2.3, 3.0], size=n_hidden, replace=True) * a
    elif equidistant_kinks:
        #b = -np.linspace(np.min(x), np.max(x), n_hidden) * a
        b = -np.hstack([np.linspace(np.min(x), np.max(x[x<0]), n_hidden//2), np.linspace(np.min(x[x>0]), np.max(x), n_hidden//2)]) * a
    elif sampled_kinks:
        b = -np.random.choice(x, size=n_hidden, replace=True) * a

    net = CheckingNN(initial_weights=(a, b, c, w), train_setups=train_setups, lrs=lrs)

    vars = net.create_training_vars(use_adam=use_adam)

    x_eval = []
    it_counts = []
    losses = []

    x_signed = [x[x>0], x[x<0]]
    y_signed = [y[x>0], y[x<0]]
    min_loss = 0.0
    for sign in [0, 1]:
        X = np.transpose(np.vstack([x_signed[sign], np.ones(len(x_signed[sign]))]))
        Y = y_signed[sign]
        print(X.shape)
        print(Y.shape)
        beta, _, _, _ = np.linalg.lstsq(X, Y)
        print(beta)
        diff = np.dot(X, beta) - Y
        min_loss += np.dot(diff, diff)
    min_loss /= (2*len(x))

    print(min_loss)


    #for step in range(n_steps):
    #    plotter.add(net, instance=0)
    #    for step_epoch in range(int(10 * 1.1**step) - int(10 * 1.1**(step-1))):
    #        net.train_one_epoch(vars)

    step_count = 0
    for it in range(10**6):
         if it in [int(1.1 ** k)-1 for k in range(step_count + 50)]:
             if kink_time_plot or loss_kink_plot:
                 x_eval_current = -net.b[0, 0, :] / net.a[0, 0, :]
                 x_eval.append(x_eval_current)
             if loss_time_plot or loss_kink_plot:
                 pred_diff = net.predict(x, 0) - y
                 loss = np.dot(pred_diff, pred_diff) / (2 * len(x))
                 losses.append(loss - min_loss)
             it_counts.append(it)
             plotter.add(net, instance=0, plot_kinks=plot_kinks)
             step_count += 1
             print('step', step_count)
             if step_count >= n_steps:
                 break
         if use_adam:
             net.adam_step(vars)
         else:
             net.train_one_epoch(vars)

         #if equidistant_kinks and it < 1000:
         #    net.b = -np.hstack([np.linspace(np.min(x), np.max(x[x<0]), n_hidden//2), np.linspace(np.min(x[x>0]), np.max(x), n_hidden//2)]) * net.a


    print('last it:', it_counts[-1])

    if reference_net is not None:
        kinks = -reference_net.b[0, 0, :] / reference_net.a[0, 0, :]
        kinks = np.sort(kinks)
        limit = 1.4 if small_domain else 4.0
        kinks = kinks[kinks>-limit]
        kinks = kinks[kinks<limit]
        x = np.hstack([[-limit], kinks, [limit]])
        y = reference_net.predict(x, instance=0)
        plt.plot(x, y, '.-', color='#88DD88', markevery=list(range(1, len(x)-1)), linewidth=1, markersize=8)

    if gif_filename is not None:
        plotter.save_gif()
    elif plot_filename is not None:
        plt.tight_layout(pad=0.2)
        utils.ensureDir(plot_filename)
        plt.savefig(plot_filename)
        plt.close()
    else:
        plt.show()

    linewidth = 0.5

    if loss_time_plot:
        plt.figure('Loss', figsize=(3, 2))
        plt.semilogy(it_counts, losses, 'k')
        plt.xlabel(r'Epoch $k$')
        plt.ylabel(r"$L_D(W_k) - \inf_{k'} L_D(W_{k'})$")
        utils.ensureDir('./plots/')
        plt.tight_layout(pad=0.2)
        plt.savefig('./plots/loss_new.pdf')
        plt.close('Loss')

    if kink_time_plot:
        #linewidth=0.1
        kink_movement = np.array([x_eval[i] for i in range(len(x_eval))])

        print('number of kinks:', kink_movement.shape[1])

        plt.figure('Kinks (semilogx)', figsize=(3, 2))
        for i in range(kink_movement.shape[1]):
            plt.semilogx(it_counts, kink_movement[:, i], 'k',
                     linewidth=linewidth)
        plt.xlabel(r'Epoch $k$')
        plt.ylabel(r'$-b_{i, k}/a_{i, k}$')
        plt.tight_layout(pad=0.2)
        plt.savefig('./plots/kink_movement_new_logx.pdf')

        plt.figure('Kinks', figsize=(3, 2))
        for i in range(kink_movement.shape[1]):
            plt.plot(it_counts, kink_movement[:, i], 'k',
                         linewidth=linewidth)
        plt.xlabel(r'Epoch $k$')
        plt.ylabel(r'$-b_{i, k}/a_{i, k}$')
        plt.tight_layout(pad=0.2)
        plt.savefig('./plots/kink_movement_new.pdf')
        plt.close('Kinks (semilogx)')
        plt.close('Kinks')

    if loss_kink_plot:
        for simple_labels in [True, False]:
            kink_movement = np.array([x_eval[i] for i in range(len(x_eval))])

            fig, ax1 = plt.subplots(num='Loss and Kinks', figsize=(5, 2))
            ax1.set_xlabel(r'Epoch $k$')
            ax2 = ax1.twinx()

            ax1.set_ylabel(r"loss $-$ final loss" if simple_labels else r"$L_D(W_k) - \inf_{k'} L_D(W_{k'})$")
            ax1.yaxis.label.set_color('red')
            ax1.spines['left'].set_color('red')
            ax2.spines['left'].set_color('red')
            ax1.tick_params(axis='y', colors='red')
            ax1.semilogy(it_counts, losses, 'red')

            for i in range(kink_movement.shape[1]):
                ax2.plot(it_counts, kink_movement[:, i], 'k',
                         linewidth=linewidth)
            ax2.set_ylabel(r'$x$ value of kink' if simple_labels else r'$-b_{i, k}/a_{i, k}$')

            plt.tight_layout(pad=0.2)
            plt.savefig('./plots/loss_and_kinks_simple.pdf' if simple_labels else './plots/loss_and_kinks.pdf')
            plt.savefig('./plots/loss_and_kinks_simple.pgf' if simple_labels else './plots/loss_and_kinks.pgf')
            plt.close('Loss and Kinks')

        # now with shorter period as well
        for simple_labels in [True, False]:
            kink_movement = np.array([x_eval[i] for i in range(len(x_eval))])

            fig, ax1 = plt.subplots(num='Loss and Kinks', figsize=(5, 2))
            ax1.set_xlabel(r'Epoch $k$')
            ax2 = ax1.twinx()

            ax1.set_ylabel(r"loss $-$ final loss" if simple_labels else r"$L_D(W_k) - \inf_{k'} L_D(W_{k'})$")
            ax1.yaxis.label.set_color('red')
            ax1.spines['left'].set_color('red')
            ax2.spines['left'].set_color('red')
            ax1.tick_params(axis='y', colors='red')
            cutoff = np.count_nonzero(np.array(it_counts) <= 500)
            ax1.semilogy(it_counts[:cutoff], losses[:cutoff], 'red')

            for i in range(kink_movement.shape[1]):
                ax2.plot(it_counts[:cutoff], kink_movement[:cutoff, i], 'k',
                         linewidth=linewidth)
            ax2.set_ylabel(r'$x$ value of kink' if simple_labels else r'$-b_{i, k}/a_{i, k}$')

            plt.tight_layout(pad=0.2)
            plt.savefig('./plots/loss_and_kinks_short_simple.pdf' if simple_labels
                        else './plots/loss_and_kinks_short.pdf')
            plt.savefig('./plots/loss_and_kinks_short_simple.pgf' if simple_labels
                        else './plots/loss_and_kinks_short.pgf')
            plt.close('Loss and Kinks')


def target_function(x):
    # \int_1^4 -sin(pi * x) dx = 1/pi [cos(pi*x)]_1^4 = 2/pi
    return -np.sin(np.pi * x) - 2/(3*np.pi)


def get_dataset(f, num_samples):
    x = np.random.uniform(1.0, 4.0, num_samples)
    y_noise = 0.3 * np.random.randn(num_samples)
    return x, f(x) + y_noise


def fancy_function(x):
    #return np.exp(0.25*x) - np.sin(2*np.pi*x)
    #return 0.25*x**2 - x*np.sin(2*np.pi*x)
    return np.exp(0.5*x) - x*np.sin(1.5*np.pi*x)

def get_fancy_dataset(f, num_samples):
    x = 4 * np.hstack([np.random.beta(5, 2, size=num_samples//2), -np.random.beta(5, 2, size=num_samples//2)])
    y = f(x) + 0.3*np.random.randn(len(x))
    ts = TrainingSetup(x=x, x_weights=np.ones(len(x))/len(x), y=y)
    vopt = ts.compute_vopt()
    y[x > 0] -= vopt[0, 1]
    y[x < 0] -= vopt[0, 3]
    return x, y

def remove_intercept(x, y):
    X = np.stack([x, np.ones_like(x)], axis=1)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=1e-8)
    print(beta[1])
    return y - beta[1]

def get_fancy_dataset2():
    np.random.seed(1234)
    x = 4 * np.hstack([np.random.beta(5, 2, size=150), -np.random.beta(5, 2, size=150)])
    y = np.exp(0.5*x) - x*np.sin(1.5*np.pi*x) + 0.3 * np.random.randn(len(x))
    y[x<0] = remove_intercept(x[x<0], y[x<0])
    y[x>0] = remove_intercept(x[x>0], y[x>0])
    return x, y

def get_fancy_dataset3():
    np.random.seed(0)
    beta_l2 = np.sqrt(105/196)  # sqrt of second moment of the involved beta distributions
    x = (1.0/beta_l2) * np.hstack([np.random.beta(5, 2, size=150), -np.random.beta(5, 2, size=150)])
    #y = 0.1*np.exp(1.5*x) - 3*x*np.sin(4*np.pi*x) + 0.3 * np.random.randn(len(x))
    #y = x**3 - x + x*np.cos(4*np.pi*x) + 0.1 * np.random.randn(len(x))
    #y = np.cos(np.pi*x) #+ x*np.cos(4*np.pi*x) + 0.1 * np.random.randn(len(x))
    #y = np.sin(8*x)/(8*x) + x*np.cos(4*np.pi*(x-x/np.sqrt(1+x**2))) + 0.1 * np.random.randn(len(x))
    #y = np.cos(6*np.pi*(x - x/(1.0 + np.abs(x)))) - 0.35 * x**2
    y = np.cos(7 * np.pi * (x - x / np.sqrt(1+x**2))) + 0.2 * x**2 + 0.1 * np.random.randn(len(x))
    y[x<0] = remove_intercept(x[x<0], y[x<0])
    y[x>0] = remove_intercept(x[x>0], y[x>0])
    ynorm_l2 = np.sqrt(np.mean(y**2))
    print('ynorm_l2:', ynorm_l2)
    y = y / ynorm_l2
    print('y mean:', np.mean(y))
    return x, y

class ExampleDistribution:
    # the distribution from Figure 1 in the paper
    def get_x_dim(self):
        return 1

    def get_name(self):
        return 'ex-distr'

    def sample(self, n_parallel, n_samples, device):
        beta = torch.distributions.Beta(5, 2).sample((n_parallel, n_samples))
        bern = torch.distributions.Bernoulli(0.5).sample((n_parallel, n_samples))
        beta_l2 = np.sqrt(105 / 196)  # factor used to normalize X, comes from second moment of the beta distribution
        X = (1.0/beta_l2) * beta * (2 * bern - 1)
        X = X.to(device)

        y = torch.cos(7 * np.pi * (X - X/torch.sqrt(1+X**2))) + 0.2 * X**2
        y_noise = 0.1 * torch.randn_like(X)
        # make optimal intercepts approximately zero
        # make torch.mean(y**2) approximately 1.0
        y = (y + 0.074) / 0.727
        y_noise = y_noise / 0.727
        bayes_rmse = (y_noise**2).mean().sqrt().item()
        return X[:, :, None], y + y_noise, bayes_rmse

def get_fancy_dataset4():
    torch.manual_seed(5)
    x, y, _ = ExampleDistribution().sample(1, 256, 'cpu')
    x, y = x[0, :, 0].numpy(), y[0, :].numpy()
    y[x < 0] = remove_intercept(x[x < 0], y[x < 0])
    y[x > 0] = remove_intercept(x[x > 0], y[x > 0])
    return x, y

def plot_opt_regression_lines(x, y, filename):
    ts = TrainingSetup(x=x, x_weights=np.ones(len(x)) / len(x), y=y)
    vopt = ts.compute_vopt()
    plt.figure('Optimal affine regression lines', figsize=(3, 2))
    plt.grid(True)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    pos_color='#0000FF'
    neg_color='#FF8800'
    pos_x_color='#4444FF'
    neg_x_color='#FFAA44'
    plt.plot(x[x>0], y[x>0], 'x', color=pos_x_color)
    plt.plot(x[x<0], y[x<0], 'x', color=neg_x_color)
    x_pos = np.linspace(0., 4., 20)
    x_neg = np.linspace(-4., 0., 20)
    plt.plot(x_pos, vopt[0, 0] * x_pos + vopt[0, 1], color=pos_color)
    plt.plot(x_neg, vopt[0, 0] * x_neg + vopt[0, 1], '--', color=pos_color)
    plt.plot(x_neg, vopt[0, 2] * x_neg + vopt[0, 3], color=neg_color)
    plt.plot(x_pos, vopt[0, 2] * x_pos + vopt[0, 3], '--', color=neg_color)
    plt.ylim(-5., 15.)

    #plt.show()

    plt.tight_layout(pad=0.2)

    plt.savefig(filename)

