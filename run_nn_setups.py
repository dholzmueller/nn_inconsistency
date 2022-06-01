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

import torch
import torch.nn as nn
import numpy as np
import time
import custom_paths
from pathlib import Path
import utils
import sys


class ParallelLinear(nn.Module):
    def __init__(self, n_parallel, in_features, out_features, act=None, weight_factor=1.0, weight_init_gain=1.0,
                 bias_init_gain=0.0, bias_init_mode='normal'):
        super().__init__()
        self.act = act

        self.weight = nn.Parameter(torch.Tensor(n_parallel, in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(n_parallel, out_features))
        with torch.no_grad():
            # maybe need to rescale for mean field / mup?
            # maybe use mean field in a form that doesn't require changing the lr?
            unif_range = np.sqrt(3) * np.sqrt(in_features) * weight_factor * weight_init_gain
            self.weight.normal_(0.0, weight_init_gain)
            if bias_init_mode == 'normal':
                self.bias.normal_(0.0, 1.0)
            elif bias_init_mode == 'uniform':
                self.bias.uniform_(-np.sqrt(3), np.sqrt(3))
            elif bias_init_mode == 'pos-unif':
                self.bias.uniform_(0, np.sqrt(3))
            elif bias_init_mode == 'neg-unif':
                self.bias.uniform_(-np.sqrt(3), 0)
            elif bias_init_mode == 'kink-unif':
                self.bias.uniform_(-np.sqrt(3), np.sqrt(3))
                self.bias *= self.weight.norm(dim=-2) * weight_factor
            elif bias_init_mode == 'kink-neg-unif':
                self.bias.uniform_(-np.sqrt(3), 0)
                self.bias *= self.weight.norm(dim=-2) * weight_factor
            elif bias_init_mode == 'unif':
                self.bias.uniform_(-unif_range, unif_range)
            elif bias_init_mode == 'unif-neg':
                self.bias.uniform_(-unif_range, 0.0)
            elif bias_init_mode == 'unif-pos':
                self.bias.uniform_(0.0, unif_range)
            elif bias_init_mode == 'pytorch':
                bound = 1 / np.sqrt(in_features)
                self.bias.uniform_(-bound, bound)
            elif bias_init_mode == 'zero':
                self.bias.zero_()
            self.bias *= bias_init_gain
        self.weight_factor = weight_factor
        self.init_batch_done = False
        self.bias_init_mode = bias_init_mode
        self.bias_init_gain = bias_init_gain

    def forward(self, x):
        x = self.weight_factor * x.matmul(self.weight)
        if not self.init_batch_done:
            with torch.no_grad():
                # this is the first batch, do initialization
                if self.bias_init_mode.startswith('he+'):
                    # compute random simplex weights
                    n_simplex = int(self.bias_init_mode[3:])
                    simplex_weights = torch.distributions.Exponential(1.0).sample((x.shape[0], n_simplex, x.shape[2]))
                    simplex_weights = simplex_weights.to(x.device)
                    simplex_weights /= simplex_weights.sum(dim=1)[:, None]
                    # compute the indices to select from
                    idxs = torch.randint(0, x.shape[1], size=(x.shape[0], n_simplex, x.shape[2]), device=x.device)
                    out_selected = x.gather(dim=1, index=idxs)
                    self.bias.set_(-(out_selected * simplex_weights).sum(dim=1))
                elif self.bias_init_mode == 'kink_uniform':
                    min, _ = x.min(dim=1)
                    max, _ = x.max(dim=1)
                    self.bias.set_(-(min + (max-min)*torch.rand_like(self.bias)))
                elif self.bias_init_mode == 'kink-neg-point':
                    idxs = torch.randint(0, x.shape[1], size=(x.shape[0], 1, x.shape[2]), device=x.device)
                    out_selected = x.gather(dim=1, index=idxs)
                    neg_idxs = out_selected[:,0,:] < 0
                    for i in range(self.weight.shape[1]):
                        self.weight[:,i,:][neg_idxs] *= -1
                    self.bias.set_(-out_selected[:, 0, :].abs())
                elif self.bias_init_mode == 'mean':
                    self.bias.set_(-x.mean(dim=1))
            #print(f'Bias std: {self.bias.std().item():g}')
            self.init_batch_done = True
        x = x + self.bias[:, None, :]
        if self.act:
            x = self.act(x)
        return x


class TwoLayerReluNet(nn.Module):
    def __init__(self, n_parallel, input_dim, n_hidden, init_param='kaiming', bias_init_gain=0.0,
                 bias_init_mode='normal'):
        super().__init__()
        if init_param == 'kaiming':
            self.layer1 = ParallelLinear(n_parallel, input_dim, n_hidden, act=torch.relu, bias_init_mode=bias_init_mode,
                                         weight_init_gain=np.sqrt(2 / input_dim), bias_init_gain=bias_init_gain)
            # self.layer2 = ParallelLinear(n_parallel, n_hidden, 1, weight_init_gain=np.sqrt(1 / n_hidden),
            #                              bias_init_gain=bias_init_gain, bias_init_mode=bias_init_mode_2)
            self.layer2 = ParallelLinear(n_parallel, n_hidden, 1, weight_init_gain=np.sqrt(2 / n_hidden),
                                         bias_init_mode='zero')
        else:  # ntk
            self.layer1 = ParallelLinear(n_parallel, input_dim, n_hidden, act=torch.relu, bias_init_mode=bias_init_mode,
                                         weight_factor=np.sqrt(2/input_dim), bias_init_gain=bias_init_gain)
            # self.layer2 = ParallelLinear(n_parallel, n_hidden, 1, weight_factor=np.sqrt(1/n_hidden),
            #                             bias_init_gain=bias_init_gain, bias_init_mode=bias_init_mode_2)
            self.layer2 = ParallelLinear(n_parallel, n_hidden, 1, weight_factor=np.sqrt(2/n_hidden),
                                         bias_init_mode='zero')

    def forward(self, x):
        return self.layer2(self.layer1(x))


def batch_randperm(n_batch, n, device='cpu'):
    # batched randperm:
    # https://discuss.pytorch.org/t/batched-shuffling-of-feature-vectors/30188/4
    # https://github.com/pytorch/pytorch/issues/42502
    return torch.stack([torch.randperm(n, device=device) for i in range(n_batch)], dim=0)


class SimpleParallelTrainer:
    def __init__(self, n_parallel, n_train, n_valid, n_test, data_distribution, init_param='kaiming',
                 bias_init_gain=0.0, bias_init_mode='normal', n_hidden=256, device='cpu', n_epochs=1000, lr=1e-3,
                 valid_epoch_interval=100, seed=0, opt='gd', batch_size=None, n_rep=1):
        self.n_parallel = n_parallel
        self.data_distribution = data_distribution
        self.init_param = init_param
        self.bias_init_gain = bias_init_gain
        self.bias_init_mode = bias_init_mode
        self.n_hidden = n_hidden
        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = device
        self.valid_epoch_interval = valid_epoch_interval
        self.seed = seed
        self.opt = opt
        self.batch_size = batch_size
        self.n_rep = n_rep

    def fit(self, do_plot=False, verbose=False, end_training_callback=None, use_same_ds=False):
        start_time = time.time()

        last_train_mses_list = []
        last_valid_mses_list = []
        last_test_mses_list = []
        best_valid_mses_list = []
        best_test_mses_list = []
        test_bayes_rmse_list = []
        ystd_list = []

        # potentially do multiple repetitions (saves memory compared to larger n_parallel)
        for rep in range(self.n_rep):
            np.random.seed(self.seed + 1024 * rep)
            torch.manual_seed(self.seed + 1024 * rep)



            if use_same_ds:
                Xtrain, ytrain, _ = self.data_distribution.sample(1, self.n_train, device=self.device)
                Xvalid, yvalid, _ = self.data_distribution.sample(1, self.n_valid, device=self.device)
                Xtest, ytest, test_bayes_rmse = self.data_distribution.sample(1, self.n_test,
                                                                              device=self.device)

                Xtrain = torch.repeat_interleave(Xtrain, self.n_parallel, dim=0)
                ytrain = torch.repeat_interleave(ytrain, self.n_parallel, dim=0)
                Xvalid = torch.repeat_interleave(Xvalid, self.n_parallel, dim=0)
                yvalid = torch.repeat_interleave(yvalid, self.n_parallel, dim=0)
                Xtest = torch.repeat_interleave(Xtest, self.n_parallel, dim=0)
                ytest = torch.repeat_interleave(ytest, self.n_parallel, dim=0)
            else:
                Xtrain, ytrain, _ = self.data_distribution.sample(self.n_parallel, self.n_train, device=self.device)
                Xvalid, yvalid, _ = self.data_distribution.sample(self.n_parallel, self.n_valid, device=self.device)
                Xtest, ytest, test_bayes_rmse = self.data_distribution.sample(self.n_parallel, self.n_test,
                                                                              device=self.device)

            model = TwoLayerReluNet(self.n_parallel, self.data_distribution.get_x_dim(), self.n_hidden, self.init_param,
                                    self.bias_init_gain, self.bias_init_mode).to(self.device)

            if self.opt == 'adam':
                opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            elif self.opt == 'gd-mom':
                opt = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
            else:
                opt = torch.optim.SGD(model.parameters(), lr=self.lr)

            # data normalization
            mean = Xtrain.mean(dim=1)
            std = Xtrain.std(dim=1)
            Xtrain = (Xtrain - mean[:, None, :]) / std[:, None, :]
            Xvalid = (Xvalid - mean[:, None, :]) / std[:, None, :]
            Xtest = (Xtest - mean[:, None, :]) / std[:, None, :]
            ymean = ytrain.mean(dim=1)
            ystd = ytrain.std(dim=1)

            ytrain = (ytrain - ymean[:, None]) / ystd[:, None]
            yvalid = (yvalid - ymean[:, None]) / ystd[:, None]
            ytest = (ytest - ymean[:, None]) / ystd[:, None]

            with torch.no_grad():
                model(Xtrain)  # init batch

            best_valid_mses = None
            best_test_mses = None
            valid_mses = None
            test_mses = None

            parallel_idxs = torch.arange(self.n_parallel, dtype=torch.int64, device=self.device)
            parallel_idxs = parallel_idxs[:, None].repeat(1, self.n_train if self.batch_size is None else self.batch_size)

            for i in range(self.n_epochs):
                if self.batch_size is None or self.batch_size == self.n_train:
                    # only one batch per epoch, no shuffling needed
                    train_loss = ((ytrain - model(Xtrain)[:, :, 0])**2).mean(dim=1).sum(dim=0)
                    if np.isnan(train_loss.item()):
                        return None
                    train_loss.backward()
                    opt.step()
                    for p in model.parameters():
                        p.grad = None
                    train_loss_item = train_loss.item()
                else:
                    # do minibatching
                    assert(self.n_train % self.batch_size == 0)
                    n_steps = self.n_train // self.batch_size
                    perm = batch_randperm(self.n_parallel, self.n_train, device=self.device)
                    train_losses = []
                    for step in range(n_steps):
                        idxs = perm[:, step*self.batch_size:(step+1)*self.batch_size]

                        diff = (ytrain[parallel_idxs, idxs] - model(Xtrain[parallel_idxs, idxs])[:, :, 0])
                        train_loss = (diff ** 2).mean(dim=1).sum(dim=0)
                        train_losses.append(train_loss.item())
                        if np.isnan(train_loss.item()):
                            return None
                        train_loss.backward()
                        opt.step()
                        for p in model.parameters():
                            p.grad = None
                    train_loss_item = np.mean(train_losses)

                if (i+1) % self.valid_epoch_interval == 0:
                    with torch.no_grad():
                        valid_mses = ((yvalid - model(Xvalid)[:, :, 0])**2).mean(dim=1)
                        test_mses = ((ytest - model(Xtest)[:, :, 0])**2).mean(dim=1)

                        if best_valid_mses is None:
                            best_valid_mses = valid_mses
                            best_test_mses = test_mses
                        else:
                            improved = valid_mses < best_valid_mses
                            best_valid_mses[improved] = valid_mses[improved]
                            best_test_mses[improved] = valid_mses[improved]

                        if verbose:
                            print(f'Epoch {i+1}:')
                            print(f'Train RMSE: {np.sqrt(train_loss_item/self.n_parallel):g}')
                            print(f'Test RMSE:  {torch.sqrt(test_mses).mean().item():g}')
                            print(f'Best Test RMSE: {torch.sqrt(best_test_mses).mean().item():g}')
                            print()

                if i+1 == self.n_epochs and end_training_callback is not None:
                    end_training_callback(model, Xtrain, ytrain, mean, std, ymean, ystd, valid_mses)

            with torch.no_grad():
                if self.batch_size is None or self.batch_size == self.n_train:
                    train_mses = ((ytrain - model(Xtrain)[:, :, 0])**2).mean(dim=1)
                else:
                    assert (self.n_train % self.batch_size == 0)
                    n_steps = self.n_train // self.batch_size
                    train_sses = []
                    for i in range(n_steps):
                        start = i * self.batch_size
                        stop = (i+1) * self.batch_size
                        train_sses.append(
                            ((ytrain[:, start:stop] - model(Xtrain[:, start:stop, :])[:, :, 0])**2).sum(dim=1))
                    train_mses = sum(train_sses) / self.n_train

            last_train_mses_list.append(train_mses)
            last_valid_mses_list.append(valid_mses)
            last_test_mses_list.append(test_mses)
            best_valid_mses_list.append(best_valid_mses)
            best_test_mses_list.append(best_test_mses)
            test_bayes_rmse_list.append(test_bayes_rmse)
            ystd_list.append(ystd)

        end_time = time.time()
        if verbose:
            print(f'Time: {end_time - start_time:g} s')

        test_bayes_rmse = np.mean(test_bayes_rmse_list)
        train_mses = torch.cat(last_train_mses_list, dim=0)
        valid_mses = torch.cat(last_valid_mses_list, dim=0)
        test_mses = torch.cat(last_test_mses_list, dim=0)
        best_valid_mses = torch.cat(best_valid_mses_list, dim=0)
        best_test_mses = torch.cat(best_test_mses_list, dim=0)
        ystd = torch.cat(ystd_list, dim=0)

        results = {
            'test_bayes_rmse': test_bayes_rmse,
            'last_train_rmse': (torch.sqrt(train_mses) * ystd).mean().item(),
            'last_valid_rmse': (torch.sqrt(valid_mses) * ystd).mean().item(),
            'last_test_rmse': (torch.sqrt(test_mses) * ystd).mean().item(),
            'best_valid_rmse': (torch.sqrt(best_valid_mses) * ystd).mean().item(),
            'best_test_rmse': (torch.sqrt(best_test_mses) * ystd).mean().item(),
            'last_train_rmse_std': (torch.sqrt(train_mses) * ystd).std().item(),
            'last_valid_rmse_std': (torch.sqrt(valid_mses) * ystd).std().item(),
            'last_test_rmse_std': (torch.sqrt(test_mses) * ystd).std().item(),
            'best_valid_rmse_std': (torch.sqrt(best_valid_mses) * ystd).std().item(),
            'best_test_rmse_std': (torch.sqrt(best_test_mses) * ystd).std().item(),
            'time': end_time - start_time,
        }

        # save results: train, valid, test, best valid, best test, respective standard deviations
        # hyperparameters
        # training time?
        # epochs of best results?

        # plot result
        if self.data_distribution.get_x_dim() == 1 and do_plot:
            with torch.no_grad():
                import matplotlib.pyplot as plt
                plt.plot(Xtest[0,:,0].numpy(), ytest[0,:].numpy(), 'x')
                x = torch.linspace(Xtest[0,:,0].min().item(), Xtest[0,:,0].max().item(), 400)
                y = model(x[None, :, None].repeat(self.n_parallel, 1, 1))[0, :, 0]
                plt.plot(x.cpu().numpy(), y.cpu().numpy())
                plt.grid(True)
                plt.show()

        return results


class RBFDataDistribution:
    def __init__(self, d):
        self.d = d

    def get_x_dim(self):
        return self.d

    def get_name(self):
        return f'rbf-distr-{self.d}'

    def sample(self, n_parallel, n_samples, device):
        normal = torch.randn(size=(n_parallel, n_samples, self.d), device=device)
        unif = torch.rand(size=(n_parallel, n_samples), device=device)
        unif_range = np.sqrt(3)  # unit variance
        radii = ((-unif_range) + 2 * unif_range * unif)
        X = radii[:, :, None] * (normal / normal.norm(dim=2, keepdim=True))
        y = torch.exp(-radii**2)
        return X, y, 0.0


class RadialDataDistribution:
    def __init__(self, d):
        self.d = d

    def get_x_dim(self):
        return self.d

    def get_name(self):
        return f'rad-distr-{self.d}'

    def sample(self, n_parallel, n_samples, device):
        normal = torch.randn(size=(n_parallel, n_samples, self.d), device=device)
        unif = torch.rand(size=(n_parallel, n_samples), device=device)
        # we used np.sqrt(3) to generate the data instead of 1, but the data is normalized anyway
        # so it doesn't matter
        unif_range = np.sqrt(3)
        # unif_range = 1.0
        radii = unif_range * unif
        X = radii[:, :, None] * (normal / normal.norm(dim=2, keepdim=True))
        y = torch.cos(2*np.pi*unif)
        return X, y, 0.0


class ExampleDistributionOld:
    # the distribution from Figure 1 in the paper
    def get_x_dim(self):
        return 1

    def get_name(self):
        return 'ex-distr-old'

    def sample(self, n_parallel, n_samples, device):
        beta = torch.distributions.Beta(5, 2).sample((n_parallel, n_samples))
        bern = torch.distributions.Bernoulli(0.5).sample((n_parallel, n_samples))
        X = 4 * beta * (2 * bern - 1)
        X = X.to(device)

        y = torch.exp(0.5*X) - X * torch.sin(1.5 * np.pi * X) + 0.3 * torch.randn_like(X)
        # approximately make intercepts of optimal regression lines zero
        y[X<0] += 1.205
        y[X>0] += 3.136
        return X[:, :, None], y, None  # bayes rmse not yet implemented


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


def get_fancy_dataset3():
    np.random.seed(0)
    beta_l2 = np.sqrt(105/196)  # sqrt of second moment of the involved beta distributions
    x = (1.0/beta_l2) * np.hstack([np.random.beta(5, 2, size=10000000), -np.random.beta(5, 2, size=10000000)])
    #y = 0.1*np.exp(1.5*x) - 3*x*np.sin(4*np.pi*x) + 0.3 * np.random.randn(len(x))
    #y = x**3 - x + x*np.cos(4*np.pi*x) + 0.1 * np.random.randn(len(x))
    #y = np.cos(np.pi*x) #+ x*np.cos(4*np.pi*x) + 0.1 * np.random.randn(len(x))
    #y = np.sin(8*x)/(8*x) + x*np.cos(4*np.pi*(x-x/np.sqrt(1+x**2))) + 0.1 * np.random.randn(len(x))
    #y = np.cos(6*np.pi*(x - x/(1.0 + np.abs(x)))) - 0.35 * x**2
    y = np.cos(7 * np.pi * (x - x / np.sqrt(1+x**2))) + 0.2 * x**2 + 0.1 * np.random.randn(len(x))
    y[x<0] = remove_intercept(x[x<0], y[x<0])
    y[x>0] = remove_intercept(x[x>0], y[x>0])
    ynorm_l2 = np.sqrt(np.mean(y**2))
    y = y / ynorm_l2
    print('ynorm_l2:', ynorm_l2)
    print('y mean:', np.mean(y))
    print('x mean:', np.mean(x))
    print('x std:', np.std(x))
    return x, y


def remove_intercept(x, y):
    X = np.stack([x, np.ones_like(x)], axis=1)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=1e-8)
    print(-beta[1])
    return y - beta[1]


def get_fancy_dataset():
    np.random.seed(1234)
    x = 4 * np.hstack([np.random.beta(5, 2, size=10000000), -np.random.beta(5, 2, size=10000000)])
    y = np.exp(0.5*x) - x*np.sin(1.5*np.pi*x) + 0.3 * np.random.randn(len(x))
    y[x<0] = remove_intercept(x[x<0], y[x<0])
    y[x>0] = remove_intercept(x[x>0], y[x>0])
    return x, y


def run(init_param='kaiming', device='cpu'):
    dist_grid = [ExampleDistribution()] + [RadialDataDistribution(d=2**k) for k in range(7)]
    std_grid = [0.1, 0.5, 1.0, 2.0]
    # bi_grid = [('zero', 0.0), ('he+5', 0.0), ('he+1', 0.0), ('kink_uniform', 0.0)] \
    #             + [(bim, big) for big in std_grid for bim in ['normal', 'uniform']] \
    #             + [('pos-unif', 1.0), ('neg-unif', 1.0), ('kink-unif', 1.0), ('kink-neg-unif', 1.0),
    #                ('kink-neg-point', 0.0)]
    bi_grid = [('zero', 0.0), ('unif', 1.0), ('unif-pos', 1.0), ('unif-neg', 1.0), ('kink-neg-unif', 1.0),
               ('pytorch', 1.0), ('kink-neg-point', 0.0)]
    for opt in ['gd', 'gd-mom', 'adam']:
        if opt == 'adam':
            base_lr = 8e-2 if init_param == 'ntk' else 1e-2
        else:
            base_lr = 8e-1 if init_param == 'ntk' else 8e-3
        lr_grid = [base_lr * np.sqrt(2)**k for k in range(-12, 11)]
        for dist in dist_grid:
            d = dist.get_x_dim()
            for bim, big in bi_grid:
                folder_name = f'{init_param}_{opt}_{dist.get_name()}_{bim}-{big:g}'
                path = Path(custom_paths.get_results_path()) / 'nn_comparison' / folder_name
                for lr in lr_grid:
                    print(f'Running combination {folder_name} with lr {lr:g}')
                    file = path / f'{lr:g}.pkl'
                    utils.ensureDir(file)
                    if utils.existsFile(file):
                        continue
                    n_rep = 2 if d == 64 else 1
                    trainer = SimpleParallelTrainer(n_parallel=100//n_rep, n_train=256*d, n_valid=1024, n_test=1024,
                                                    data_distribution=dist, lr=lr, bias_init_gain=big, batch_size=256,
                                                    bias_init_mode=bim, init_param=init_param, n_epochs=8192//d, seed=0,
                                                    device=device, n_hidden=512, opt=opt, valid_epoch_interval=64//d,
                                                    n_rep=n_rep)
                    results = trainer.fit(do_plot=False, verbose=False)
                    if results is None:
                        print('Got NaN values')
                    utils.serialize(file, {'trainer': trainer, 'results': results})


def run_finer_lrs(init_param='kaiming', device='cpu'):
    dist_grid = [ExampleDistribution()] + [RadialDataDistribution(d=2**k) for k in range(7)]
    std_grid = [0.1, 0.5, 1.0, 2.0]
    # bi_grid = [('zero', 0.0), ('he+5', 0.0), ('he+1', 0.0), ('kink_uniform', 0.0)] \
    #             + [(bim, big) for big in std_grid for bim in ['normal', 'uniform']] \
    #             + [('pos-unif', 1.0), ('neg-unif', 1.0), ('kink-unif', 1.0), ('kink-neg-unif', 1.0),
    #                ('kink-neg-point', 0.0)]
    bi_grid = [('zero', 0.0), ('unif', 1.0), ('unif-pos', 1.0), ('unif-neg', 1.0), ('kink-neg-unif', 1.0),
               ('pytorch', 1.0), ('kink-neg-point', 0.0)]
    for opt in ['gd', 'gd-mom', 'adam']:
        for dist in dist_grid:
            d = dist.get_x_dim()
            for bim, big in bi_grid:
                folder_name = f'{init_param}_{opt}_{dist.get_name()}_{bim}-{big:g}'
                path = Path(custom_paths.get_results_path()) / 'nn_comparison' / folder_name
                best_lr_file = Path(custom_paths.get_results_path()) / 'nn_comparison' / f'{folder_name}_bestlr.pkl'
                if not utils.existsFile(best_lr_file):
                    sys.stderr.write('best lr file {best_lr_file} does not exist!\n')
                    continue
                best_lr = utils.deserialize(best_lr_file)
                lr_grid = [best_lr * (2**(k/8)) for k in range(-3, 4)]
                for lr in lr_grid:
                    print(f'Running combination {folder_name} with lr {lr:g}')
                    file = path / f'{lr:g}.pkl'
                    utils.ensureDir(file)
                    if utils.existsFile(file):
                        continue
                    n_rep = 2 if d == 64 else 1
                    trainer = SimpleParallelTrainer(n_parallel=100//n_rep, n_train=256*d, n_valid=1024, n_test=1024,
                                                    data_distribution=dist, lr=lr, bias_init_gain=big, batch_size=256,
                                                    bias_init_mode=bim, init_param=init_param, n_epochs=8192//d, seed=0,
                                                    device=device, n_hidden=512, opt=opt, valid_epoch_interval=64//d,
                                                    n_rep=n_rep)
                    results = trainer.fit(do_plot=False, verbose=False)
                    if results is None:
                        print('Got NaN values')
                    utils.serialize(file, {'trainer': trainer, 'results': results})


def run_old(init_param='kaiming', device='cpu'):
    dist_grid = [ExampleDistribution()] + [RBFDataDistribution(d=2**k) for k in range(7)]
    std_grid = [0.1, 0.5, 1.0, 2.0]
    bi_grid = [('zero', 0.0), ('he+5', 0.0), ('he+1', 0.0), ('kink_uniform', 0.0)] \
                + [(bim, big) for big in std_grid for bim in ['normal', 'uniform']]
    for opt in ['gd', 'gd-mom', 'adam']:
        base_lr = 1e-2 if opt == 'adam' else (4e-1 if init_param == 'ntk' else 8e-3)
        lr_grid = [base_lr * np.sqrt(2)**k for k in range(-8, 9)]
        for dist in dist_grid:
            for bim, big in bi_grid:
                folder_name = f'{init_param}_{opt}_{dist.get_name()}_{bim}-{big:g}'
                path = Path(custom_paths.get_results_path()) / 'nn_comparison' / folder_name
                for lr in lr_grid:
                    print(f'Running combination {folder_name} with lr {lr:g}')
                    file = path / f'{lr:g}.pkl'
                    utils.ensureDir(file)
                    if utils.existsFile(file):
                        continue
                    torch.cuda.empty_cache()
                    trainer = SimpleParallelTrainer(n_parallel=100, n_train=256, n_valid=1024, n_test=1024,
                                                    data_distribution=dist, lr=lr, bias_init_gain=big,
                                                    bias_init_mode=bim, init_param=init_param, n_epochs=10000, seed=0,
                                                    device=device, n_hidden=256, opt=opt)
                    results = trainer.fit(do_plot=False, verbose=False)
                    if results is None:
                        print('Got NaN values')
                    utils.serialize(file, {'trainer': trainer, 'results': results})


def test_radial(d):
    dist = RadialDataDistribution(d=d)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x = torch.ones(1, device=device)  # warmup, doesn't count towards time measurement
    # trainer = SimpleParallelTrainer(n_parallel=10, n_train=256, n_valid=1000, n_test=1000, data_distribution=dist,
    #                                 lr=4e-3, bias_init_gain=0.0, bias_init_mode='he+1', init_param='kaiming',
    #                                 n_epochs=1000, seed=0, device=device, opt='gd')
    trainer = SimpleParallelTrainer(n_parallel=10, n_train=256*d, n_valid=1000, n_test=1000, data_distribution=dist,
                                    lr=4e-3, bias_init_gain=1.0, bias_init_mode='kink-neg-point', init_param='ntk',
                                    n_epochs=16384//d, seed=0, device=device, opt='adam', batch_size=256,
                                    valid_epoch_interval=64//d)
    results = trainer.fit(verbose=True, do_plot=True)
    print(results)


def save_best_lrs():
    base_path = Path(custom_paths.get_results_path()) / 'nn_comparison'

    for results_dir in base_path.iterdir():
        if not results_dir.is_dir():
            continue
        bestlr_filename = base_path / f'{results_dir.name}_bestlr.pkl'
        if utils.existsFile(bestlr_filename):
            continue  # has already been computed, don't recompute
            # since maybe now results from run_finer_lrs are there and would change best_lr
        valid_dir_results = []
        for results_file in results_dir.iterdir():
            results = utils.deserialize(results_file)
            if results['results'] is not None:
                valid_dir_results.append(results)

        if len(valid_dir_results) > 0:
            best_idx = np.argmin([r['results']['best_valid_rmse'] for r in valid_dir_results])
            best_lr = valid_dir_results[best_idx]['trainer'].lr
            print(best_lr)
            utils.serialize(bestlr_filename, best_lr)


if __name__ == '__main__':
    # the following two run() statements can also be executed separately / in parallel on two different GPUs
    run(init_param='kaiming', device='cuda:0')
    run(init_param='ntk', device='cuda:0')
    # this saves the best lrs so far. should not be run again after run_finer_lrs() 
    # since this might then update the best lrs around which run_finer_lrs() refines
    save_best_lrs()
    # the following two run() statements can also be executed separately / in parallel on two different GPUs
    run_finer_lrs(init_param='kaiming', device='cuda:0')
    run_finer_lrs(init_param='ntk', device='cuda:0')
    pass

