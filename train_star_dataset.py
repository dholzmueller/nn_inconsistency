# Copyright 2020 The nn_inconsistency Authors. All Rights Reserved.
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
import torch.nn.functional as F
import numpy as np
import fire
from pathlib import Path
import datetime
import custom_paths

import utils


class ParallelLinear(nn.Module):
    def __init__(self, n_parallel, in_features, out_features, act=None, random_bias=False):
        super().__init__()
        self.act = act

        # use Kaiming init
        self.weight = nn.Parameter(torch.Tensor(n_parallel, in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(n_parallel, out_features))
        with torch.no_grad():
            self.weight.normal_(0.0, np.sqrt(2.0/in_features))
            if random_bias:
                self.bias.normal_(0.0, np.sqrt(2.0/in_features))
            else:
                self.bias.zero_()

    def forward(self, x):
        x = torch.bmm(x, self.weight) + self.bias[:, None, :]
        if self.act:
            x = self.act(x)
        return x


def get_standard_model(n_in, n_parallel, hidden_sizes, random_bias=False, act='relu'):
    act_func = None
    if act == 'tanh':
        act_func = F.tanh
    elif act == 'relu':
        act_func = F.relu
    sizes = [n_in] + hidden_sizes + [1]
    layers = [ParallelLinear(n_parallel, in_features, out_features, act=act_func, random_bias=random_bias)
              for in_features, out_features in zip(sizes[:-1], sizes[1:])]
    layers[-1].act = None
    return nn.Sequential(*layers)


def get_standard_layers(n_in, n_parallel, hidden_sizes, device, random_bias=False, act='relu'):
    act_func = None
    if act == 'tanh':
        act_func = F.tanh
    elif act == 'relu':
        act_func = F.relu
    sizes = [n_in] + hidden_sizes + [1]
    layers = [ParallelLinear(n_parallel, in_features, out_features, act=act_func, random_bias=random_bias).to(device)
              for in_features, out_features in zip(sizes[:-1], sizes[1:])]
    layers[-1].act = None
    return layers


def get_sample_weights(n_parallel, n_samples, n_virtual_samples):
    return np.random.multinomial(n_virtual_samples, [1. / n_samples] * n_samples, size=n_parallel) / n_virtual_samples


def get_kernel_matrices(model, x):
    n_parallel = x.shape[0]
    batch_size = x.shape[1]
    grads = []
    for i in range(batch_size):
        y = model(x[:, i:i+1, :])[:, 0, 0]
        y.backward(torch.ones(n_parallel, device=x.device))
        grads.append(torch.cat([p.view(n_parallel, -1) for p in model.parameters()], dim=1).clone())

        model.zero_grad()

    grads_matrix = torch.stack(grads, dim=1)
    return grads_matrix.bmm(grads_matrix.transpose(1, 2))


def get_lin_lrs(model, x, y):
    device = x.device
    n_parallel = x.shape[0]
    # automatically determine lr (such that linearization is quite accurate)
    start_params = [p.clone() for p in model.parameters()]
    # lr_candidates = [2**k for k in range(-20, 1)]
    lr_candidates = np.logspace(-6, 0, 200)
    start_mses = ((model(x).squeeze(2) - y) ** 2).mean(dim=1)
    start_mses.backward(torch.ones(n_parallel, device=device))
    grad_sq_norm = torch.zeros(n_parallel, device=device)
    for p in model.parameters():
        grad_sq_norm += (p.grad ** 2).sum(dim=list(range(1, len(p.grad.shape))))

    lrs = torch.zeros(n_parallel, device=device)
    lin_close = torch.ones(n_parallel, dtype=torch.bool, device=device)

    for lr in lr_candidates:
        with torch.no_grad():
            for p in model.parameters():
                p += (-lr) * p.grad

        mses = ((model(x).squeeze(2) - y) ** 2).mean(dim=1)
        lin_close = lin_close & (mses <= start_mses - 0.9 * lr * grad_sq_norm) \
                    & (mses >= start_mses - 1.1 * lr * grad_sq_norm)
        lrs[lin_close] = lr
        # print(lin_close)

        with torch.no_grad():  # reset parameters to start parameters
            for p, sp in zip(model.parameters(), start_params):
                p.set_(sp.clone())

    model.zero_grad()

    return lrs


def compute_min_linearized_predictions(x: torch.Tensor, y: torch.Tensor):
    x_norms = x.norm(dim=1)
    x_normalized = x / x_norms[:, None]
    done = []
    y_pred = y.clone()
    for i in range(x.shape[0]):
        if i in done:
            continue
        new_group = []
        for j in range(x.shape[0]):
            if (x_normalized[i] - x_normalized[j]).norm().item() < 1e-5:
                new_group.append(j)
        done += new_group
        if len(new_group) <= 2:
            continue  # can be perfectly fitted using an affine function
        idxs = torch.LongTensor(new_group, device=x.device)
        X = torch.stack([x_norms[idxs], torch.ones_like(x_norms[idxs])], dim=1)
        sol, qr = y[idxs, None].lstsq(X)
        y_pred[idxs] = X.matmul(sol[:2, 0])
    return y_pred


def compute_min_linearized_prediction_errors(x: torch.Tensor, y: torch.Tensor, sample_weights: torch.Tensor):
    x_norms = x.norm(dim=1)
    x_normalized = x / x_norms[:, None]
    done = []
    errors = torch.zeros_like(sample_weights[:, 0])
    sqrt_sample_weights = sample_weights.sqrt()
    for i in range(x.shape[0]):
        if i in done:
            continue
        new_group = []
        for j in range(x.shape[0]):
            if (x_normalized[i] - x_normalized[j]).norm().item() < 1e-5:
                new_group.append(j)
        done += new_group
        if len(new_group) <= 2:
            continue  # can be perfectly fitted using an affine function
        idxs = torch.as_tensor(new_group, dtype=torch.long, device=x.device)
        X = torch.stack([x_norms[idxs], torch.ones_like(x_norms[idxs])], dim=1)
        for j in range(errors.shape[0]):
            sol, qr = (sqrt_sample_weights[j, idxs, None] * y[idxs, None]).lstsq(sqrt_sample_weights[j, idxs, None] * X)
            y_pred = X.matmul(sol[:2, 0])
            errors[j] += ((y[idxs] - y_pred)**2 * sample_weights[j, idxs]).sum()
    return errors


def get_device(device_number=0):
    return torch.device(f'cuda:{device_number}') if torch.cuda.is_available() else torch.device('cpu')


class ModelTrainer:
    def __init__(self, x_train, y_train, n_parallel, hidden_sizes, random_bias=False,
                 act='relu', device_number=0, n_virtual_samples=0, version=0):
        torch.manual_seed(version)
        np.random.seed(version)
        self.device = get_device(device_number)
        self.layers = get_standard_layers(x_train.shape[1], n_parallel, hidden_sizes, self.device,
                                          random_bias=random_bias, act=act)
        self.model = nn.Sequential(*self.layers)
        x_train = torch.as_tensor(x_train).to(self.device).float()
        y_train = torch.as_tensor(y_train).to(self.device).float()
        self.x = x_train[None].repeat(n_parallel, 1, 1)
        self.y = y_train[None].repeat(n_parallel, 1)
        self.n_parallel = n_parallel
        self.epoch_count = 0
        self.mse_first_crossed = -torch.ones(n_parallel, dtype=torch.long, device=self.device)
        self.kink_first_crossed = -torch.ones(n_parallel, dtype=torch.long, device=self.device)
        self.n_virtual_samples = n_virtual_samples
        if n_virtual_samples > 0:
            self.sample_weights = torch.as_tensor(get_sample_weights(n_parallel, self.x.shape[1], n_virtual_samples),
                                                  device=self.device, dtype=torch.float32)
        else:
            self.sample_weights = torch.ones(n_parallel, self.x.shape[1], device=self.device) / self.x.shape[1]

        self.activ_patterns = []
        val = self.x
        for layer_idx, l in enumerate(self.layers[:-1]):
            val = l(val)
            self.activ_patterns.append(val > 0)

        # can't use compute_min_linearized_prediction because it does not take into account the sample weights
        # lin_y_pred = compute_min_linearized_predictions(x_train, y_train)
        # print('lin_y_pred[0:5]:', lin_y_pred[0:5])
        # sq_errs = (lin_y_pred - y_train)**2
        # self.lin_thresholds = 0.5 * self.sample_weights.matmul(sq_errs)
        # use compute_min_linearized_prediction_errors instead
        self.lin_thresholds = 0.5 * compute_min_linearized_prediction_errors(x_train, y_train, self.sample_weights)
        print('lin_thresholds:', self.lin_thresholds)
        self.thresholds = 0.999999 * self.lin_thresholds

        self.lin_lrs = get_lin_lrs(self.model, self.x, self.y)
        kernel_matrices = get_kernel_matrices(self.model, self.x)
        # Let K = kernel_matrices[i] for some i.
        # Then, the kernel matrix for virtual samples is EKE^T,
        # where E is a "one-hot mapping"
        # from the (duplicated) virtual samples to the real samples. Then, \lambda_max(E K E^T) = \lambda_max(K E^T E),
        # where E^TE = diag(n_1, ..., n_n) contains the frequencies of the individual points in the virtual dataset.
        # To make the matrix symmetric again, we can then use \lambda_max(sqrt(E^T E)Ksqrt(E^T E))
        # moreover, we really want 1/\lambda_max(1/n * kernel matrix), since
        # kernel matrix = basis-change(n*AM), where n is the number of training samples
        # we want the analog of having 1/\lambda_max(AM), so 1/\lambda_max(1/n * kernel matrix)
        # (this is because we use the MSE and not the sum of squared errors)
        sqrt_weights = self.sample_weights.sqrt()
        kernel_matrices = kernel_matrices * sqrt_weights[:, :, None] * sqrt_weights[:, None, :]
        eigvals, eigvecs = kernel_matrices.symeig()
        max_eigs = eigvals[:, -1]

        self.eig_lrs = 1.0 / max_eigs

        print('lrs via linearization:', self.lin_lrs)
        print('lrs via 1/maxeig:', self.eig_lrs)

        self.lrs = self.eig_lrs

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.mse_first_crossed = self.mse_first_crossed.to(device)
        self.kink_first_crossed = self.kink_first_crossed.to(device)
        self.sample_weights = self.sample_weights.to(device)
        self.activ_patterns = [p.to(device) for p in self.activ_patterns]
        self.lin_lrs = self.lin_lrs.to(device)
        self.eig_lrs = self.eig_lrs.to(device)
        self.lrs = self.lrs.to(device)
        self.model.to(device)
        return self

    def train(self, n_epochs):
        for i in range(self.epoch_count, n_epochs+1):
            val = self.x
            for layer_idx, l in enumerate(self.layers[:-1]):
                val = l(val)
                activ_pattern_identical = (self.activ_patterns[layer_idx] == (val > 0)).all(dim=2).all(dim=1)
                self.kink_first_crossed[(self.kink_first_crossed == -1) & (~activ_pattern_identical)] = self.epoch_count

            y_pred = self.layers[-1](val).squeeze(2)
            mses = 0.5 * ((y_pred - self.y) ** 2 * self.sample_weights).sum(dim=1)
            # todo: periodically save y_pred (but detach it to avoid memory overflow)
            # todo: track when which NN crosses the threshold
            # todo: also track when a kink reaches a data point for which NN

            self.mse_first_crossed[(self.mse_first_crossed == -1) & (mses < self.thresholds)] = self.epoch_count

            if self.epoch_count % 1000 == 0:
                # todo print mean and quantiles
                print('Epoch', i)
                print(f'MMSE: {mses.mean().item():g}')
                sorted_mses, _ = (mses/self.lin_thresholds).sort()  # todo this requires the thresholds not to be zero
                quantiles = [sorted_mses[int(i * (self.n_parallel - 1) / 10)].item() for i in range(11)]
                quantile_strings = [f'{q:g}' for q in quantiles]
                quantiles_str = '[' + ', '.join(quantile_strings) + ']'
                print(f'Quantiles: {quantiles_str}')
                print(f'Fraction of NNs with MSE crossed: {(self.mse_first_crossed != -1).float().mean().item():g}')
                print(f'Fraction of NNs with kink crossed: {(self.kink_first_crossed != -1).float().mean().item():g}')
                print()
            mses.backward(torch.ones(self.n_parallel, device=self.device))
            with torch.no_grad():
                for p in self.model.parameters():
                    lr_factor = (-self.lrs)[:, None, None] if len(p.shape) == 3 else (-self.lrs)[:, None]
                    p.addcmul_(lr_factor, p.grad, value=1.0)
                    p += lr_factor * p.grad
                    p.grad.zero_()
            self.epoch_count += 1


def get_cross_dataset():
    x_train = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    y_train = torch.tensor([-1.0, 2.0, -1.0, 1.0, -2.0, 1.0])
    x1 = torch.stack([torch.zeros_like(x_train), x_train], dim=1)
    x2 = torch.stack([x_train, torch.zeros_like(x_train)], dim=1)
    x_train = torch.cat([x1, x2], dim=0)
    y_train = torch.cat([y_train, y_train], dim=0)
    #print(x_train)
    #print(y_train)
    return x_train, y_train

def get_double_cross_dataset():
    x_train = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    y_train = torch.tensor([-1.0, 2.0, -1.0, 1.0, -2.0, 1.0])
    x1 = torch.stack([torch.zeros_like(x_train), x_train], dim=1)
    x2 = torch.stack([x_train, torch.zeros_like(x_train)], dim=1)
    x3 = torch.stack([x_train, x_train], dim=1)
    x4 = torch.stack([x_train, -x_train], dim=1)
    x_train = torch.cat([x1, x2, x3, x4], dim=0)
    y_train = y_train.repeat(4)
    #print(x_train)
    #print(y_train)
    return x_train, y_train

def get_standard_dataset():
    x_train = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])[:, None]
    y_train = torch.tensor([-1.0, 2.0, -1.0, 1.0, -2.0, 1.0])
    return x_train, y_train

def get_2d_star_dataset(k=11, dist=0.5):
    x_train = torch.tensor([1.0-dist, 1.0, 1.0+dist])[:, None]
    y_train = torch.tensor([1.0, -2.0, 1.0])
    x_train = torch.cat([x_train * torch.as_tensor([[np.cos(2*np.pi*i/k), np.sin(2*np.pi*i/k)]]) for i in range(k)],
                        dim=0)
    y_train = y_train.repeat(k)
    return x_train, y_train

def get_random_with_line_ds():
    torch.manual_seed(1)
    x_train = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    y_train = torch.tensor([-1.0, 2.0, -1.0, 1.0, -2.0, 1.0])
    x_train = torch.stack([torch.zeros_like(x_train), x_train], dim=1)
    x_train = torch.cat([x_train, torch.randn_like(x_train)], dim=0)
    y_train = torch.cat([y_train, torch.randn_like(y_train)], dim=0)
    #import matplotlib.pyplot as plt
    #plt.plot(x_train[:, 0].numpy(), x_train[:, 1].numpy(), '.')
    #plt.show()
    #print(x_train)
    #print(y_train)
    return x_train, y_train


def get_multi_axis_ds(d=5):
    torch.manual_seed(1)
    np.random.seed(1)
    n_points_per_dim = 5
    x_train = torch.zeros(size=(d*n_points_per_dim, d))
    y_train = torch.zeros(size=(d*n_points_per_dim,))

    thresholds = []

    for dim in range(d):
        x = np.random.uniform(1.0, 2.0, size=n_points_per_dim)
        y = np.random.uniform(-1.0, 1.0, size=n_points_per_dim)
        X = np.stack([x, np.ones_like(x)], axis=1)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=1e-8)
        intercept = beta[1]
        diff = X @ beta - y
        thresholds.append(0.5*np.mean(diff**2))
        y -= intercept
        x_train[dim*n_points_per_dim:(dim+1)*n_points_per_dim, dim] = torch.as_tensor(x)
        y_train[dim*n_points_per_dim:(dim+1)*n_points_per_dim] = torch.as_tensor(y)

    return x_train, y_train, np.mean(thresholds)



def test_performance(n_hidden=256, ds_type='2d_star_11', n_parallel=1000, n_epochs=1000000,
                     random_bias=False, act='relu', n_layers=1, device_number=0, version=0):
    print('Start time:', datetime.datetime.now())

    if n_layers > 1:
        name = f'{n_hidden}x{n_layers}-{n_parallel}-{random_bias}-{act}-v{version}'
        #name = f'{n_hidden}x{n_layers}-{ds_type}-{lr_factor}-{random_bias}-{act}-v{version}'
    else:
        name = f'{n_hidden}-{n_parallel}-{random_bias}-{act}-v{version}'
        #name = f'{n_hidden}-{ds_type}-{lr_factor}-{random_bias}-{act}-v{version}'
    if ds_type == 'double_cross':
        x_train, y_train = get_double_cross_dataset()
    elif ds_type == 'cross':
        x_train, y_train = get_cross_dataset()
    elif ds_type == 'random_with_line':
        x_train, y_train = get_random_with_line_ds()
    elif ds_type.startswith('multi_axis'):
        d = int(ds_type[ds_type.rfind('_')+1:])
        x_train, y_train, threshold = get_multi_axis_ds(d)
    elif ds_type == '2d_star_11':
        x_train, y_train = get_2d_star_dataset(k=11, dist=0.1)
    elif ds_type == '2d_star_7':
        x_train, y_train = get_2d_star_dataset(k=7, dist=0.1)
    else:
        x_train, y_train = get_standard_dataset()

    print(f'Running model for {n_epochs} epochs on dataset {ds_type}: {name}')
    base_dir = Path(get_results_path())
    file_dir = base_dir/ds_type/name
    file_path = file_dir/'model_trainer.p'
    if utils.existsFile(file_path):
        print('Loading existing model')
        mt = utils.deserialize(file_path)
        mt.to(get_device(device_number))
    else:
        print('Creating new model')
        mt = ModelTrainer(x_train, y_train, n_parallel=n_parallel,
                           hidden_sizes=[n_hidden] * n_layers, n_virtual_samples=n_hidden**2,
                           random_bias=random_bias, act=act, device_number=device_number, version=version)
    mt.train(n_epochs)
    mt.to('cpu')
    utils.serialize(file_path, mt)
    utils.serialize(file_dir/'config.p', dict(ds_type=ds_type, n_parallel=n_parallel, n_layers=n_layers,
                                              random_bias=random_bias, act=act, n_epochs=n_epochs, version=version,
                                              n_hidden=n_hidden))
    # todo: maybe save a config file that would make it easier to read off the desired quantities?
    print('Saved trained model')
    print('End time:', datetime.datetime.now())


def run_experiments_1():
    n_hidden_list = [2**k for k in range(7, 13)]
    for mult in range(1, 11):
        n_epochs = mult * 100000
        for n_hidden in n_hidden_list:
            test_performance(n_hidden=n_hidden, ds_type='2d_star_11', n_parallel=1000, n_epochs=n_epochs,
                             device_number=0)

def run_experiments_2():
    for mult in range(1, 11):
        n_epochs = mult * 100000
        for version in [0, 1]:
            # running n_parallel=1000 leads to memory overflow on an 8GB GPU,
            # therefore we run two versions with 500 parallel each
            test_performance(n_hidden=8192, ds_type='2d_star_11', n_parallel=500, n_epochs=n_epochs,
                             device_number=1, version=version)


def run_experiments_3():
    n_hidden_list = [2**k for k in range(4, 9)]
    for mult in range(1, 11):
        n_epochs = mult * 10000
        for n_hidden in n_hidden_list:
            test_performance(n_hidden=n_hidden, ds_type='default', n_parallel=1000,
                             n_epochs=n_epochs, device_number=0, version=0, n_layers=3)


def get_results_path():
    return Path(custom_paths.get_results_path())


if __name__ == '__main__':
    #train()
    #threshold = 2
    #x_train, y_train = get_cross_dataset()
    #x_train, y_train = get_standard_dataset()
    #x_train, y_train = get_random_with_line_ds()
    #x_train, y_train = get_double_cross_dataset()
    # x_train, y_train, threshold = get_multi_axis_ds(d=5)
    # print('threshold:', threshold)
    # import matplotlib.pyplot as plt
    # # plt.plot(x_train[:, -1], y_train, '.')
    # x_train, y_train = get_2d_star_dataset(k=7, dist=0.1)
    # plt.plot(x_train[:, 0], x_train[:, 1], '.')
    # plt.show()
    # train_models('256', x_train, y_train, n_parallel=100, n_epochs=1000000, hidden_sizes=[256], lr=0.1/256,
    #              random_bias=False, threshold=threshold-0.0001)
    # fire.Fire(test_performance)
    run_experiments_1()

