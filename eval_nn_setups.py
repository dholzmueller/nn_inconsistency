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

from run_nn_setups import *
import math

import matplotlib
#matplotlib.use('Agg')
matplotlib.use('pdf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb}'
})
import matplotlib.pyplot as plt


def get_alg_name(results):
    trainer = results['trainer']
    return f'{trainer.init_param}_{trainer.opt}_{trainer.bias_init_mode}-{trainer.bias_init_gain:g}'


def get_latex_name(results):
    trainer = results['trainer']
    param_name = 'He' if trainer.init_param == 'kaiming' else 'NTK'
    opt_name = 'Adam' if trainer.opt == 'adam' else 'SGDM' if trainer.opt == 'gd-mom' else 'SGD'
    bim = trainer.bias_init_mode
    big = trainer.bias_init_gain
    if bim == 'he+1':
        bias_name = 'KinkPoint'
    elif bim == 'he+5':
        bias_name = 'He+5'
    elif bim == 'kink_uniform':
        bias_name = r'KinkUnif'
    elif bim == 'zero':
        bias_name = 'Zero'
    elif bim == 'uniform':
        bias_name = r'$\hat U(' + f'{big:g})$'
        #bias_name = r'$\mathcal{U}(\sigma=' + f'{big:g})$'
    elif bim == 'normal':
        bias_name = r'$N(' + f'{big:g})$'
        #bias_name = r'$\mathcal{N}(\sigma=' + f'{big:g})$'
    elif bim == 'pos-unif':
        bias_name = r'$\hat U_+(' + f'{big:g})$'
    elif bim == 'neg-unif':
        bias_name = r'$\hat U_-(' + f'{big:g})$'
    elif bim == 'kink-unif':
        bias_name = r'$U_k(' + f'{big:g})$'
    elif bim == 'kink-neg-unif':
        bias_name = r'$U_{k-}(' + f'{big:g})$'
    elif bim == 'kink-neg-point':
        bias_name = '$X_{k-}$'
    elif bim == 'unif':
        bias_name = r'$U(' + f'{big:g})$'
    elif bim == 'unif-neg':
        bias_name = r'$U_-(' + f'{big:g})$'
    elif bim == 'unif-pos':
        bias_name = r'$U_+(' + f'{big:g})$'
    elif bim == 'pytorch':
        bias_name = 'PyTorch'
    else:
        raise RuntimeError(f'Unknown bias_init_mode {bim}')
    return '/'.join([param_name, opt_name, bias_name])


def pretty_table_str(str_table):
    max_lens = [np.max([len(row[i]) for row in str_table]) for i in range(len(str_table[0]))]
    whole_str = ''
    for row in str_table:
        for i, entry in enumerate(row):
            whole_str += entry + (' ' * (max_lens[i] - len(entry)))
        whole_str += '\n'
    return whole_str[:-1]  # remove last newline


def argsort(lst):
    # from https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
    return sorted(range(len(lst)), key=lst.__getitem__)


class ResultsTable:
    def __init__(self, best_results):
        self.rad_ds_dims = [2**k for k in range(7)]
        self.rad_ds_names = [f'rad-distr-{dim}' for dim in self.rad_ds_dims]
        potential_ds_names = ['ex-distr'] + self.rad_ds_names
        self.ds_names = list(best_results.keys())
        self.ds_names = [name for name in potential_ds_names if name in self.ds_names]
        self.ds_to_latex_names = {f'rad-distr-{dim}': r'$P^{\mathrm{data}}_{' + str(dim) + r'}$'
                                  for dim in self.rad_ds_dims}
        self.ds_to_latex_names['ex-distr'] = r'$P^{\mathrm{data}}_{\mathrm{ex}}$'

        names = list(set([(get_alg_name(r), get_latex_name(r)) for i in range(len(self.ds_names))
                          for r in best_results[self.ds_names[i]]]))
        self.alg_names = [name[0] for name in names]
        self.alg_to_latex_names = {name[0]: name[1] for name in names}
        self.alg_names.sort()
        self.alg_ds_results = {alg_name: [None] * len(self.ds_names) for alg_name in self.alg_names}

        for i in range(len(self.ds_names)):
            for r in best_results[self.ds_names[i]]:
                self.alg_ds_results[get_alg_name(r)][i] = r

    def print(self):
        str_table = [['Configuration: '] + [ds_name + ' ' for ds_name in self.ds_names]]
        for alg_name, ds_results in self.alg_ds_results.items():
            row = [alg_name + ' ']
            for r in ds_results:
                if r is not None:
                    test_rmse_mean = r["results"]["best_test_rmse"]
                    test_rmse_std = r["results"]["best_test_rmse_std"] / np.sqrt(r["trainer"].n_parallel - 1)
                    row.append(f'{test_rmse_mean:5.4f}+-{test_rmse_std:5.4f}  ')
                else:
                    row.append(' ')
            str_table.append(row)
        print(pretty_table_str(str_table))

    def save_latex(self, path, alg_names=None):
        begin_tabular = r'\begin{tabular}{' + ''.join(['c'] * (len(self.ds_names) + 1)) + r'}'
        end_tabular = r'\end{tabular}'
        header_row = ' & '.join([r'Param/Opt/Bias-init'] + [self.ds_to_latex_names[ds_name]
                                                                for ds_name in self.ds_names]) + r'\\' + '\n\\hline'
        content_rows = []
        rmses = [[r['results']['best_test_rmse'] for r in ds_results] for ds_results in self.alg_ds_results.values()]
        log_max_rmses = np.log(np.max(rmses, axis=0))
        log_min_rmses = np.log(np.min(rmses, axis=0))

        if alg_names is None:
            items = self.alg_ds_results.items()
        else:
            items = [(alg_name, self.alg_ds_results[alg_name]) for alg_name in alg_names]

        for alg_name, ds_results in items:
            row = [self.alg_to_latex_names[alg_name]]
            for i, r in enumerate(ds_results):
                if r is not None:
                    factor = 10000
                    rmse = r["results"]["best_test_rmse"]
                    log_scale = (np.log(rmse) - log_min_rmses[i]) / (log_max_rmses[i] - log_min_rmses[i])
                    test_rmse_mean = int(round(factor * r["results"]["best_test_rmse"]))
                    test_rmse_std = int(round(factor * r["results"]["best_test_rmse_std"]
                                              / np.sqrt(r["trainer"].n_parallel - 1)))
                    row.append(r'\cellcolor{red!' + str(int(100*log_scale)) + r'} '
                               + f'${test_rmse_mean} \\pm {test_rmse_std}$')
                    # row.append(f'${test_rmse_mean} \\pm {test_rmse_std}$')
                    # row.append(f'${test_rmse_mean:4.3f} \\pm {test_rmse_std:4.3f}$')
                else:
                    row.append('')
            content_rows.append(' & '.join(row) + r'\\')

        latex_str = '\n'.join([begin_tabular, header_row] + content_rows + [end_tabular])
        utils.writeToFile(path, latex_str)

    def plot_vs_dim(self, path, alg_names):
        figsize = (4.4, 3.5)
        plt.figure(figsize=figsize)
        rmses = [[r['results']['best_test_rmse'] for r in self.alg_ds_results[alg_name]] for alg_name in alg_names]
        #base_rmses = np.max(rmses, axis=0)
        base_rmses = rmses[0]
        # todo: create a color list?
        ds_idxs = [self.ds_names.index(rad_ds_name) for rad_ds_name in self.rad_ds_names]
        for alg_name in alg_names:
            ds_results = self.alg_ds_results[alg_name]
            means = [ds_results[ds_idx]['results']['best_test_rmse'] / base_rmses[ds_idx] for ds_idx in ds_idxs]
            stds = [ds_results[ds_idx]['results']['best_test_rmse_std'] / base_rmses[ds_idx]
                    / np.sqrt(ds_results[ds_idx]['trainer'].n_parallel - 1) for ds_idx in ds_idxs]
            plt.loglog(self.rad_ds_dims, means, label=self.alg_to_latex_names[alg_name])
            plt.fill_between(self.rad_ds_dims, [means[i] - 2*stds[i] for i in range(len(means))],
                             [means[i] + 2*stds[i] for i in range(len(means))], alpha=0.3)
        plt.xlabel(r'Input dimension $d$')
        plt.ylabel(r'Relative Mean RMSE')
        plt.xticks(self.rad_ds_dims, [str(dim) for dim in self.rad_ds_dims])
        #plt.minorticks_off()
        plt.tick_params('x', which='minor', bottom=False)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def plot_grid_vs_dim(self, path, bias_inits):
        figsize=(5.5, 5.5)
        fig, axs = plt.subplots(3, 2, figsize=figsize, sharex=True, sharey=True)
        for i, opt in enumerate(['gd', 'gd-mom', 'adam']):
            for j, param in enumerate(['kaiming', 'ntk']):
                if i == 0:
                    axs[i, j].set_title('He' if param == 'kaiming' else 'NTK')
                opt_names = ['SGD', 'SGDM', 'Adam']
                axs[i, j].set(xlabel=r'Input dimension $d$', ylabel=opt_names[i])

                alg_names = [f'{param}_{opt}_{bias_init}' for bias_init in bias_inits]
                rmses = [[r['results']['best_test_rmse'] for r in self.alg_ds_results[alg_name]] for alg_name in
                         alg_names]
                base_rmses = rmses[0]
                ds_idxs = [self.ds_names.index(rad_ds_name) for rad_ds_name in self.rad_ds_names]
                for alg_name in alg_names:
                    ds_results = self.alg_ds_results[alg_name]
                    means = [ds_results[ds_idx]['results']['best_test_rmse'] / base_rmses[ds_idx] for ds_idx in ds_idxs]
                    stds = [ds_results[ds_idx]['results']['best_test_rmse_std'] / base_rmses[ds_idx]
                            / np.sqrt(ds_results[ds_idx]['trainer'].n_parallel - 1) for ds_idx in ds_idxs]
                    axs[i,j].loglog(self.rad_ds_dims, means, label=self.alg_to_latex_names[alg_name].split('/')[-1])
                    axs[i,j].fill_between(self.rad_ds_dims, [means[i] - 2 * stds[i] for i in range(len(means))],
                                     [means[i] + 2 * stds[i] for i in range(len(means))], alpha=0.3)

        for ax in axs.flat:
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            ax.label_outer()
            ax.grid(True)
            ax.set_xticks(self.rad_ds_dims)
            ax.set_xticklabels([str(dim) for dim in self.rad_ds_dims])
            ax.minorticks_off()
            ax.tick_params('x', which='minor', bottom=False)

        fig.legend(*axs[0,0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, -0.0),
                  fancybox=True, shadow=True, ncol=3)

        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

    def plot_grid_vs_dim_2(self, path, bias_inits):
        figsize=(5.5, 5.5)
        fig, axs = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
        opt_titles = ['SGD', 'SGDM', 'Adam']
        param_names = ['He\nRel. mean test RMSE', 'NTK\nRel. mean test RMSE']
        for i, param in enumerate(['kaiming', 'ntk']):
            for j, opt in enumerate(['gd', 'gd-mom', 'adam']):
                if i == 0:
                    axs[i, j].set_title(opt_titles[j])
                axs[i, j].set(xlabel=r'Input dimension $d$', ylabel=param_names[i])

                alg_names = [f'{param}_{opt}_{bias_init}' for bias_init in bias_inits]
                rmses = [[r['results']['best_test_rmse'] for r in self.alg_ds_results[alg_name]] for alg_name in
                         alg_names]
                base_rmses = rmses[0]
                ds_idxs = [self.ds_names.index(rad_ds_name) for rad_ds_name in self.rad_ds_names]
                for alg_name in alg_names:
                    ds_results = self.alg_ds_results[alg_name]
                    means = [ds_results[ds_idx]['results']['best_test_rmse'] / base_rmses[ds_idx] for ds_idx in ds_idxs]
                    stds = [ds_results[ds_idx]['results']['best_test_rmse_std'] / base_rmses[ds_idx]
                            / np.sqrt(ds_results[ds_idx]['trainer'].n_parallel - 1) for ds_idx in ds_idxs]
                    axs[i,j].loglog(self.rad_ds_dims, means, label=self.alg_to_latex_names[alg_name].split('/')[-1])
                    axs[i,j].fill_between(self.rad_ds_dims, [means[k] - 2 * stds[k] for k in range(len(means))],
                                     [means[k] + 2 * stds[k] for k in range(len(means))], alpha=0.3)

        for ax in axs.flat:
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            ax.label_outer()
            ax.grid(True)
            ax.set_xticks(self.rad_ds_dims)
            ax.set_xticklabels([str(dim) for dim in self.rad_ds_dims])
            #ax.minorticks_off()
            ax.tick_params('x', which='minor', bottom=False)

        fig.legend(*axs[0,0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, -0.0), ncol=3)

        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def load():
        base_path = Path(custom_paths.get_results_path()) / 'nn_comparison'
        best_results = {}

        for results_dir in base_path.iterdir():
            if not results_dir.is_dir():
                continue
            valid_dir_results = []
            for results_file in results_dir.iterdir():
                results = utils.deserialize(results_file)
                if results['results'] is not None:
                    valid_dir_results.append(results)

            ds_name = results_dir.stem.split('_')[2]

            if len(valid_dir_results) > 0:
                best_idx = np.argmin([r['results']['best_valid_rmse'] for r in valid_dir_results])
                if ds_name in best_results:
                    best_results[ds_name].append(valid_dir_results[best_idx])
                else:
                    best_results[ds_name] = [valid_dir_results[best_idx]]
            else:
                print(f'No valid results for {results_dir.stem}')

        return ResultsTable(best_results)


class ExDistrPlotCallback:
    def __init__(self):
        self.x_linspace = None
        self.Xtrain = None
        self.ytrain = None
        self.y_pred = None

    def __call__(self, model, Xtrain, ytrain, mean, std, ymean, ystd, best_valid_mses):
        perm = torch.argsort(best_valid_mses)
        med_idx = perm[len(perm)//2]
        self.x_linspace = torch.linspace(-1.4, 1.4, 500, device=Xtrain.device)
        self.Xtrain = Xtrain[med_idx, :, 0] * std[med_idx, 0] + mean[med_idx, 0]
        self.ytrain = ytrain[med_idx, :] * ystd[med_idx] + ymean[med_idx]
        x = (self.x_linspace[None, :, None] - mean[:, None, :]) / std[:, None, :]
        with torch.no_grad():
            self.y_pred = model(x)[med_idx, :] * ystd[med_idx] + ymean[med_idx]


def plot_ex_distr_results(path, alg_names):
    base_path = Path(custom_paths.get_results_path()) / 'nn_comparison'
    plt.figure(figsize=(3, 2))
    for i, alg_name in enumerate(alg_names):
        parts = alg_name.split('_')
        parts.insert(2, 'ex-distr')
        dir_name = '_'.join(parts)
        results = [utils.deserialize(file) for file in (base_path / dir_name).iterdir()]
        results = [r for r in results if r['results'] is not None]
        best_idx = np.argmin([r['results']['best_valid_rmse'] for r in results])
        trainer = results[best_idx]['trainer']
        kwargs = trainer.__dict__
        kwargs['n_parallel'] = 11
        kwargs['device'] = 'cpu'
        kwargs['seed'] = 5
        new_trainer = SimpleParallelTrainer(**kwargs)
        cb = ExDistrPlotCallback()
        new_trainer.fit(verbose=True, end_training_callback=cb, use_same_ds=True)
        # new_trainer = SimpleParallelTrainer(n_parallel=1, n_train=trainer.n_train, n_valid=trainer.n_valid,
        #                                     n_test=trainer.n_test, data_distribution=trainer.data_distribution,
        #                                     init_param=trainer.init_param, bias_init_gain=trainer.bias_init_gain)
        if i==0:
            plt.plot(cb.Xtrain.cpu(), cb.ytrain.cpu(), 'x', color='k', markersize=4)
        plt.plot(cb.x_linspace.cpu(), cb.y_pred.cpu(), label=get_latex_name(results[best_idx]))
        pass
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def interleave_arrays(arr1, arr2):
    return np.stack([arr1, arr2], axis=1).flatten()


def get_xy_cdf(arr, xmin, xmax):
    x = np.sort(arr)
    x = interleave_arrays(np.hstack([[xmin], x]), np.hstack([x, [xmax]]))
    y = np.linspace(0.0, 1.0, len(arr)+1)
    y = interleave_arrays(y, y)
    return x, y


def plot_kink_distributions(path):
    plt.figure(figsize=(4.5, 3))
    plt.plot([0.0, np.sqrt(3)], [0.0, 1.0], 'k', label=r'$U_{k-}(1)$')
    plt.xlim(0.0, 4.0)
    plt.ylim(0.0, 1.0)
    n = 50001
    for d in [1, 2, 4, 8, 16, 32, 64]:
        np.random.seed(0)
        biases = np.random.uniform(low=0.0, high=np.sqrt(6), size=(n,))
        a = (np.sqrt(2/d)) * np.random.normal(size=(n,d))
        kinks = biases / np.linalg.norm(a, axis=1)
        kinks = np.sort(kinks)
        kinks = kinks[::100]  # subsample such that plot doesn't get too large
        plt.plot(kinks, np.linspace(0.0, 1.0, len(kinks)), label=r'$U_-(1), ' + f'd={d}$')

    plt.xlabel('Distance of kink hyperplane to origin')
    plt.ylabel('CDF')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    table = ResultsTable.load()
    table.print()

    plot_kink_distributions('results/kink_distrs.pdf')

    plot_ex_distr_results('results/nn_ex.pdf', ['kaiming_gd_zero-0', 'kaiming_gd_kink-neg-unif-1',
                                                'kaiming_adam_kink-neg-unif-1'])
    plot_ex_distr_results('results/nn_ex_2.pdf', ['kaiming_gd_zero-0', 'kaiming_gd_kink-neg-unif-1',
                                                  'ntk_gd_zero-0', 'ntk_gd_kink-neg-unif-1'])

    param_opts = ['kaiming_gd', 'kaiming_gd-mom', 'kaiming_adam', 'ntk_gd', 'ntk_gd-mom', 'ntk_adam']
    bias_inits = ['zero-0', 'pytorch-1', 'unif-1', 'unif-pos-1', 'unif-neg-1', 'kink-neg-unif-1']
    extended_bias_inits = bias_inits + ['kink-neg-point-0', 'he+5-0']
    table.save_latex('results/nn_results_latex.txt', alg_names=[f'{param_opt}_{bias_init}' for param_opt in param_opts
                                                                    for bias_init in extended_bias_inits])
    #table.plot_grid_vs_dim('results/nn_grid.pdf', bias_inits)
    table.plot_grid_vs_dim_2('results/nn_grid_2.pdf', bias_inits)
    for param_opt in param_opts:
        table.plot_vs_dim(f'results/nn_{param_opt}.pdf', [f'{param_opt}_{bias_init}' for bias_init in bias_inits])
    table.plot_vs_dim(f'results/nn_plot_bias.pdf', [f'{param}_{opt}_{bias_init}' for param in ['kaiming', 'ntk']
                                                    for opt in ['gd', 'adam']
                                                    for bias_init in ['zero-0', 'kink-neg-unif-1']])
    table.plot_vs_dim(f'results/nn_plot_zero.pdf', [f'{param}_{opt}_zero-0' for param in ['kaiming', 'ntk']
                                                    for opt in ['gd', 'gd-mom', 'adam']])

    pass
