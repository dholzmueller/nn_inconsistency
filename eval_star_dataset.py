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

from train_star_dataset import *


def print_aggregate_results(ds_type):
    base_path = Path(get_results_path()) / ds_type
    configs = []
    mt_results = []
    for path in base_path.iterdir():
        mt_path = path / 'model_trainer.p'
        config_path = path / 'config.p'
        if not utils.existsFile(mt_path) or not utils.existsFile(config_path):
            continue
        mt = utils.deserialize(mt_path)
        config = utils.deserialize(config_path)
        configs.append(config)
        mt_results.append(dict(kink_fraction=(mt.kink_first_crossed != -1).float().mean().item(),
                               mse_fraction=(mt.mse_first_crossed != -1).float().mean().item()))

    n_hidden_list = list(set([c['n_hidden'] for c in configs]))
    n_hidden_list.sort()

    for name, attr in [('Fraction of kink crosses', 'kink_fraction'), ('Fraction of mse crosses', 'mse_fraction')]:
        print(f'{name}:')
        for n_hidden in n_hidden_list:
            idxs = [i for i, cfg in enumerate(configs) if cfg['n_hidden'] == n_hidden]
            fraction = sum([mt_results[i][attr] * configs[i]['n_parallel'] for i in idxs]) \
                       / sum([configs[i]['n_parallel'] for i in idxs])
            print(f'({n_hidden}, {fraction:g})')
        print()



if __name__ == '__main__':
    print_aggregate_results('2d_star_11')
