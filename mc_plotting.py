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
from collections import defaultdict
from mc_training import CheckingNN, TrainingStatus, MCConfiguration  # need this for deserialize


# find subfolders of a data folder where identical experiments with different timestamps are deduplicated
# (only the one with the newest timestamp) is used
def get_subfolders_without_duplicates(offset_subfolder):
    subfolders = utils.getSubfolders(offset_subfolder)
    subfolders_without_timestamp = set([sf[:sf.rfind('-')] for sf in subfolders])
    cleaned_subfolders = []
    for prefix in subfolders_without_timestamp:
        matching = [sf for sf in subfolders if sf.startswith(prefix)]
        timestamps = [int(sf[sf.rfind('-')+1:]) for sf in matching]
        sort_perm = np.argsort(timestamps)
        matching_sorted = [matching[sort_perm[i]] for i in range(len(timestamps))]
        cleaned_subfolders.append(matching_sorted[-1])
    return cleaned_subfolders

# load data into a dictionary that for each shift / offset $\Delta$ contains all the results and configurations
# from the corresponding folder
def load_data(folder, from_keras=False):
    data_list_dict = {}
    offset_subfolder_names = utils.getSubfolderNames(folder)
    for offset_subfolder_name in offset_subfolder_names:
        offset_subfolder = folder + offset_subfolder_name + '/'
        offset_str = offset_subfolder_name[len('mc-data-'):]

        subfolders = get_subfolders_without_duplicates(offset_subfolder)
        data_list = []
        for subfolder in subfolders:
            try:
                if from_keras:
                    from mc_sgd_keras import MCResult  # need this for deserialize
                    data_list.append((utils.deserialize(subfolder + '/result.p'), utils.deserialize(subfolder + '/config.p')))
                else:
                    data_list.append((utils.deserialize(subfolder + '/net.p'), utils.deserialize(subfolder + '/config.p')))
            except:
                print('nothing found in folder ' + subfolder)
                pass

        data_list_dict[offset_str] = data_list

    return data_list_dict

# sums the results from different workers together
def get_statistics_dicts(data_list_dict, from_keras=False):
    statistics_dicts = {}
    for offset_str, data_list in data_list_dict.items():
        statistics_dict = defaultdict(lambda: np.zeros(6, dtype=np.int))

        if from_keras:
            for result, config in data_list:
                statistics_dict[config.n_hidden] += np.array([0, result.num_crossed, 0, 0, 0, result.num_total - result.num_crossed])
        else:
            for net, config in data_list:
                statistics_dict[net.num_hidden] += np.array([net.get_num_unfinished(), net.get_num_crossed(),
                                                            net.get_num_locally_converged(), net.get_num_a_degenerate(),
                                                            net.get_num_x_degenerate(), net.get_num_early_stopped()])
        statistics_dicts[offset_str] = statistics_dict

    return statistics_dicts

# prints a summary for each number of hidden neurons
def print_statistics(statistics_dicts):
    offset_strs = list(statistics_dicts.keys())
    offset_strs.sort()
    for offset_str in offset_strs:
        statistics_dict = statistics_dicts[offset_str]
        print('Results for offset {}:'.format(offset_str))
        keys = statistics_dict.keys()

        for n_hidden in np.sort(np.array(list(keys))):
            freq = statistics_dict[n_hidden]
            print('{} hidden: [{} unfinished, {} crossed, {} locally converged, {} a-degenerate, {} x-degenerate, {} early stopped]'.format(
                n_hidden, freq[0], freq[1], freq[2], freq[3], freq[4], freq[5]
            ))

        print('')

# creates a LaTeX plot of the kink crossing probabilities
def plot_statistics(statistics_dicts, out_filename):
    tex_str = ''
    offset_strs = list(statistics_dicts.keys())
    offset_strs.sort()
    for offset_str in offset_strs:
        statistics_dict = statistics_dicts[offset_str]
        keys = statistics_dict.keys()

        # all but locally converged (2) and early stopped (5) count as possibly crossed
        lines = ['({}, {})'.format(n_hidden, 1.0 - ((statistics_dict[n_hidden][2] + statistics_dict[n_hidden][5]) / np.sum(statistics_dict[n_hidden])))
                 for n_hidden in np.sort(np.array(list(keys)))]
        tex_str += '\\addplot coordinates {\n' + '\n'.join(lines) + '};\n\\addlegendentry{$\\Delta = ' + offset_str + '$}\n'

    print('LaTeX code excerpt:')
    print(tex_str)

    tex_str = utils.readFromFile('tex_head.txt') + tex_str + utils.readFromFile('tex_tail.txt')
    utils.writeToFile(out_filename, tex_str)


def plot_main():
    base_dir = './mc-data/'
    data_list_dict = load_data(base_dir, from_keras=False)
    statistics_dicts = get_statistics_dicts(data_list_dict, from_keras=False)
    print_statistics(statistics_dicts)
    plot_statistics(statistics_dicts, out_filename=base_dir+'plot.tex')

    # base_dir = './mc-data-sgd-keras/'
    # data_list_dict = load_data(base_dir, from_keras=True)
    # statistics_dicts = get_statistics_dicts(data_list_dict, from_keras=True)
    # print_statistics(statistics_dicts)
    # plot_statistics(statistics_dicts, out_filename=base_dir + 'plot.tex')


if __name__ == '__main__':
    plot_main()
