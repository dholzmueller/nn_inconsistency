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
import keras
import mc_training
import multiprocessing
import utils
import time
import tensorflow as tf

# Callback for a model.fit() in keras that checks whether some kinks crossed datapoints and stops training in this case
class CrossingCheckCallback(keras.callbacks.Callback):
    def __init__(self, min_abs_x):
        super(CrossingCheckCallback, self).__init__()
        self.stopped = False
        self.min_abs_x = min_abs_x

    def on_train_batch_end(self, batch, logs=None):
        first_layer = self.model.layers[0]
        weights = first_layer.get_weights()
        a = weights[0][0, :]
        b = weights[1]
        crossed = np.max(np.abs(b / a)) >= self.min_abs_x
        if crossed:
            # print('crossed on batch {}'.format(batch))
            self.model.stop_training = True
            self.stopped = True

# Callbac for a model.fit() in keras that uses a keras early stopping criterion
# but after a certain number of minibatches instead of after each epoch
class BatchEarlyStoppingCallback(keras.callbacks.Callback):
    def __init__(self, min_delta, patience, num_minibatches_per_check):
        super(BatchEarlyStoppingCallback, self).__init__()
        self.es_callback = keras.callbacks.EarlyStopping(patience=patience, min_delta=min_delta)
        self.batch_count = 0
        self.num_minibatches_per_check = num_minibatches_per_check

    def on_train_begin(self, logs=None):
        self.es_callback.on_train_begin(logs=logs)

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count % self.num_minibatches_per_check == 0:
            pseudo_epoch = self.batch_count // self.num_minibatches_per_check
            self.es_callback.on_epoch_end(epoch=pseudo_epoch, logs=logs)

# this class is the same as the keras.callbacks.EarlyStopping callback but without resetting for every fit() call
class WrappedEarlyStoppingCallback(keras.callbacks.Callback):
    def __init__(self, min_delta, patience):
        super(WrappedEarlyStoppingCallback, self).__init__()
        self.es_callback = keras.callbacks.EarlyStopping(patience=patience, min_delta=min_delta)
        self.es_callback.on_train_begin()
        self.epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_loss') is None:
            return  # probably already crossed
        self.epoch_count += 1
        # print(epoch, self.epoch_count, logs)
        self.es_callback.model = self.model
        self.es_callback.on_epoch_end(self.epoch_count, logs=logs)

    def has_stopped(self):
        return self.es_callback.stopped_epoch != 0

# runs keras training with a single neural network
def run_single(n_hidden, n_samples, x, y, lr, batch_size=16, num_minibatches_per_check=1000, patience=10, min_delta=1e-8):
    # print('.', end='', flush=True)
    # print('run_single')
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_hidden, input_dim=1, activation='relu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(1, kernel_initializer='he_normal'))
    model.compile(optimizer=keras.optimizers.SGD(lr=0.5*lr), loss='mse')  # mse does not have factor 0.5

    # es_callback = keras.callbacks.EarlyStopping(patience=patience, min_delta=min_delta)
    # es_callback = BatchEarlyStoppingCallback(patience=patience, min_delta=min_delta, num_minibatches_per_check=num_minibatches_per_check)
    es_callback = WrappedEarlyStoppingCallback(patience=patience, min_delta=2*min_delta)  # mse does not have factor 0.5
    cc_callback = CrossingCheckCallback(min_abs_x=np.min(np.abs(x)))
    val_sample_weights = np.random.multinomial(n_samples, [1./len(x)] * len(x)) / n_samples
    x_weights = np.random.multinomial(n_samples, [1./len(x)] * len(x)) / n_samples

    while not (es_callback.has_stopped() or cc_callback.stopped):
        indices = np.random.choice(len(x), size=batch_size*num_minibatches_per_check, p=x_weights)
        model.fit(x[indices], y[indices], epochs=1, batch_size=batch_size, callbacks=[cc_callback, es_callback],
              validation_data=(x, y, val_sample_weights), verbose=False)

    # print('')
    # print(cc_callback.stopped)
    # print('')
    # print('')

    return cc_callback.stopped

# worker function that runs keras training for certain parameters
def compute_mc(n_parallel=1000, offset_str='0'):
    x = np.array([-1.0, -2.0, -3.0, 1.0, 2.0, 3.0])
    y = np.array([-1.0, 2.0, -1.0, 1.0, -2.0, 1.0]) + float(offset_str)

    for k in range(4, 13):
        n_hidden = 2**k
        print('Running MC Experiment for n_hidden = {}'.format(n_hidden))
        results = [run_single(n_hidden, n_hidden**2, x, y, lr=1e-2/n_hidden) for i in range(n_parallel)]
        ratio = np.count_nonzero(results) / len(results)
        print('Resulting probability estimate: {}'.format(ratio))

# class that can be serialized to save results of keras training
class MCResult(object):
    def __init__(self, num_crossed, num_total):
        self.num_crossed = num_crossed
        self.num_total = num_total

# Class that can be used as a worker for a thread pool
class KerasMCRunner(object):
    def __init__(self, offset_str, base_dir):
        self.offset_str = offset_str
        self.base_dir = base_dir

    def __call__(self, config):
        start_time = time.time()
        np.random.seed(config.seed)
        tf.set_random_seed(config.seed)
        print('Starting id = {}, n_parallel = {}, n_hidden = {}'.format(config.index, config.n_parallel, config.n_hidden))
        x = np.array([-1.0, -2.0, -3.0, 1.0, 2.0, 3.0])
        y = np.array([-1.0, 2.0, -1.0, 1.0, -2.0, 1.0]) + float(self.offset_str)
        results = [run_single(config.n_hidden, config.n_samples, x, y, lr=1e-2/config.n_hidden) for i in range(config.n_parallel)]

        try:
            summary = MCResult(num_crossed=np.count_nonzero(results), num_total=len(results))

            print('id = {}, n_parallel = {}, n_hidden = {}: {}/{}'.format(config.index, config.n_parallel, config.n_hidden, summary.num_crossed, summary.num_total))

            target_folder = self.base_dir + 'mc-data-{}/id-{}_hidden-{}_parallel-{}_time-{}/'.format(self.offset_str, config.index,
                                                                                                     config.n_hidden,
                                                                                                     config.n_parallel,
                                                                                                     int(start_time * 1000))
            utils.serialize(target_folder + 'result.p', summary)
            utils.serialize(target_folder + 'config.p', config)
        except Exception as e:
            print(e)

# returns task configurations for thread pool workers
def get_param_combinations():
    param_combinations = []

    np.random.seed(1234567890)

    np.random.randn(1000)  # generate some random samples for warming up

    min_log = 4  # 16
    max_log = 11  # 2048
    num_mc_runs = 1000
    index = 0
    for k in range(min_log, max_log + 1):
        n_hidden = 2 ** k
        n_samples = n_hidden ** 2
        n_parallel = 10

        num_remaining_parallel = num_mc_runs
        while num_remaining_parallel > 0:
            num_parallel_here = min(n_parallel, num_remaining_parallel)
            seed = np.random.randint(2**30)
            param_combinations.append(mc_training.MCConfiguration(num_parallel_here, n_hidden, n_samples, seed, index))
            num_remaining_parallel -= num_parallel_here
            index += 1

    return param_combinations

# executes all tasks in a thread pool
def execute_mc(offset_str, base_dir='./mc-data/'):
    param_combinations = get_param_combinations()
    num_processes = max(1, multiprocessing.cpu_count()//2)
    pool = multiprocessing.Pool(processes=num_processes)
    utils.ensureDir(base_dir + 'mc-data-{}/'.format(offset_str))
    pool.map(KerasMCRunner(offset_str, base_dir), param_combinations, chunksize=1)
    pool.terminate()
    pool.join()


def main():
    # compute_mc(n_parallel=100)
    for offset_str in ['0', '0.01', '0.1']:
        execute_mc(offset_str, './mc-data-sgd-keras/')


if __name__ == '__main__':
    main()
