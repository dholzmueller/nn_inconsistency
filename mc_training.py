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

from enum import Enum
import numpy as np
import time
from TrainingSetup import TrainingSetup
import utils
import multiprocessing


class TrainingStatus(Enum):
    UNFINISHED = 0  # neural network has not been stopped yet
    X_DEGENERATE = 1  # all x points have the same sign
    A_DEGENERATE = 2  # all a_i values have the same sign
    CROSSED = 3   # a kink crossed a datapoint
    LOCALLY_CONVERGED = 4   # a kink will never cross a datapoint
    EARLY_STOPPED = 5  # an early stopping criterion was satisfied


def expand(x, dims):
    for d in dims:
        x = np.expand_dims(x, d)
    return x


def get_standard_dataset():
    return np.array([-3., -2., -1., 1., 2., 3.]), np.array([-1.0, 2.0, -1.0, 1.0, -2.0, 1.0])


# contains temporary variables for training that should not be saved
class TrainingVariables(object):
    def __init__(self, num_parallel, num_hidden, x, y, x_weights, lrs, use_adam=False):
        self.num_parallel = num_parallel
        self.batch_size = len(x)
        self.num_hidden = num_hidden

        # temporary variables for intermediate results during forward- and backpropagation
        self.relu_result = np.zeros(shape=(self.num_parallel, self.batch_size, self.num_hidden))
        self.inactive = np.zeros(shape=(self.num_parallel, self.batch_size, self.num_hidden), dtype=np.bool)
        self.wr = np.copy(self.relu_result)
        self.f_minus_y_times_lr = np.zeros(shape=(self.num_parallel, self.batch_size, 1))
        self.step_c = np.zeros(shape=(self.num_parallel, 1, 1))
        self.step = np.zeros(shape=(self.num_parallel, 1, self.num_hidden))
        self.fmy_lr_w = np.copy(self.wr)

        # for early stopping
        self.wait_times = np.zeros(num_parallel, dtype=np.int)
        self.best_validation_losses = np.inf * np.ones(num_parallel)

        self.X = expand(x, [0, 2])
        self.Y = expand(y, [0, 2])
        self.X_weights = np.expand_dims(x_weights, axis=2)
        self.neg_lrs = -expand(lrs, [1, 2])
        self.use_adam = use_adam

        if use_adam:
            self.beta_1 = 0.9
            self.beta_2 = 0.999
            self.eps = 1e-8
            self.beta_1_power = 1.0
            self.beta_2_power = 1.0
            self.grad_acc_a = np.zeros(shape=(self.num_parallel, 1, self.num_hidden))
            self.grad_acc_b = np.zeros(shape=(self.num_parallel, 1, self.num_hidden))
            self.grad_acc_c = np.zeros(shape=(self.num_parallel, 1, 1))
            self.grad_acc_w = np.zeros(shape=(self.num_parallel, 1, self.num_hidden))
            self.sq_grad_acc_a = np.zeros(shape=(self.num_parallel, 1, self.num_hidden))
            self.sq_grad_acc_b = np.zeros(shape=(self.num_parallel, 1, self.num_hidden))
            self.sq_grad_acc_c = np.zeros(shape=(self.num_parallel, 1, 1))
            self.sq_grad_acc_w = np.zeros(shape=(self.num_parallel, 1, self.num_hidden))


# stops all neural nets where a kink has potentially crossed a datapoint, i.e. |x_{kink}| >= min_j |x_j|
# where x_j are the datapoints
class CrossingStoppingCriterion(object):
    def check(self, net, vars):
        crossed_point_indices = net.max_kink_movement[net.orig_indices] >= net.min_abs_x
        net.terminate_instances(vars, [crossed_point_indices], [TrainingStatus.CROSSED])

# stops all neural nets if during the last $patience$ checks, the validation loss did not improve by at least $min_delta$
class EarlyStoppingCriterion(object):
    def __init__(self, min_delta, patience):
        self.min_delta = min_delta
        self.patience = patience

    def check(self, net, vars):
        vars.wait_times += 1
        val_losses = net.compute_validation_losses()
        improved_indices = val_losses < vars.best_validation_losses - self.min_delta
        vars.best_validation_losses[improved_indices] = val_losses[improved_indices]
        vars.wait_times[improved_indices] = 0

        early_stopping_indices = vars.wait_times >= self.patience

        net.terminate_instances(vars, [early_stopping_indices],
                                 [TrainingStatus.EARLY_STOPPED])

# stops all neural nets where theory ensures that no kink will ever cross a datapoint (using gradient descent)
class SufficientStoppingCriterion(object):
    def check(self, net, vars):
        locally_converged_indices = np.array([
            net.train_setups[net.orig_indices[i]].check_convergence(net.a[i, 0, :], net.b[i, 0, :],
                                                                    net.c[i, 0, 0], net.w[i, 0, :],
                                                                    net.lrs[net.orig_indices[i]])
            for i in range(net.a.shape[0])])

        # indices where all neurons point to the same side (i. e. the same sign)
        a_degenerate_indices = np.logical_or(np.all(net.a[:, 0, :] > 0, axis=1), np.all(net.a[:, 0, :] < 0, axis=1))

        x_degenerate_indices = np.logical_or(
            np.count_nonzero(net.x_weights[net.orig_indices, :][:, net.x > 0], axis=1) <= 1,
            np.count_nonzero(net.x_weights[net.orig_indices, :][:, net.x < 0], axis=1) <= 1)

        net.terminate_instances(vars, [locally_converged_indices, a_degenerate_indices,
                                       x_degenerate_indices],
                                [TrainingStatus.LOCALLY_CONVERGED, TrainingStatus.A_DEGENERATE,
                                 TrainingStatus.X_DEGENERATE])

# Neural net class which uses numpy to evaluate multiple
# two-layer relu networks with one input and one output neuron in parallel.
# It can also check whether certain networks satisfy a stopping criterion and remove them from the training process.
class CheckingNN(object):
    def __init__(self, initial_weights, train_setups, lrs, val_x_weights=None):
        (initial_a, initial_b, initial_c, initial_w) = initial_weights

        # dimensions: num_parallel_networks x batch_size x (num_hidden or 1)
        # Important: Stopped networks are removed from some of the arrays in the terminate_instances() method
        # while they are kept in others (to have some statistics about training afterwards)
        # Of course, these arrays have to be indexed differently. The link is the array orig_indices, which initially
        # contains [0, 1, ..., num_parallel_networks - 1] and from which the stopped networks then are removed.
        # Hence, if stopped networks are not removed from array A,
        # then A[orig_indices] is the array A with the stopped indices removed.

        # network parameters
        self.a = np.copy(initial_a)
        self.b = np.copy(initial_b)
        self.c = np.copy(initial_c)
        self.w = np.copy(initial_w)

        self.num_hidden = self.a.shape[2]
        self.num_parallel = self.a.shape[0]

        # a copy such that the initial values can be retrieved later on
        self.a_initial = np.copy(self.a)
        self.b_initial = np.copy(self.b)
        self.c_initial = np.copy(self.c)
        self.c_initial = np.copy(self.w)

        # will be filled whenever a network stops
        self.a_terminal = np.zeros(self.a.shape)
        self.b_terminal = np.zeros(self.b.shape)
        self.c_terminal = np.zeros(self.c.shape)
        self.w_terminal = np.zeros(self.w.shape)

        self.iteration_count = 0
        self.check_count = 0

        # whenever a network is stopped for a reason, the reason is stored in this array
        self.training_status = np.array([TrainingStatus.UNFINISHED] * self.num_parallel)
        self.stop_iteration = np.array([-1] * self.num_parallel)  # the iteration count in which a network was stopped

        # see above
        self.orig_indices = np.arange(self.num_parallel)

        # stores the maximum absolute kink value of each network,
        # where the maximum is taken over all training iterations and hidden neurons
        self.max_kink_movement = np.zeros(self.num_parallel)

        self.train_setups = train_setups
        self.lrs = lrs

        # assumes that all the train_setups have the same x and y values, although they might have different weights
        self.x = np.copy(self.train_setups[0].x)
        self.y = np.copy(self.train_setups[0].y)
        self.x_weights = np.copy(np.array([train_setups[i].x_weights for i in range(self.num_parallel)]))
        self.min_abs_x = np.min(np.abs(self.x))
        self.val_x_weights = val_x_weights  # validation weights (for the x and y values above)

    def create_training_vars(self, use_adam=False):
        # creates an intermediate object which contains temporary variables
        # these are in a different class so that they are not saved when an object of CheckingNN is serialized
        return TrainingVariables(self.get_num_unfinished(), self.num_hidden, self.x, self.y,
                                 self.x_weights[self.orig_indices], self.lrs[self.orig_indices], use_adam=use_adam)

    # trains for one epoch using gradient descent (no minibatching)
    def train_one_epoch(self, vars : TrainingVariables):
        np.multiply(self.a, vars.X, out=vars.relu_result)
        vars.relu_result += self.b
        np.less(vars.relu_result, 0, out=vars.inactive)
        vars.relu_result[vars.inactive] = 0
        np.multiply(self.w, vars.relu_result, out=vars.wr)
        np.sum(vars.wr, axis=2, keepdims=True, out=vars.f_minus_y_times_lr)
        vars.f_minus_y_times_lr += self.c
        vars.f_minus_y_times_lr -= vars.Y

        vars.f_minus_y_times_lr *= vars.X_weights  # allows different weighting for the different dataset samples
        vars.f_minus_y_times_lr *= vars.neg_lrs  # lr should be a vector of learning rates for each parallel instance

        np.sum(vars.f_minus_y_times_lr, axis=1, keepdims=True, out=vars.step_c)
        self.c += vars.step_c

        np.multiply(vars.f_minus_y_times_lr, self.w, out=vars.fmy_lr_w)
        vars.fmy_lr_w[vars.inactive] = 0
        np.sum(vars.fmy_lr_w, axis=1, keepdims=True, out=vars.step)
        self.b += vars.step

        np.multiply(vars.fmy_lr_w, vars.X, out=vars.fmy_lr_w)
        np.sum(vars.fmy_lr_w, axis=1, keepdims=True, out=vars.step)
        self.a += vars.step

        np.multiply(vars.f_minus_y_times_lr, vars.relu_result, out=vars.fmy_lr_w)
        np.sum(vars.fmy_lr_w, axis=1, keepdims=True, out=vars.step)
        self.w += vars.step

        self.max_kink_movement[self.orig_indices] = np.maximum(self.max_kink_movement[self.orig_indices],
                                                               np.max(np.abs(self.b / self.a), axis=2).flatten())

        self.iteration_count += 1

    def adam_step(self, vars : TrainingVariables):
        vars.beta_1_power *= vars.beta_1
        vars.beta_2_power *= vars.beta_2

        np.multiply(self.a, vars.X, out=vars.relu_result)
        vars.relu_result += self.b
        np.less(vars.relu_result, 0, out=vars.inactive)
        vars.relu_result[vars.inactive] = 0
        np.multiply(self.w, vars.relu_result, out=vars.wr)
        np.sum(vars.wr, axis=2, keepdims=True, out=vars.f_minus_y_times_lr)
        vars.f_minus_y_times_lr += self.c
        vars.f_minus_y_times_lr -= vars.Y

        vars.f_minus_y_times_lr *= vars.X_weights  # allows different weighting for the different dataset samples
        #vars.f_minus_y_times_lr *= vars.neg_lrs  # lr should be a vector of learning rates for each parallel instance

        np.sum(vars.f_minus_y_times_lr, axis=1, keepdims=True, out=vars.step_c)
        vars.sq_grad_acc_c *= vars.beta_2
        vars.sq_grad_acc_c += (1-vars.beta_2) * vars.step_c**2
        vars.grad_acc_c *= vars.beta_1
        vars.grad_acc_c += (1 - vars.beta_1) * vars.step_c
        self.c += vars.neg_lrs * (vars.grad_acc_c / (1.0 - vars.beta_1_power)) \
                  / (np.sqrt(vars.sq_grad_acc_c / (1.0 - vars.beta_2_power)) + vars.eps)

        np.multiply(vars.f_minus_y_times_lr, self.w, out=vars.fmy_lr_w)
        vars.fmy_lr_w[vars.inactive] = 0
        np.sum(vars.fmy_lr_w, axis=1, keepdims=True, out=vars.step)
        vars.sq_grad_acc_b *= vars.beta_2
        vars.sq_grad_acc_b += (1 - vars.beta_2) * vars.step ** 2
        vars.grad_acc_b *= vars.beta_1
        vars.grad_acc_b += (1 - vars.beta_1) * vars.step
        self.b += vars.neg_lrs * (vars.grad_acc_b / (1.0 - vars.beta_1_power)) \
                  / (np.sqrt(vars.sq_grad_acc_b / (1.0 - vars.beta_2_power)) + vars.eps)

        np.multiply(vars.fmy_lr_w, vars.X, out=vars.fmy_lr_w)
        np.sum(vars.fmy_lr_w, axis=1, keepdims=True, out=vars.step)
        vars.sq_grad_acc_a *= vars.beta_2
        vars.sq_grad_acc_a += (1 - vars.beta_2) * vars.step ** 2
        vars.grad_acc_a *= vars.beta_1
        vars.grad_acc_a += (1 - vars.beta_1) * vars.step
        self.a += vars.neg_lrs * (vars.grad_acc_a / (1.0 - vars.beta_1_power)) \
                  / (np.sqrt(vars.sq_grad_acc_a / (1.0 - vars.beta_2_power)) + vars.eps)

        np.multiply(vars.f_minus_y_times_lr, vars.relu_result, out=vars.fmy_lr_w)
        np.sum(vars.fmy_lr_w, axis=1, keepdims=True, out=vars.step)
        vars.sq_grad_acc_w *= vars.beta_2
        vars.sq_grad_acc_w += (1 - vars.beta_2) * vars.step ** 2
        vars.grad_acc_w *= vars.beta_1
        vars.grad_acc_w += (1 - vars.beta_1) * vars.step
        self.w += vars.neg_lrs * (vars.grad_acc_w / (1.0 - vars.beta_1_power)) \
                  / (np.sqrt(vars.sq_grad_acc_w / (1.0 - vars.beta_2_power)) + vars.eps)

        self.max_kink_movement[self.orig_indices] = np.maximum(self.max_kink_movement[self.orig_indices],
                                                               np.max(np.abs(self.b / self.a), axis=2).flatten())

        self.iteration_count += 1

    def terminate_instances(self, vars, bool_arrays, statuses, testing=False):
        # takes a list of boolean arrays that indicate for different reasons
        # which instances should be terminated for that reason
        # statuses contains the reasons represented by TrainingStatus enumeration items
        # order matters for setting the statuses in case that an instance should be terminated for more than one reason
        # (later reasons are dominant)

        # if testing is true, locally_converged entries will be kept running to check that they are not crossing later
        if testing:
            reduced_bool_arrays = [bool_arrays[i] for i in range(len(bool_arrays)) if statuses[i] != TrainingStatus.LOCALLY_CONVERGED]
            finished_indices = np.any(reduced_bool_arrays, axis=0)
        else:
            finished_indices = np.any(bool_arrays, axis=0)

        unfinished_indices = ~finished_indices

        self.a_terminal[self.orig_indices[finished_indices], :, :] = self.a[finished_indices, :, :]
        self.b_terminal[self.orig_indices[finished_indices], :, :] = self.b[finished_indices, :, :]
        self.c_terminal[self.orig_indices[finished_indices], :, :] = self.c[finished_indices, :, :]
        self.w_terminal[self.orig_indices[finished_indices], :, :] = self.w[finished_indices, :, :]

        for arr, status in zip(bool_arrays, statuses):
            # this can be used with the above code to test that no locally_converged net will also cross later
            if testing:
                if status == TrainingStatus.CROSSED and np.any(self.training_status[self.orig_indices[arr]] == TrainingStatus.LOCALLY_CONVERGED):
                    print('Error: Locally converged and crossed')
                    exit()

            self.training_status[self.orig_indices[arr]] = status

        self.stop_iteration[self.orig_indices[finished_indices]] = self.iteration_count

        # remove values for finished networks
        self.a = self.a[unfinished_indices, :, :]
        self.b = self.b[unfinished_indices, :, :]
        self.c = self.c[unfinished_indices, :, :]
        self.w = self.w[unfinished_indices, :, :]

        vars.neg_lrs = vars.neg_lrs[unfinished_indices, :, :]
        vars.X_weights = vars.X_weights[unfinished_indices, :, :]
        vars.relu_result = vars.relu_result[unfinished_indices, :, :]
        vars.inactive = vars.inactive[unfinished_indices, :, :]
        vars.wr = vars.wr[unfinished_indices, :, :]
        vars.f_minus_y_times_lr = vars.f_minus_y_times_lr[unfinished_indices, :, :]
        vars.step_c = vars.step_c[unfinished_indices, :, :]
        vars.step = vars.step[unfinished_indices, :, :]
        vars.fmy_lr_w = vars.fmy_lr_w[unfinished_indices, :, :]
        vars.wait_times = vars.wait_times[unfinished_indices]
        vars.best_validation_losses = vars.best_validation_losses[unfinished_indices]

        if vars.use_adam:
            vars.sq_grad_acc_a = vars.sq_grad_acc_a[unfinished_indices, :, :]
            vars.sq_grad_acc_b = vars.sq_grad_acc_b[unfinished_indices, :, :]
            vars.sq_grad_acc_c = vars.sq_grad_acc_c[unfinished_indices, :, :]
            vars.sq_grad_acc_w = vars.sq_grad_acc_w[unfinished_indices, :, :]

            vars.grad_acc_a = vars.grad_acc_a[unfinished_indices, :, :]
            vars.grad_acc_b = vars.grad_acc_b[unfinished_indices, :, :]
            vars.grad_acc_c = vars.grad_acc_c[unfinished_indices, :, :]
            vars.grad_acc_w = vars.grad_acc_w[unfinished_indices, :, :]

        self.orig_indices = self.orig_indices[unfinished_indices]

    # computes validation losses for each network
    def compute_validation_losses(self):
        pred = self.predict(self.x)
        errors = expand(self.y, [0]) - pred
        return np.sum(errors * errors * self.val_x_weights[self.orig_indices], axis=1)

    def train(self, stopping_criteria, max_num_checks, num_minibatches_per_check, minibatch_size=None, verbose=True, use_adam=False):
        # If minibatch_size is None, non-stochastic gradient descent is used.
        # For stochastic gradient descent,
        # minibatches are subsampled independently from the data and not cycled through the data.

        vars = self.create_training_vars(use_adam=use_adam)

        for check_count in range(max_num_checks):
            if minibatch_size is None:
                for i in range(num_minibatches_per_check):
                    if verbose and ((num_minibatches_per_check < 10) or (i % (num_minibatches_per_check // 10) == 0)):
                        print('.', end='', flush=True)

                    if use_adam:
                        self.adam_step(vars)
                    else:
                        self.train_one_epoch(vars)
            else:
                relevant_x_weights = self.x_weights[self.orig_indices, :]
                stochastic_X_weights = np.asarray([np.random.multinomial(minibatch_size, relevant_x_weights[i, :], num_minibatches_per_check) / minibatch_size
                                                            for i in range(relevant_x_weights.shape[0])])
                for i in range(num_minibatches_per_check):
                    vars.X_weights = np.expand_dims(stochastic_X_weights[:, i, :], axis=2)
                    if verbose and ((num_minibatches_per_check < 10) or (i % (num_minibatches_per_check // 10) == 0)):
                        print('.', end='', flush=True)

                    if use_adam:
                        self.adam_step(vars)
                    else:
                        self.train_one_epoch(vars)

            for stopping_criterion in stopping_criteria:
                stopping_criterion.check(self, vars)
                if self.get_num_unfinished() == 0:
                    break

            self.check_count += 1

            print('Check {} [{} unfinished, {} crossed, {} early stopped, {} locally converged, {} a-degenerate, {} x-degenerate] for n_hidden={}'.format(
                        self.check_count, self.get_num_unfinished(), self.get_num_crossed(), self.get_num_early_stopped(),
                        self.get_num_locally_converged(), self.get_num_a_degenerate(), self.get_num_x_degenerate(),
                        self.num_hidden))

            if self.get_num_unfinished() == 0:
                break



        print(
            'Terminated [{} unfinished, {} crossed, {} early stopped, {} locally converged, {} a-degenerate, {} x-degenerate] for n_hidden={}'.format(
                self.get_num_unfinished(), self.get_num_crossed(),
                self.get_num_early_stopped(), self.get_num_locally_converged(), self.get_num_a_degenerate(),
                self.get_num_x_degenerate(), self.num_hidden))

        print('')

    def get_num_crossed(self):
        return np.count_nonzero(self.training_status == TrainingStatus.CROSSED)

    def get_num_a_degenerate(self):
        return np.count_nonzero(self.training_status == TrainingStatus.A_DEGENERATE)

    def get_num_x_degenerate(self):
        return np.count_nonzero(self.training_status == TrainingStatus.X_DEGENERATE)

    def get_num_locally_converged(self):
        return np.count_nonzero(self.training_status == TrainingStatus.LOCALLY_CONVERGED)

    def get_num_unfinished(self):
        return np.count_nonzero(self.training_status == TrainingStatus.UNFINISHED)

    def get_num_early_stopped(self):
        return np.count_nonzero(self.training_status == TrainingStatus.EARLY_STOPPED)

    def predict(self, x, instance=None):
        # evaluates all nets (if instance is None) or a single net on a list x of input points
        if instance is None:
            X = expand(x, [0, 2])
            relu_result = self.a * X + self.b
            relu_result[relu_result < 0] = 0
            pred = np.sum(self.w * relu_result, axis=2, keepdims=True) + self.c
            return pred[:, :, 0]
        else:
            X = expand(x, [1])
            a = self.a[instance, :, :]
            b = self.b[instance, :, :]
            c = self.c[instance, :, :]
            w = self.w[instance, :, :]
            relu_result = a * X + b
            inactive = relu_result < 0
            relu_result[inactive] = 0
            pred = np.sum(w * relu_result, axis=1, keepdims=True) + c

            return pred[:, 0]

    def reset_locally_converged(self):
        # resets locally converged to unfinished. Does not reset a vars instance,
        # hence must not be called during a training run
        self.a_terminal[self.orig_indices, :, :] = self.a
        self.b_terminal[self.orig_indices, :, :] = self.b
        self.c_terminal[self.orig_indices, :, :] = self.c
        self.w_terminal[self.orig_indices, :, :] = self.w

        self.training_status[self.training_status == TrainingStatus.LOCALLY_CONVERGED] = TrainingStatus.UNFINISHED
        self.orig_indices = np.arange(self.num_parallel)[self.training_status == TrainingStatus.UNFINISHED]
        self.a = self.a_terminal[self.orig_indices, :, :]
        self.b = self.b_terminal[self.orig_indices, :, :]
        self.c = self.c_terminal[self.orig_indices, :, :]
        self.w = self.w_terminal[self.orig_indices, :, :]


    @staticmethod
    def createRandomWeights(n_parallel, n_hidden, swap_variances=False):
        # creates random weights according to He et al. (for swap_variances = False)
        # or with swapped variances for a and w (for swap_variances = True)
        a_variance = 2/1
        w_variance = 2/n_hidden

        if swap_variances:
            tmp = a_variance
            a_variance = w_variance
            w_variance = tmp

        a = np.sqrt(a_variance) * np.random.randn(n_parallel, 1, n_hidden)
        b = np.zeros(shape=(n_parallel, 1, n_hidden))
        c = np.zeros(shape=(n_parallel, 1, 1))
        w = np.sqrt(w_variance) * np.random.randn(n_parallel, 1, n_hidden)

        return a, b, c, w


class MCConfiguration(object):
    # Configuration for a monte carlo run
    def __init__(self, n_parallel, n_hidden, n_samples, seed, index, description=None):
        self.n_parallel = n_parallel
        self.n_hidden = n_hidden
        self.n_samples = n_samples
        self.index = index
        self.seed = seed  # each thread should have a different seed, otherwise they all use the same random samples
        self.description = description


def get_param_combinations():
    # computes a list of MCConfiguration objects such that computations are done for various network sizes
    param_combinations = []

    np.random.seed(1234567890)

    np.random.randn(1000)  # generate some random samples for warming up

    min_log = 4  # 16
    max_log = 11  # 2048
    num_mc_runs = 10000  # use 10000 runs per network size
    index = 0
    for k in range(min_log, max_log + 1):
        n_hidden = 2 ** k
        n_samples = n_hidden ** 2
        n_parallel = min(2 ** (max(16 - k, 3)), num_mc_runs)  # the larger each individual net, the less nets per worker

        num_remaining_parallel = num_mc_runs
        while num_remaining_parallel > 0:
            num_parallel_here = min(n_parallel, num_remaining_parallel)
            seed = np.random.randint(2**30)
            param_combinations.append(MCConfiguration(num_parallel_here, n_hidden, n_samples, seed, index))
            num_remaining_parallel -= num_parallel_here
            index += 1

    return param_combinations


def mc_runner(config, offset_str, base_dir, use_sgd, use_early_stopping, use_sufficient_stopping, use_small_lr, initialize_custom, use_adam):
    # method that can be run for a MCConfiguration object to do the computations and save them to a folder
    start_time = time.time()
    np.random.seed(config.seed)
    x = np.array([-1.0, -2.0, -3.0, 1.0, 2.0, 3.0])
    y = np.array([-1.0, 2.0, -1.0, 1.0, -2.0, 1.0]) + float(offset_str)
    x_weights = np.random.multinomial(config.n_samples, [1. / 6.] * 6, size=config.n_parallel) / config.n_samples
    val_x_weights = np.random.multinomial(config.n_samples, [1. / 6.] * 6, size=config.n_parallel) / config.n_samples
    # x_weights = 1./6. * np.ones(shape=(n_parallel, 6))
    (a, b, c, w) = CheckingNN.createRandomWeights(config.n_parallel, config.n_hidden, swap_variances=initialize_custom)
    train_setups = [TrainingSetup(x, x_weights[i, :], y) for i in range(config.n_parallel)]

    if use_small_lr:
        lrs = np.array([1e-2 / config.n_hidden] * config.n_parallel)
    else:
        lrs = np.array([train_setups[i].compute_lr(a[i, 0, :], b[i, 0, :], w[i, 0, :]) for i in range(config.n_parallel)])

    net = CheckingNN((a, b, c, w), train_setups, lrs, val_x_weights=val_x_weights)
    max_num_checks = 10000 if initialize_custom else 1000
    num_minibatches_per_check = 100 if initialize_custom else 1000
    minibatch_size = 16 if (use_sgd or use_adam) else None  # non-stochastic gradient descent
    stopping_criteria = [CrossingStoppingCriterion()]
    if use_early_stopping:
        stopping_criteria.append(EarlyStoppingCriterion(min_delta=1e-8, patience=10))
    if use_sufficient_stopping:
        stopping_criteria.append(SufficientStoppingCriterion())

    net.train(stopping_criteria=stopping_criteria, max_num_checks=max_num_checks,
              num_minibatches_per_check=num_minibatches_per_check, minibatch_size=minibatch_size, verbose=False,
              use_adam=use_adam)

    target_folder = base_dir + 'mc-data-{}/hidden-{}_parallel-{}_id-{}_time-{}/'.format(offset_str, config.n_hidden, config.n_parallel, config.index, int(start_time*1000))
    utils.serialize(target_folder+'net.p', net)
    utils.serialize(target_folder+'config.p', config)


class OffsetMCRunner(object):
    # Class that saves some more parameters for mc_runner
    def __init__(self, offset_str, base_dir, use_sgd, use_early_stopping, use_sufficient_stopping, use_small_lr, initialize_custom, use_adam):
        self.offset_str = offset_str
        self.use_sgd = use_sgd
        self.base_dir = base_dir
        self.use_early_stopping = use_early_stopping
        self.use_sufficient_stopping = use_sufficient_stopping
        self.use_small_lr = use_small_lr
        self.initialize_custom = initialize_custom
        self.use_adam = use_adam

    def __call__(self, config):
        mc_runner(config, self.offset_str, self.base_dir, self.use_sgd, self.use_early_stopping, self.use_sufficient_stopping, self.use_small_lr, self.initialize_custom, self.use_adam)


def execute_mc(offset_str, base_dir='./mc-data/', use_sgd=False, use_early_stopping=False, use_sufficient_stopping=True,
               use_small_lr=False, initialize_custom=False, use_adam=False):
    # creates a thread pool and runs all computation tasks
    param_combinations = get_param_combinations()
    num_processes = max(1, multiprocessing.cpu_count()//2)
    pool = multiprocessing.Pool(processes=num_processes)
    utils.ensureDir(base_dir + 'mc-data-{}/'.format(offset_str))
    pool.map(OffsetMCRunner(offset_str, base_dir, use_sgd, use_early_stopping, use_sufficient_stopping, use_small_lr, initialize_custom, use_adam), param_combinations, chunksize=1)
    pool.terminate()
    pool.join()


if __name__ == '__main__':
    # always add a / at the end of all directories
    # (code is intended for linux use, otherwise you need to convert / to \\)
    base_dir = './mc-data/'
    use_sgd = False  # whether sgd or gd should be used
    use_early_stopping = False  # should be used for sgd since sufficient stopping is not constructed for sgd
    # the sufficient stopping criterion only works provably for (non-stochastic) gradient descent
    use_sufficient_stopping = True
    use_small_lr = False  # if true, use 0.01*n^{-1} instead of \lambda_{max}(A^{ref} M)^{-1}
    initialize_custom = False  # if true, swap the variances of a and w initialization
    use_adam = False

    for offset_str in ['0', '0.01', '0.1']:
        execute_mc(offset_str, base_dir=base_dir, use_sgd=use_sgd, use_early_stopping=use_early_stopping,
                   use_sufficient_stopping=use_sufficient_stopping, use_small_lr=use_small_lr, initialize_custom=initialize_custom, use_adam=use_adam)


    # this can be used to run a single task for quick experiments
    #OffsetMCRunner('0', './mc-data-sgd/', use_sgd=False, use_early_stopping=False,
    #               use_sufficient_stopping=True, use_small_lr=False, initialize_custom=True)(MCConfiguration(n_parallel=100, n_hidden=2048, n_samples=2048**2, seed=0, index=0))
