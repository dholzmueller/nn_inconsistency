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
from mc_training import CheckingNN
from TrainingSetup import TrainingSetup

class MCEventData(object):
    def __init__(self, n, N, eps, gamma, x, y, num_mc_samples, rhs_factor = 1.0):
        self.n = n
        self.N = N
        self.eps = eps
        self.gamma = gamma
        self.x = x
        self.y = y
        self.num_mc_samples = num_mc_samples
        self.rhs_factor = rhs_factor  # the rhs of the conditions is multiplied by this factor for faster convergence

        self.x_weights = np.random.multinomial(self.N, [1. / len(x)] * len(x), size=self.num_mc_samples) / self.N
        (self.a, self.b, self.c, self.w) = CheckingNN.createRandomWeights(n_parallel=self.num_mc_samples, n_hidden=self.n)
        self.train_setups = [TrainingSetup(self.x, self.x_weights[i, :], self.y) for i in range(self.num_mc_samples)]
        self.valid_indices = None

    def compute_ratio(self):
        ca = 2
        cw = 2
        a_ = 0
        b_ = 1
        w_ = 2

        n_to_the_eps = np.power(self.n, self.eps)
        # (W1) and (D3) are always satisfied
        W2_indices = np.max(np.abs(self.w[:, 0, :]), axis=1) <= self.rhs_factor * np.power(self.n, -0.5+self.eps)
        W3_indices = np.max(np.abs(self.a[:, 0, :]), axis=1) <= self.rhs_factor * n_to_the_eps
        W4_indices = np.min(np.abs(self.a[:, 0, :]), axis=1) >= (1.0 / self.rhs_factor) * np.power(self.n, -1.0-self.gamma)
        Sigma_matrices = [self.train_setups[i].compute_Sigma(self.a[i, 0, :], self.b[i, 0, :], self.w[i, 0, :])
                          for i in range(self.num_mc_samples)]
        W5_indices = np.array([np.all([0.25 * self.n * ca <= Sigma_matrices[i][sign][a_, a_] <= self.n * ca for sign in [0, 1]]) for i in range(self.num_mc_samples)])
        W6_indices = np.array([np.all([0.25 * cw <= Sigma_matrices[i][sign][w_, w_] <= cw for sign in [0, 1]]) for i in range(self.num_mc_samples)])
        W7_indices = np.array([np.all([np.abs(Sigma_matrices[i][sign][w_, a_]) <= self.rhs_factor * n_to_the_eps for sign in [0, 1]]) for i in range(self.num_mc_samples)])

        train_setup_ref = TrainingSetup(self.x, x_weights=np.array([1./len(self.x)] * len(self.x)), y=self.y)
        N_power = np.power(self.N, (self.eps - 1) * 0.5)
        vopt_ref = train_setup_ref.compute_vopt()
        D1_indices = np.array([np.max(np.abs(vopt_ref - self.train_setups[i].compute_vopt())) <= self.rhs_factor * N_power
                               for i in range(self.num_mc_samples)])
        eigvals_ref, eigvecs_ref = np.linalg.eigh(train_setup_ref.M)
        eig_lower_bound = 0.5 * np.min(eigvals_ref)
        eig_upper_bound = 2.0 * np.max(eigvals_ref)
        eigvals = [np.linalg.eigh(self.train_setups[i].M)[0] for i in range(self.num_mc_samples)]
        D2_indices = np.array([eig_lower_bound <= np.min(eigvals[i]) and np.max(eigvals[i]) <= eig_upper_bound for i in range(self.num_mc_samples)])

        self.valid_indices = np.all([W2_indices, W3_indices, W4_indices, W5_indices, W6_indices, W7_indices, D1_indices, D2_indices], axis=0)
        print(np.count_nonzero(W2_indices))
        print(np.count_nonzero(W3_indices))
        print(np.count_nonzero(W4_indices))
        print(np.count_nonzero(W5_indices))
        print(np.count_nonzero(W6_indices))
        print(np.count_nonzero(W7_indices))
        print(np.count_nonzero(D1_indices))
        print(np.count_nonzero(D2_indices))


    def get_ratio(self):
        return np.count_nonzero(self.valid_indices) / self.num_mc_samples


def mc_event_main():
    # Estimates the probability of E_{n, N, \varepsilon, \gamma} for different n and N=n^2 using Monte Carlo.
    x = np.array([-1.0, -2.0, -3.0, 1.0, 2.0, 3.0])
    y = np.array([-1.0, 2.0, -1.0, 1.0, -2.0, 1.0])
    num_mc_samples = 100
    gamma = 0.25
    eps = 0.25 - 0.5 * gamma

    for k in range(4, 20):
        n = 2**k
        N = n**2

        data = MCEventData(n, N, eps, gamma, x, y, num_mc_samples, rhs_factor=1.0)
        data.compute_ratio()
        print('Estimated event probability for n = {}: {}'.format(n, data.get_ratio()))


if __name__ == '__main__':
    mc_event_main()
