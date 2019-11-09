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
import scipy.linalg

class TrainingSetup(object):
    # An object of TrainingSetup describes properties of a dataset and, given a weight vector W,
    # can compute certain properties
    # Warning: The case alpha != 0 is not fully implemented.
    def __init__(self, x, x_weights, y, alpha=0.0):
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.x_weights = np.copy(x_weights)
        self.x_signed = [self.x[self.x > 0], self.x[self.x < 0]]
        self.y_signed = [self.y[self.x > 0], self.y[self.x < 0]]
        self.x_weights_signed = [self.x_weights[self.x > 0], self.x_weights[self.x < 0]]
        self.alpha = alpha
        self.min_abs_x = np.min(np.abs(x))
        self.pos = 0  # index for sign 1
        self.neg = 1  # index for sign -1

        xx_sum = [np.dot(self.x_weights_signed[sign], self.x_signed[sign] * self.x_signed[sign]) for sign in [0, 1]]
        x_sum = [np.dot(self.x_weights_signed[sign], self.x_signed[sign]) for sign in [0, 1]]

        # we're working with the tilde versions of the matrices from the thesis, i.e. the non-permuted ones.
        # Permutation does not matter for eigenvalue computation.

        M_signed = [np.asmatrix([[xx_sum[sign], x_sum[sign]], [x_sum[sign], np.sum(self.x_weights_signed[sign])]]) for sign in [0, 1]]

        self.M = np.asmatrix(np.vstack([np.hstack([M_signed[self.pos], np.zeros(shape=(2, 2))]),
                                      np.hstack([np.zeros(shape=(2, 2)), M_signed[self.neg]])]))

        self.M_inv = np.linalg.pinv(self.M)

        sqrtM_signed = [np.asarray(scipy.linalg.sqrtm(M_signed[sign])) for sign in [0, 1]]

        self.sqrtM = np.asmatrix(np.vstack([np.hstack([sqrtM_signed[self.pos], np.zeros(shape=(2, 2))]),
                                      np.hstack([np.zeros(shape=(2, 2)), sqrtM_signed[self.neg]])]))

        self.sqrtM_inv = np.linalg.pinv(self.sqrtM)

        self.C = np.asmatrix([[0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 1.0]])

        self.B = np.asmatrix([[1.0, 0.0, alpha, 0.0],
                         [0.0, 1.0, 0.0, alpha],
                         [alpha, 0.0, 1.0, 0.0],
                         [0.0, alpha, 0.0, 1.0]])

        self.Qtilde = np.asmatrix([[0, 0, 1],
                                   [0, 0, 1],
                                   [1, 1, 0]])

    # Computes a list containing the matrices \Sigma_1 and \Sigma_{-1}
    def compute_Sigma(self, a, b, w):
        Sigma = [None, None]
        indices_list = [a > 0, a < 0]
        for sign in [0, 1]:
            indices = indices_list[sign]
            asigma = a[indices]
            bsigma = b[indices]
            wsigma = w[indices]
            aa = np.dot(asigma, asigma)
            ab = np.dot(asigma, bsigma)
            bb = np.dot(bsigma, bsigma)
            wa = np.dot(asigma, wsigma)
            wb = np.dot(bsigma, wsigma)
            ww = np.dot(wsigma, wsigma)
            Sigma[sign] = np.asmatrix([[aa, ab, wa],
                                      [ab, bb, wb],
                                      [wa, wb, ww]])
        return Sigma

    # Computes the matrix A^{ref} from [\Sigma_1, \Sigma_{-1}]
    def compute_Aref(self, Sigma):
        pos = self.pos
        neg = self.neg
        Gref_list = [Sigma[sign][0:2, 0:2] + Sigma[sign][2, 2] * np.eye(2, 2) for sign in [0, 1]]
        Gref = np.asmatrix(np.vstack([np.hstack([Gref_list[pos], np.zeros(shape=(2, 2))]),
                                     np.hstack([np.zeros(shape=(2, 2)), Gref_list[neg]])]))
        return self.C + self.B * Gref * self.B

    def compute_sum_bounds(self, Aref, lr, psi_0):
        # computes upper bounds for the sums
        # S = h\sum_{k=0}^\infty \|(I - h M^{1/2} A^{ref} M^{1/2})^k\|_\infty
        # S_psi = h\sum_{k=0}^\infty \|(I - hM^{1/2} A^{ref} M^{1/2})^k \psi_0\|_\infty
        # where \psi_0 := M^{1/2} \overline{v}_0 and h = lr
        mat = self.sqrtM * Aref * self.sqrtM
        try:
            eigvals, eigvecs = np.linalg.eigh(mat)
        except np.linalg.LinAlgError:
            return np.inf, np.inf  # no bounds could be computed
        # eigenvalues of (I - lr * sqrtM * Aref * sqrtM)
        iteration_eigvals = 1 - lr * eigvals
        # eigvals are >= 0, iteration_eigvals are <= 1, must be in (-1, 1) for the geometric series to converge
        max_abs_iteration_eigval = np.max(np.abs(iteration_eigvals))

        if max_abs_iteration_eigval >= 1:
            # print('max_abs_iteration_eigval:', max_abs_iteration_eigval)
            return np.inf, np.inf  # at least S is infinity, which is a fail

        # ||U D^k U^T||_infty <= 2||U D^k U^T||_2 = 2||D^k||_2 = (max |d_ii|)^k
        S_bound = 2 * lr / (1 - max_abs_iteration_eigval)

        # transpose of orthogonal matrix is inverse
        psi_coefficients = np.asmatrix(np.transpose(eigvecs)) * psi_0
        S_psi_bound = np.sum([lr / np.abs(1 - iteration_eigvals[i]) * np.abs(psi_coefficients[i]) * np.linalg.norm(eigvecs[i, :], np.inf) for i in range(len(psi_0))])

        return S_bound, S_psi_bound

    def compute_uhat(self, Sigma, c):
        # computes the vector \hat{u} from [\Sigma_1, \Sigma_{-1}] and c
        # todo: only works for alpha = 0 at the moment
        a_ = 0
        b_ = 1
        w_ = 2
        errors = [self.x_weights_signed[sign] * (self.y_signed[sign] - (c + Sigma[sign][w_, a_] * self.x_signed[sign] + Sigma[sign][w_, b_])) for sign in [0, 1]]
        # print('errors:', errors)
        uhat_signed = [np.array([np.dot(self.x_signed[sign], errors[sign]), np.sum(errors[sign])]) for sign in [self.pos, self.neg]]
        uhat = np.hstack(uhat_signed)
        return uhat.flatten()

    def compute_ovlv(self, Sigma, c):
        # computes \overline{v}
        return self.M_inv * self.compute_uhat(Sigma, c)

    def compute_vopt(self):
        # computes v^{opt}
        uhatzero_signed = [np.array([np.sum(self.x_weights_signed[sign] * self.x_signed[sign] * self.y_signed[sign]),
                                     np.sum(self.x_weights_signed[sign] * self.y_signed[sign])]) for sign in [0, 1]]
        uhatzero = np.hstack(uhatzero_signed)
        return np.matmul(self.M_inv, uhatzero)

    def check_convergence(self, a, b, c, w, lr):
        # checks whether the theory guarantees that no kink will ever cross a datapoint
        # this is only a sufficient but not a necessary criterion

        # indices of a, b, w in Sigma
        a_ = 0
        b_ = 1
        w_ = 2

        Sigma = self.compute_Sigma(a, b, w)
        Aref = self.compute_Aref(Sigma)
        uhat = self.compute_uhat(Sigma, c)
        psi_0 = self.sqrtM_inv * np.reshape(uhat, (-1, 1))

        S_bound, S_psi_bound = self.compute_sum_bounds(Aref, lr, psi_0)

        # print('norm of psi_0:', np.linalg.norm(psi_0), ', S_psi_bound:', S_psi_bound, 'eig_Aref:', np.linalg.eigh(Aref)[0])
        # print('u_hat:', uhat)

        if S_bound == np.inf:
            # print('infty')
            return False

        norm_B_infty = 1 + np.abs(self.alpha)
        norm_sqrtM_infty = np.linalg.norm(self.sqrtM, np.inf)
        kappa_uk_bound = 2 * norm_B_infty * norm_sqrtM_infty * S_psi_bound

        # print('kappa_uk_bound:', kappa_uk_bound)
        if kappa_uk_bound >= 0.5:
            # don't even try because it probably won't work
            # (new_delta is typically >> kappa_uk_bound and must be <= 0.5)
            return False

        # difference bounds for Sigma entries according to Proposition 5.31
        difference_bounds = [kappa_uk_bound * (self.Qtilde * np.abs(Sigma[sign]) + np.abs(Sigma[sign]) * self.Qtilde)
                             + 8 * kappa_uk_bound ** 2 * np.exp(4 * kappa_uk_bound) * np.linalg.norm(Sigma[sign],
                                                                                                     np.inf)
                             for sign in [0, 1]]
        # compute difference bounds for G matrices and then for A as in the proof of Proposition 5.33
        Gw_diff_bound = np.max([difference_bounds[sign][w_, w_] for sign in [0, 1]])
        Gab_diff_bound = np.max(
            [difference_bounds[sign][a_, a_] + difference_bounds[sign][a_, b_] + difference_bounds[sign][b_, b_] for
             sign in [0, 1]])
        Gwab_bound = kappa_uk_bound * np.max([np.abs(Sigma[sign][w_, a_]) + np.abs(Sigma[sign][w_, b_])
                                              + difference_bounds[sign][w_, a_] + difference_bounds[sign][w_, b_]
                                              for sign in [0, 1]])
        A_diff_bound = norm_B_infty ** 2 * (Gw_diff_bound + Gab_diff_bound + Gwab_bound)

        # new bound for delta as in the proof Proposition 5.33
        new_delta = S_bound * (norm_sqrtM_infty ** 2) * A_diff_bound

        # print('new_delta: {}'.format(new_delta))

        if new_delta > 0.5:
            # cannot use induction argument to bound kappa_uk using S_psi
            return False

        # otherwise, kappa_uk_bound is a valid bound
        a_indices = [a > 0, a < 0]
        # print('kappa_uk_bound:', kappa_uk_bound)
        for sign in [0, 1]:
            theta_vectors = np.asmatrix(np.vstack([a[a_indices[sign]], b[a_indices[sign]], w[a_indices[sign]]]))
            # print('theta_vectors:', theta_vectors)
            # print('first product:', kappa_uk_bound * self.Qtilde * theta_vectors)
            # print('second product:', 2 * kappa_uk_bound ** 2 * np.exp(2 * kappa_uk_bound) * np.expand_dims(np.max(np.abs(theta_vectors), axis=0), axis=0))
            # print('maxabs:', np.max(np.abs(theta_vectors), axis=0))

            # compute difference bounds for the theta vectors according to Proposition 5.31
            difference_bounds = kappa_uk_bound * self.Qtilde * np.abs(theta_vectors) \
                                + 2 * (kappa_uk_bound ** 2) * np.exp(2 * kappa_uk_bound) \
                                * np.max(np.abs(theta_vectors), axis=0)  # somehow max keeps the first dimension here

            # check if the bounds allow for any kink to cross a datapoint
            if np.any(np.abs(theta_vectors[a_, :]) - difference_bounds[a_, :] <= np.abs(
                    theta_vectors[b_, :]) + difference_bounds[b_, :]):
                return False

        return True

    def compute_lr(self, a, b, w):
        # compute the learning rate 1/\lambda_{max}(A^{ref}M)
        Sigma = self.compute_Sigma(a, b, w)
        Aref = self.compute_Aref(Sigma)
        symm_matrix = self.sqrtM * Aref * self.sqrtM
        try:
            eigvals, eigvecs = np.linalg.eigh(symm_matrix)
            lmaxArefM = np.max(eigvals)
        except np.linalg.LinAlgError:
            lmaxArefM = np.linalg.norm(symm_matrix)

        return 1.0 / lmaxArefM



