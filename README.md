# Training Two-Layer ReLU Networks with Gradient Descent is Inconsistent
![Identifier]( https://img.shields.io/badge/doi-10.18419%2Fdarus--2978-d45815.svg)

*Update (2021)*: We have added code for evaluations of different training configurations (optimizer, init, parameterization) on various data sets of various dimensions for an upcoming version of our paper.

*Update (2020)*: We have added code for an evaluation on another star-like dataset (or rather its corresponding distribution) for a new version of our paper.

This code can be used to reproduce figures from our paper ["Training Two-Layer ReLU Networks with Gradient Descent is Inconsistent"](https://arxiv.org/abs/2002.04861), building on my master's thesis ["Convergence Analysis of Neural Networks"](https://elib.uni-stuttgart.de/handle/11682/10729?locale=en). It is licensed under the Apache 2.0 license.
It mainly requires numpy, scipy and matplotlib, the newer code involving higher-dimensional data sets requires PyTorch and one of the plots can be compared against a (slower) keras implementation. The older code for computations on 1D data uses CPU computations only and is parallelized so that it can be used efficiently on a many-core shared-RAM system (although distributing it across multiple machines should not be too hard).

## Installation

If the keras code (slower, only for comparison of one specific experiment) should be used, Python <= 3.7 is required to use the old keras and tensorflow versions. The code should also run with newer Python versions.

We have tested the dependencies in `requirements.txt` on Python 3.7.3. They can be installed using `pip3 install requirements.txt`. If you want to use the newest versions of the libraries, you can use
```
pip3 install numpy scipy matplotlib torch keras==2.3.1 tensorflow==1.15.5 protobuf==3.20.0 imageio fire
```
It might be helpful to use `pip3 install --upgrade pip` and `pip3 install wheel` before.

## Files and Functionality

The functionality is distributed as follows:
- `mc_training.py` contains a NN class that allows to train multiple two-layer SISO (single-input single-output) ReLU networks in parallel (vectorized). It also contains code to train and save many such networks in multiple threads. The code is configured to reproduce the plot for GD with the larger step size and only the sufficient stopping criterion, but it can be easily reconfigured for other plots at the bottom of the file. When running the code with multiple configurations, one should use different output directories for each configuration. Note that this might take more than 1000 hours to evaluate on a single core machine, so it might be useful to reduce the number of Monte Carlo evaluations or the maximum number of neurons in `get_param_combinations()`. Code with early stopping takes much less time to execute since network training is stopped earlier. It is not recommended to use the small learning rate without early stopping since this might lead to very long execution times. The code uses a fixed random seed so the results should be reproducible.
- `mc_plotting.py` contains code that takes the data generated by `mc_training.py`, prints statistics and generates a LaTeX plot which can then be compiled manually.
- `mc_sgd_keras.py` contains code for training equivalent NNs with keras with SGD, early stopping and the small learning rate. This can be used to verify the numpy implementation but is much slower since only one network is trained at the same time in a thread. The results can also be plotted using `mc_plotting.py` by setting the keras argument to true at the bottom of the file.
- `plot_examples.py` contains code to plot some figures that visualize the training process using matplotlib.
- `show_training.py` contains additional plotting code used by `plot_examples.py`.
- `mc_event_estimation.py` contains code to compute Monte Carlo estimates for the probability of the event `E_{n, N, \varepsilon, \gamma}` defined in the thesis. It is not used in the thesis since the results heavily depend on epsilon, gamma and a potential scaling of the right-hand side in conditions such as (W7). For example, (W7) is seldom satisfied for small epsilon and realistic network sizes, but if the right hand side is scaled to `10 * N^{\varepsilon}`, then (W7) is frequently satisfied already for small network sizes.
- `TrainingSetup.py` contains a helper class that performs computations based on the theory of the thesis.
- `train_star_dataset.py` contains code to train NNs on a GPU with PyTorch on the star dataset. Executing this code as-is might take a day on a modern GPU. You can also modify the `run_experiments()` function to run less configurations.
- `eval_star_dataset.py` contains code that prints the results of the evaluation on the star dataset.
- `custom_paths.py` allows to configure a folder under which some of the results should be saved
- `run_nn_setups.py` contains code to train various configurations (optimizer, init, parameterization) of NNs on a GPU with PyTorch on randomly sampled datasets of various distributions, which is used for the new Section 10 of the upcoming version of our paper. Running this might take around 2-4 GPU-days on a modern GPU.
- `eval_nn_setups.py` generates plots using the results generated by `run_nn_setups.py`.
- `utils.py` contains helper functions.

The files `tex_head.txt` and `tex_tail.txt` are used by `mc_plotting.py` for generating the LaTeX plot. 
