# Convergence Analysis of Neural Networks
This code can be used to reproduce figures from my master's thesis ["Convergence Analysis of Neural Networks"](https://elib.uni-stuttgart.de/handle/11682/10729?locale=en). It is licensed under the Apache 2.0 license.
It mainly requires numpy, scipy and matplotlib although one of the plots can be compared against a (slower) keras implementation. It uses CPU computations only and is parallelized so that it can be used efficiently on a many-core shared-RAM system (although distributing it across multiple machines should not be too hard).
The functionality is distributed as follows:
- `mc_training.py` contains a NN class that allows to train multiple two-layer SISO (single-input single-output) ReLU networks in parallel. It also contains code to train and save many such networks in parallel. The code is configured to reproduce the plot for GD with the larger step size and only the sufficient stopping criterion, but it can be easily reconfigured for other plots at the bottom of the file. When running the code with multiple configurations, one should use different output directories for each configuration. Note that this might take more than 1000 hours to evaluate on a single core machine, so it might be useful to reduce the number of Monte Carlo evaluations or the maximum number of neurons in `get_param_combinations()`. Code with early stopping takes much less time to execute since network training is stopped earlier. It is not recommended to use the small learning rate without early stopping since this might lead to very long execution times. The code uses a fixed random seed so the results should be reproducible.
- `mc_plotting.py` contains code that takes the data generated by `mc_training.py`, prints statistics and generates a LaTeX plot which can then be compiled manually.
- `mc_sgd_keras.py` contains code for training equivalent NNs with keras with SGD, early stopping and the small learning rate. This can be used to verify the numpy implementation but is much slower since only one network is trained at the same time in a thread. The results can also be plotted using `mc_plotting.py` by setting the keras argument to true at the bottom of the file.
- `plot_examples.py` contains code to plot some figures that visualize the training process using matplotlib.
- `mc_event_estimation.py` contains code to compute Monte Carlo estimates for the probability of the event `E_{n, N, \varepsilon, \gamma}` defined in the thesis. It is not used in the thesis since the results heavily depend on epsilon, gamma and a potential scaling of the right-hand side in conditions such as (W7). For example, (W7) is seldom satisfied for small epsilon and realistic network sizes, but if the right hand side is scaled to `10 * N^{\varepsilon}`, then (W7) is frequently satisfied.
- `TrainingSetup.py` contains a helper class that performs computations based on the theory of the thesis.
- `utils.py` contains helper functions.

The files `tex_head.txt` and `tex_tail.txt` are used by `mc_plotting.py` for generating the LaTeX plot. 
