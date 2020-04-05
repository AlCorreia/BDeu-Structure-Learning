# BDeu-Structure-Learning
This is the source code for the experiments reported in papers "On Pruning for Score-Based Bayesian Network Structure Learning" [1] and "An Experimental Study of Prior Dependence in Bayesian Network Structure Learning" [2]. If this code has been useful to you, consider citing the corresponding paper (see references at the bottom).
 

### Code usage


The main code for both papers is contained in `scoring.py`. There are, however, two different scripts to run the experiments of each paper.

`pruning.py` concerns the experiments in [1] to compare different upper bounds for BDeu structure learning. The command line accepts 6 arguments:
  - **data** (-d): the name of the file containing the dataset (should include the extension .csv).
  We assume all such files are in a Datasets folder in the same path as pruning.py
  - **child** (-c): the name or index of the child
  If 'all' is passed, the program will run over all variables in the dataset
  - **bound** or (-b): the type of bound to be used.
  Either 'f', 'g', 'h', 'min' or 'all'. The latter sets the program to compute all three bounds. See [1] for details on each bound.
  - **timeout** (-t): the number of time (secs) the program waits before exiting
  - **alpha** (-a): the equivalent sample size (ESS). 
  - **palim** (-p): the maximum number of parents per variable.
  
The program outputs a csv file which is saved to a folder named after *data* (the dataset name). The csv contains the best parents and the number of scores computed for each pair (child variable, palim).

`vary_alpha.py` runs the experiments in [2] where the influence of the equivalent sample size (ESS) is studied. The command line accept similar parameters.

- **data** (-d), **child** (-c) and **timeout** (-t) as above.
- **n_runs** (-n): How many times the experiment is run. Each run corresponds to a different (random) topological ordering of the variables.

The program outputs a csv file which is saved to a folder named after alpha_*data* (the dataset name). The csv contains the best parents for each pair (child, palim, alpha).

### References

[1] Correia, AHC, Cussens J, de Campos, CP. "On Pruning for Score-Based Bayesian Network Structure Learning." In the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS 2020), 2020.

[2] Correia, AHC, Campos, CP, Gaag, LC. "An Experimental Study of Prior Dependence in Bayesian Network Structure Learning." In International Symposium on Imprecise Probabilities: Theories and Applications, pp. 78-81, 2019.

