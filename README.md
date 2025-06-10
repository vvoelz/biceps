
BICePs - Bayesian Inference of Conformational Populations, Version 3.0
=========================================================

<!-- List badges here: -->
[![Documentation Status](https://readthedocs.org/projects/biceps/badge/?version=latest)](https://biceps.readthedocs.io/en/latest/?badge=latest)

[![DOI for Citing BICePs v3.0a — BICePs with replica averaging](https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.5c00044-blue.svg)](https://doi.org/10.1021/acs.jctc.5c00044) — BICePs with replica averaging (BICePs v3.0a)

[![DOI for Citing BICePs v2.0](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.2c01296-purple.svg)](https://doi.org/10.1021/acs.jcim.2c01296) - BICePs v2.0


The BICePs algorithm (Bayesian Inference of Conformational Populations)
is a statistically rigorous Bayesian inference method to reconcile
theoretical predictions of conformational state populations with sparse
and/or noisy experimental measurements and objectively compare different
models. Supported experimental observables include: 

- [NMR nuclear Overhauser effect](https://en.wikipedia.org/wiki/Nuclear_Overhauser_effect)  (`NOE`).

- [NMR chemical shifts](https://en.wikipedia.org/wiki/Chemical_shift) (`HA`,`NH`, `CA` and `N`). 

- [J couplings](https://en.wikipedia.org/wiki/J-coupling) (both small molecules and amino acids) (`J`).

- [Hydrogen--deuterium exchange](https://en.wikipedia.org/wiki/Hydrogen–deuterium_exchange) (`HDX`).


Installation (BICePs v3.0a)
---------------------------


First, you will need to clone the repo. Then, you will need to navigate to the directory contain the source code: `biceps/`

You will need to compile by running the following:

```bash
bash rebuild.sh
```

...Fingers crossed that compilation works with your stock clang compiler. 

It is possible that parallelization might not work right away. Run a few tests (see below for basic BICePs scripts) to evaluate if multiprocessing is working correctly. 

If no luck, installing llvm with OMP enabled might be required.



### Basic examples & BICePs scripts

- [`examples/simple_three_state_reweighting.py`](examples/simple_three_state_reweighting.py): a script implementing BICePs to reweight populations for a simple three state toy model system.  Here, our prior comes from random generation of the Boltzmann distribution and reweighting is performed using two experimental observables both set to 0.0 A.U. 

- [`examples/replica_avg.py`](examples/replica_avg.py): a slightly more comprehensive script demonstrating the broad functionality of the BICePs module with many code blocks can be found here: `examples/replica_averaging.py`

- [`examples/Tutorial_checking_convergence_iteratively.ipynb`](examples/Tutorial_checking_convergence_iteratively.ipynb): a notebook demonstrating the ability to stop and start MCMC sampling and periodically check convergence using the built-in convergence module.

- [`examples/enforcing_uniform_reference_state.ipynb`](examples/enforcing_uniform_reference_state.ipynb): a notebook of a simple three state toy model system to demonstrate four different approaches of computing the BICePs score, $f_{\xi=0 \rightarrow 1}$ represented as the free energy of "turning on" the data restraints.
- [`examples/enforcing_uniform_reference_state.py`](examples/enforcing_uniform_reference_state.py): same as above, but in the form of a python script. The code will likely run faster and smoother in this python script. 

- [`examples/fmo_toy_by_sampling_parameters.py`](examples/fmo_toy_by_sampling_parameters.py): a toy model system for forward model optimization of Karplus parameters. In this example, we simultaneously infer the joint posterior of 100 conformational state populations as well as the optimal Karplus parameters using 60 J-coupling observables. We run 4 independent chains in parallel, each starting from different initial parameters, and see that they all converge to the same location in parameter space. Using the average parameters over these 4 chains as our optimal Karplus parameters, we quantify the BICePs score, $f_{\xi=0 \rightarrow 1}$ represented as the free energy of "turning on" the data restraints. The energy separation between end states is rather large, so we apply the pylambdaopt module to determine the optimal positioning of our intermediates to increase the quality of our free energy calculation. For more information, please read and cite: Robert M. Raddi, Tim Marshall and Vincent A. Voelz. "Automatic Forward Model Parameterization with Bayesian Inference of Conformational Populations." https://arxiv.org/abs/2405.18532 

- ['github.com/robraddi/chignolin'](https://github.com/robraddi/chignolin): a repository for the BICePs scripts used to reweight conformational ensembles of the mini-protein chignolin simulated using nine different force fields in TIP3P water, using a set of 158 experimental measurements (139 NOE distances, 13 chemical shifts, and 6 vicinal J-coupling constants for HN and Hα. For more information, please refer to https://pubs.acs.org/doi/10.1021/acs.jctc.5c0004://pubs.acs.org/doi/10.1021/acs.jctc.5c00044.



#### There are many more examples/scripts/notebooks on the way...


These remaining python files contain useful routines that are need for forward model optimization, plotting results, etc.:

- `examples/plot_marginal_likelihood_for_each_iteration.py`

- `examples/repXroutines.py`

- `examples/FwdModelOpt_routines.py`




The general BICePs protocol looks like this:

```python
ensemble = biceps.ExpandedEnsemble(energies, lambda_values)
ensemble.initialize_restraints(input_data, options, verbose=1)
sampler = biceps.PosteriorSampler(ensemble, nreplicas, write_every=1)
sampler.sample(nsteps, verbose=0, progress=1, multiprocess=1, capture_stdout=0)
A = biceps.Analysis(sampler, outdir=outdir, nstates=len(energies))
```



### Important Python classes and scripts

`biceps.Restraint`
  - Preparation
  - ExpandedEnsemble
  - get_restraint_options
  
`biceps.PosteriorSampler`
  - PosteriorSampler
  - PosteriorSamplingTrajectory



`biceps.Analysis`
  - Analysis

`biceps.convergence`
  - Convergence


`biceps.toolbox`

`biceps.decorators import multiprocess`

`biceps.J_coupling`

`biceps.KarplusRelation`







Some dependencies of BICePs
---------------------------

> -   [pymbar](https://pymbar.readthedocs.io) >= 4.0.1
> -   [mdtraj](https://mdtraj.org)



Documentation
-------------

[https://biceps.readthedocs.io/en/latest/](https://biceps.readthedocs.io/en/latest/)


Citing BICePs
-------------

```
@article{doi:10.1021/acs.jctc.5c00044,
author = {Raddi, Robert M. and Marshall, Tim and Ge, Yunhui and Voelz, Vincent A.},
title = {Model Selection Using Replica Averaging with Bayesian Inference of Conformational Populations},
journal = {Journal of Chemical Theory and Computation},
doi = {10.1021/acs.jctc.5c00044},
URL = {https://doi.org/10.1021/acs.jctc.5c00044}
}
```














