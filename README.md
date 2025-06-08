
BICePs - Bayesian Inference of Conformational Populations
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


Installation (BICePs v2.0)
--------------------------

We recommend that you install `biceps` via `pip`:

```bash
    $ pip install BICePs
```

Installation (BICePs v3.0a)
---------------------------

Please navigate to the BICePs v3.0a branch using this link: [`biceps_v3.0a`](../../tree/biceps_v3.0a)


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



