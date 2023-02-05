
BICePs - Bayesian Inference of Conformational Populations
=========================================================

<!-- List badges here: -->
[![Documentation Status](https://readthedocs.org/projects/biceps/badge/?version=latest)](https://biceps.readthedocs.io/en/latest/?badge=latest)
[![DOI for Citing BICePs](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2022--1b24c-green.svg)](https://doi.org/10.26434/chemrxiv-2022-1b24c)
      


The BICePs algorithm (Bayesian Inference of Conformational Populations)
is a statistically rigorous Bayesian inference method to reconcile
theoretical predictions of conformational state populations with sparse
and/or noisy experimental measurements and objectively compare different
models. Supported experimental observables include: 

- [NMR nuclear Overhauser effect](https://en.wikipedia.org/wiki/Nuclear_Overhauser_effect)  (`NOE`).

- [NMR chemical shifts](https://en.wikipedia.org/wiki/Chemical_shift) (`HA`,`NH`, `CA` and `N`). 

- [J couplings](https://en.wikipedia.org/wiki/J-coupling) (both small molecules and amino acids) (`J`).

- [Hydrogen--deuterium exchange](https://en.wikipedia.org/wiki/Hydrogen–deuterium_exchange) (`HDX`).


Installation
------------

We recommend that you install `biceps` via `pip`:

```bash
    $ pip install BICePs
```


Some dependencies of BICePs
---------------------------

> -   [pymbar](https://pymbar.readthedocs.io) >= 4.0.1
> -   [mdtraj](https://mdtraj.org) >= 1.5.0



Documentation
-------------

[https://biceps.readthedocs.io/en/latest/](https://biceps.readthedocs.io/en/latest/)









