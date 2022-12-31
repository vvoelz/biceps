
BICePs - Bayesian Inference of Conformational Populations
=========================================================

<!-- List badges here: -->
[![Documentation Status](https://readthedocs.org/projects/biceps/badge/?version=latest)](https://biceps.readthedocs.io/en/latest/?badge=latest)
      

<!--                   -->

The BICePs algorithm (Bayesian Inference of Conformational Populations)
is a statistically rigorous Bayesian inference method to reconcile
theoretical predictions of conformational state populations with sparse
and/or noisy experimental measurements and objectively compare different
models. Supported experimental observables include: 

- [NMR nuclear Overhauser effect](https://en.wikipedia.org/wiki/Nuclear_Overhauser_effect)  (`NOE`).

- [NMR chemical shifts](https://en.wikipedia.org/wiki/Chemical_shift) (`HA`,`NH`, `CA` and `N`). 

- [J couplings](https://en.wikipedia.org/wiki/J-coupling) (both small molecules and amino acids) (`J`).

- [Hydrogen--deuterium exchange](https://en.wikipedia.org/wiki/Hydrogen–deuterium_exchange) (`HDX`).

<!--
Citation [![DOI for Citing BICePs](https://img.shields.io/badge/DOI-10.1021.acs.jpcb.7b11871-green.svg)](http://doi.org/10.1021/acs.jpcb.7b11871)
-->

Citation [![DOI for Citing BICePs](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2022--1b24c-green.svg)](https://doi.org/10.26434/chemrxiv-2022-1b24c)


### Check our [BICePs website](https://biceps.readthedocs.io/en/latest/) for more details!

### Please check out the [theory of **BICePs**](https://biceps.readthedocs.io/en/latest/theory.html) to learn more.

Installation (in progress)
==========================

<!--
We recommend that you install `BICePs` with `conda`. :

```bash
    $ conda install -c conda-forge BICePs
```

You can install also `BICePs` with `pip`, if you prefer. :

```bash
    $ pip install BICePs
```
-->
<!--
Conda is a cross-platform package manager built especially for
scientific python. It will install `BICePs` along with all dependencies
from a pre-compiled binary. If you don\'t have Python or the `conda`
package manager, we recommend starting with the [Anaconda Scientific
Python distribution \<https://store.continuum.io/cshop/anaconda/\>](),
which comes pre-packaged with many of the core scientific python
packages that BICePs uses (see below), or with the [Miniconda Python
distribution](http://conda.pydata.org/miniconda.html), which is a
bare-bones Python installation.
-->

BICePs supports Python 2.7 (see [tag v1.0](https://github.com/vvoelz/biceps/releases/tag/v1.0)) or Python 3.4+ (v2.0 or greater) on Mac, Linux, and Windows.


Dependencies of BICePs
----------------------

> -   [pymbar](https://pymbar.readthedocs.io) >= 4.0.1
> -   [mdtraj](https://mdtraj.org) >= 1.5.0
> -   matplotlib >= 2.1.2
> -   numpy >= 1.14.0
> -   multiprocessing 

-------------------------------------------


### View [the workflow of BICePs](https://biceps.readthedocs.io/en/latest/workflow.html).

### BICePs is research software. If you make use of BICePs in scientific publications, please cite it.

# To get started, see [biceps/releases](https://github.com/vvoelz/biceps/releases) for the latest version of BICePs.




