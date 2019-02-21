Workflow
========

A typical BICePs sampling includes four core python objects: `Preparation`, `Restraint`, `Posteriorsampler` and `Analysis`.

# Overview of a BICePs calculation

A BICePs calculation involves the following steps:
1. Creating a set up input files with the `Preparation` object.
2. Instantiating one or more `Restraint` classes that describe experimental and model observables.
3. Bundling these restraints to create a `Posteriorsampler` object.
4. Using the `Posteriorsampler` methods to perform MCMC sampling.
5. Analyzing the results using the methods of `Analysis` class. 


# Objects

Preparation
-----------

The `Preparation` class converts all raw
data to BICePs readable format. It asks for experimental data as well as
correspondingly precomputed experimental observables from simulation. We
recommend users to use [MDTraj](http://mdtraj.org) to compute all
necessary experimental quantities from simulation or use our prepared
functions in the `toolbox`. A tutorial jupyter notebook is availabel [here](https://github.com/vvoelz/biceps/blob/master/BICePs_2.0/tutorials/Preparation/Preparation.ipynb).

Restraint
---------

The `Restraint` class initializes all necessary functions to construct numpy array containing information for BICePs sampling. As a parent class, it also includes child classes for different experimental restraints.

Posteriorsampler
----------------

The `Posteriorsampler` class is closely working with the `Restraint` class. A Markov chain Monte Carlo sampling is performed based on the [Metroplis-Hastings criterion](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm). 

Analysis
--------

The `Analysis` is consist of two parts:
1. Using [MBAR](https://pymbar.readthedocs.io/en/master/index.html) algorithm to compute populations and `BICePs scores <theory>`.
2. plot the figures to show population and `nuisance parameters <theory>`.




