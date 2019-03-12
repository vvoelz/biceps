Workflow
========

A typical BICePs sampling includes four core steps: :class:`biceps.Preparation`, :class:`biceps.Restraint`, :class:`biceps.PosteriorSampler` and :class:`biceps.Analysis`.

Preparation
-----------

The :class:`biceps.Preparation` class converts all raw data to  BICePs readable format. It asks for experimental data as well as correspondingly precomputed experimental observables from simulation. We recommend users to use `MDTraj <http://mdtraj.org>`_ to compute all necessary experimental quantities from simulation or use our prepared functions in the `toolbox <api/toolbox>`. Check more details in the :ref:`examples <examples/index>` page.

Restraint
---------

The :class:`biceps.Restraint` class initializes all necessary functions to construct numpy array containing information for BICePs sampling. As a parent class, it also includes child classes for different experimental restraints.

PosteriorSampler
----------------

The :class:`biceps.PosteriorSampler` class is closely working with the :class:`biceps.Restraint` class. A Markov chain Monte Carlo sampling is performed based on the `Metroplis-Hastings criterion <https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm>`_ .

Analysis
--------

The :class:`biceps.Analysis` is consist of two parts:

1. Using `MBAR <https://pymbar.readthedocs.io/en/master/index.html>`_ algorithm to compute populations and :ref:`BICePs scores <theory>`

2. Plot the figures to show population and :ref:`nuisance parameters <theory>`.

