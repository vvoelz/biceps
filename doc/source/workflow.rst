workflow
=============

A typical BICePs sampling includes four core steps: :class:`Preparation`, :class:`Restraint`, :class:`Posteriorsampler` and :func:`Analysis`.

Preparation
--------------

The :class:`Preparation` class converts all raw data to  BICePs readable format. It asks for experimental data as well as correspondingly precomputed experimental observables from simulation. We recommend users to use `MDTraj <http://mdtraj.org>`_ to compute all necessary experimental quantities from simulation or use our prepared functions in the :func:`toolbox`. Check more details in the :ref:`examples <examples/index>` page.


Restraint
--------------

The :class:`Restraint` class initializes all necessary functions to construct numpy array containing information for BICePs sampling. As a parent class, it also includes child classes for different experimental restraints.

Posteriorsampler
--------------

The :class:`Posteriorsampler` class is closely working with the :class:`Restraint` class. A Markov chain Monte Carlo sampling is performed based on the `Metroplis-Hastings criterion <https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm>`_ .  

Analysis
--------------

The :func:`Analysis` is consist of two parts: 1/ using `MBAR <https://pymbar.readthedocs.io/en/master/index.html>`_ algorithm to compute populations and :ref:`BICePs scores <theory>` and 2/ plot the figures to show population and :ref:`nuisance parameters <theory>`.

