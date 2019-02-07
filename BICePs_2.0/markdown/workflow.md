Workflow
========

A typical BICePs sampling includes four core steps:
`Preparation`{.interpreted-text role="class"},
`Restraint`{.interpreted-text role="class"},
`Posteriorsampler`{.interpreted-text role="class"} and
`Analysis`{.interpreted-text role="func"}.

Preparation
-----------

The `Preparation`{.interpreted-text role="class"} class converts all raw
data to BICePs readable format. It asks for experimental data as well as
correspondingly precomputed experimental observables from simulation. We
recommend users to use [MDTraj](http://mdtraj.org) to compute all
necessary experimental quantities from simulation or use our prepared
functions in the `toolbox`{.interpreted-text role="func"}. Check more
details in the `examples <examples/index>`{.interpreted-text role="ref"}
page.

Restraint
---------

The `Restraint`{.interpreted-text role="class"} class initializes all
necessary functions to construct numpy array containing information for
BICePs sampling. As a parent class, it also includes child classes for
different experimental restraints.

Posteriorsampler
----------------

The `Posteriorsampler`{.interpreted-text role="class"} class is closely
working with the `Restraint`{.interpreted-text role="class"} class. A
Markov chain Monte Carlo sampling is performed based on the
[Metroplis-Hastings
criterion](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm)
.

Analysis
--------

The `Analysis`{.interpreted-text role="func"} is consist of two parts:
1/ using [MBAR](https://pymbar.readthedocs.io/en/master/index.html)
algorithm to compute populations and
`BICePs scores <theory>`{.interpreted-text role="ref"} and 2/ plot the
figures to show population and
`nuisance parameters <theory>`{.interpreted-text role="ref"}.
