Workflow
========

.. raw:: html

    <hr style="height:2.5px">

    <h3 style="align: justify;font-size: 12pt">A typical BICePs sampling
    includes four core steps: <code><a href="api.html#preparation">biceps.Preparation</a></code>,
    <code><a href="api.html#restraint">biceps.Restraint</a></code>,
    <code><a href="api.html#posteriorsampler">biceps.PosteriorSampler</a></code> and
    <code><a href="api.html#analysis">biceps.Analysis</a></code>.
    </h3>



Preparation
-----------

.. raw:: html

    <h3 style="align: justify;font-size: 12pt">The
    <code><a href="api.html#preparation">biceps.Preparation</a></code> class
    converts all raw data to  BICePs readable format. It asks for experimental
    data as well as correspondingly precomputed experimental observables from simulation.
    We recommend users to use <a href="http://mdtraj.org">MDTraj</a> to compute all
    necessary experimental quantities from simulation or use our prepared
    functions in the <code><a href="api.html#toolbox">toolbox</a></code>.
    Check more details in the <a href="examples/index.html">examples</a> page.
    </h3>



Restraint
---------

.. raw:: html

    <h3 style="align: justify;font-size: 12pt">
    The <code><a href="api.html#restraint">biceps.Restraint</a></code> class
    initializes all necessary functions to construct numpy array containing
    information for BICePs sampling. As a parent class, it also includes child
    classes for different experimental restraints.
    </h3>


PosteriorSampler
----------------

.. raw:: html

    <h3 style="align: justify;font-size: 12pt">
    The <code><a href="api.html#posteriorsampler">biceps.PosteriorSampler</a></code>
    class is closely working with the
    <code><a href="api.html#restraint">biceps.Restraint</a></code> class.
    A Markov chain Monte Carlo sampling is performed based on the
    <a hre="https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm">Metroplis-Hastings criterion </a>.
    </h3>


Analysis
--------

.. raw:: html

    <h3 style="align: justify;font-size: 12pt">
    The <code><a href="api.html#analysis">biceps.Analysis</a></code> is consist of two parts:
    <br>
    1. Using <code><a href="https://pymbar.readthedocs.io/en/master/index.html">MBAR</a></code>
    algorithm to compute populations and <code><a href="theory.html">BICePs scores</a></code>.
    <br>
    2. Plot the figures to show population and <code><a href="theory.html">nuisance parameters</a></code>.

    </h3>





