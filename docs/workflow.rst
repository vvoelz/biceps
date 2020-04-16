Workflow
========

.. raw:: html

    <hr style="height:2.5px">

    <h3 style="align: justify;font-size: 12pt">A typical <code>biceps</code> procedure
    includes four core steps: <code><a href="biceps.html#preparation">biceps.Restraint.Preparation</a></code>,
    <code><a href="biceps.html#ensemble">biceps.Ensemble</a></code>,
    <code><a href="biceps.html#posteriorsampler">biceps.PosteriorSampler</a></code> and
    <code><a href="biceps.html#analysis">biceps.Analysis</a></code>.
    </h3>


Preparation
-----------

.. raw:: html

    <h3 style="align: justify;font-size: 12pt">The
    <code><a href="biceps.html#preparation">biceps.Restraint.Preparation</a></code> class
    converts all raw data to BICePs readable format. It asks for experimental
    data as well as correspondingly precomputed experimental observables from simulation.
    We recommend users to use <a href="http://mdtraj.org">MDTraj</a> to compute all
    necessary experimental quantities from simulation or use our prepared
    functions in <code><a href="biceps.html#toolbox">biceps.toolbox</a></code>.
    Check more details in the <a href="examples/index.html">examples</a> page.
    </h3>


Ensemble
---------

.. raw:: html

    <h3 style="align: justify;font-size: 12pt">
    The <code><a href="biceps.html#ensemble">biceps.Ensemble</a></code> class
    was built as a container class for all the <code><a href="biceps.html#restraint">
    biceps.Restraint</a></code> objects. Initialize all the restraints in a single quick-n-easy
    step with <code><a href="biceps.Ensemble.initialize_restraints"></code>. Upon
    recieving the neessary parameters, the <b>input_data</b> argument is used to automatically
    call the corresponding <code><a href="biceps.html#restraint">biceps.Restraint</a></code>
    classes to build the ensemble.
    </h3>


PosteriorSampler
----------------

.. raw:: html

    <h3 style="align: justify;font-size: 12pt">
    The <code><a href="biceps.html#posteriorsampler">biceps.PosteriorSampler</a></code>
    class only requires a list of <code><a href="biceps.html#restraint">biceps.Restraint</a></code>
    objects. Fortunately, we can convert the <a href="#Ensemble">Ensemble</a> to a list
    using <code><a href="biceps.html#biceps.Ensemble.to_list">biceps.Ensemble.to_list()</a></code>. Thus,
    the <a href="#PosteriorSampler">PosteriorSampler</a> works very closely with the
    <code><a href="biceps.html#restraint">biceps.Restraint</a></code> classes.
    Sampling (<code><a href="biceps.html#biceps.PosteriorSampler.sample">sample</a></code>)
    via Markov chain Monte Carlo sampling is performed and uses the
    <a hre="https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm">Metroplis-Hastings criterion </a>.
    </h3>


Analysis
--------

.. raw:: html

    <h3 style="align: justify;font-size: 12pt">
    The <code><a href="biceps.html#analysis">biceps.Analysis</a></code> is consist of two parts:
    <br>
    1. Using <code><a href="https://pymbar.readthedocs.io/en/master/index.html">MBAR</a></code>
    algorithm to compute populations and <code><a href="theory.html">BICePs scores</a></code>.
    <br>
    2. Plot the figures to show population and <code><a href="theory.html">nuisance parameters</a></code>.
    </h3>
    <hr>


Convergence
-----------

.. raw:: html

    <h3 style="align: justify;font-size: 12pt">
    The <code><a href="biceps.html#convergence">biceps.Convergence</a></code> class
    is separate from the workflow requirements. Convergence tests are able to be
    performed using MCMC trajectories from <a href="#PosteriorSampler">PosteriorSampler</a>.
    </h3>





