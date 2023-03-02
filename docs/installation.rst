.. _installation::

Installation
============

.. raw:: html

    <hr style="height:2.5px">

.. raw:: html

    <h3 style="align: justify;font-size: 12pt"> You can install BICePs from
    <a href="https://pypi.org/project/biceps/">PyPI</a> using
    <code>pip</code>:<h3>

    <div class="highlight-default notranslate">
        <div class="highlight">
        <pre>$ pip install biceps</pre>
        </div>
    </div>

    <h3 style="align: justify;font-size: 12pt"> Coming soon: </h3>

    <div class="highlight-default notranslate">
        <div class="highlight">
        <pre>$ conda install -c conda-forge biceps</pre>
        </div>
    </div>




.. raw:: html

    <h3 style="text-align: justify;font-size: 10pt">
    Conda is a cross-platform package manager built especially for scientific
    python. It will install <code>biceps</code> along with all dependencies from a
    pre-compiled binary. If you don't have Python or the <b>Anaconda</b> package
    manager, we recommend starting with the
    <a href="https://store.continuum.io/cshop/anaconda/">
    Anaconda Scientific Python distribution</a>, which comes
    pre-packaged with many of the core scientific python packages that biceps
    uses (see below), or with the <a href="http://conda.pydata.org/miniconda.html">
    <b>Miniconda</b> Python distribution</a>, which is a bare-bones Python installation.
    </h3>
    <!--<a href=""> </a>-->
    <br>
    <h3 style="text-align: justify;font-size: 12pt">
    <a class="github-button" href="https://github.com/vvoelz/biceps/" data-size="large" data-show-count="false" aria-label="BICePs">GitHub</a><script async defer src="https://buttons.github.io/buttons.js"></script> Take a look at our repository, peruse through our source code and submit issues.</h3>



Dependencies
------------

.. raw:: html

   <ul>
    <li style="list-style-type: none;"><a href="https://pymbar.readthedocs.io">pymbar</a> >= 4.0.1</li>
    <li style="list-style-type: none;"><a href="https://mdtraj.org">mdtraj</a></li>
    Python versions 2.7-3.7)</li>
   </ul>

Testing Your Installation
-------------------------

Coming soon.


Versions
--------

.. raw:: html

   <ul>
    <li style="list-style-type: none;">
    <a href="https://github.com/vvoelz/biceps/releases/tag/v1.0">Version 1.0</a>: This release is non-production ready. This release contains archived scripts for various systems.</li>
    <li style="list-style-type: none;">Version 2.0: Redesigned/generalized source
    code with convergence submodule for checking MCMC trajectories. Optional
    multiprocessing functionality for running simulations for each lambda value
    in parallel. Additional experimental observables include hydrogen–deuterium
    exchange (HDX).</li>
   </ul>


.. vim: tw=75
