BICePs - Bayesian Inference of Conformational Populations
===========================================================

.. raw:: html

    <hr style="height:2.5px">

.. raw:: html

    <h3 style="align: justify;font-size: 12pt">The BICePs algorithm (Bayesian
    Inference of Conformational Populations) is a statistically rigorous
    Bayesian inference method to reconcile theoretical predictions of
    conformational state populations with sparse and/or noisy experimental
    measurements and objectively compare different models.
    Supported experimental observables include:
    <br>
    <br>
    <ul>
        <li style="list-style-type: none;font-size: 12pt;">
            <a href="https://en.wikipedia.org/wiki/Nuclear_Overhauser_effect">NMR nuclear Overhauser effect (NOE)</a>
        </li>
        <li style="list-style-type: none;font-size: 12pt;">
            <a href="https://en.wikipedia.org/wiki/Chemical_shift">NMR chemical shifts</a> (HA, NH, CA and N)
        </li>
        <li style="list-style-type: none;font-size: 12pt;">
            <a href="https://en.wikipedia.org/wiki/J-coupling">J couplings</a> (both small molecules and amino acids)
        </li>
        <li style="list-style-type: none;font-size: 12pt;">
            <a href="https://en.wikipedia.org/wiki/Hydrogen–deuterium_exchange">Hydrogen–deuterium exchange (HDX)</a>
        </li>
    </ul>
    <br>
    All raw source code and examples/data are available <a href="https://github.com/vvoelz/biceps/">here</a>.</h3>
    <hr style="height:2.5px">
    <h3 style="align: justify;font-size: 10pt">

.. toctree::
   :maxdepth: 4
   :glob:

   installation
   theory
   workflow
   examples/index
   biceps
   </h3>


Citation |DOI for Citing BICePs|
--------------------------------

Please apply BICePs in your research and cite it in any scientific publications.

::

    @article{VAV-2018,
        title = {Model selection using BICePs: A Bayesian approach to forcefield validation and parameterization},
        author = {Yunhui Ge and Vincent A. Voelz},
        journal = {Journal of Physical Chemistry B},
        volume = {122},
        number = {21},
        pages = {5610 -- 5622},
        year = {2018},
        doi = {doi:10.1021/acs.jpcb.7b11871}
    }


License
-------



.. |DOI for Citing BICePs| image:: https://img.shields.io/badge/DOI-10.1021.acs.jpcb.7b11871-green.svg
   :target: http://doi.org/10.1021/acs.jpcb.7b11871

.. vim: tw=75
