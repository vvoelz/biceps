*********************************************************
BICePs - Bayesian Inference of Conformational Populations
*********************************************************

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

    <hr style="height:2.5px">
    <h3 style="align: justify;font-size: 10pt">



.. toctree::
   :maxdepth: 4
   :glob:

   installation
   theory
   tutorials_and_examples
   biceps
   </h3>


Citation |DOI for Citing BICePs|
--------------------------------

Please apply BICePs in your research and cite it in any scientific publications.

::

    @article{raddi2023biceps,
        title={Biceps v2. 0: software for ensemble reweighting using Bayesian inference of conformational populations},
        author={Raddi, Robert M and Ge, Yunhui and Voelz, Vincent A},
        journal={Journal of chemical information and modeling},
        volume={63},
        number={8},
        pages={2370--2381},
        year={2023},
        publisher={ACS Publications},
        doi = {10.1021/acs.jcim.2c01296}
}

    @article{VAV-2018,
        title = {Model selection using BICePs: A Bayesian approach to forcefield validation and parameterization},
        author = {Yunhui Ge and Vincent A. Voelz},
        journal = {Journal of Physical Chemistry B},
        volume = {122},
        number = {21},
        pages = {5610 -- 5622},
        year = {2018},
        doi = {10.1021/acs.jpcb.7b11871}
    }


License
-------



.. |DOI for Citing BICePs| image:: https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2022--1b24c-green.svg
   :target: https://doi.org/10.26434/chemrxiv-2022-1b24c

.. vim: tw=75
