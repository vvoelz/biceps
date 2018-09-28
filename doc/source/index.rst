
BICePs - Bayesian Inference of Conformational Populations
===========================================================

The BICePs algorithm (Bayesian Inference of Conformational Populations) is a statistically rigorous Bayesian inference method to reconcile theoretical predictions of conformational state populations with sparse and/or noisy experimental measurements and objectively compare different models.
Supported experimental observables include:
 - `NMR nuclear Overhauser effect (NOE) <https://en.wikipedia.org/wiki/Nuclear_Overhauser_effect>`_.
 - `NMR chemical shifts <https://en.wikipedia.org/wiki/Chemical_shift>`_ (``HA``, ``NH``, ``CA`` and ``N``).
 - `J couplings <https://en.wikipedia.org/wiki/J-coupling>`_ (both small molecules and amino acids)
 - `Hydrogen–deuterium exchange (HDX) <https://en.wikipedia.org/wiki/Hydrogen–deuterium_exchange>`_.

All source codes and examples are available here: `GitHub Repository for BICePs <https://github.com/vvoelz/biceps/>`_

---------------------------------------------

.. toctree::
   :maxdepth: 2
   :glob:

   installation
   theory
   workflow
   examples/index
   examples/albo/*
   versions/index



Citation |DOI for Citing BICePs|
--------------------------------

Please apply BICePs in your research and cite it in any scientific publications.

::

    @article{VAV-2018,
        title = {Model selection using BICePs: A Bayesian approach to force
        field validation and parameterization},
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
