.. _api:


API Reference
=============

Preparation
-----------
.. currentmodule:: biceps

.. autoclass:: Preparation


   .. automethod:: __init__


   .. rubric:: Methods

   .. autosummary::

       ~Preparation.__init__
       ~Preparation.write

Observables
-----------
.. currentmodule:: biceps

.. autoclass:: Observable

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

       ~Observable.__init__

.. autoclass:: NMR_Chemicalshift

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

       ~NMR_Chemicalshift.__init__

.. autoclass:: NMR_Dihedral

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::


       ~NMR_Dihedral.__init__
       ~NMR_Distance.__init__
       ~NMR_Protectionfactor.__init__



Restraint
---------
.. currentmodule:: biceps

.. autoclass:: Restraint

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

       ~Restraint.__init__
       ~Restraint.load_data
       ~Restraint.add_restraint
       ~Restraint.compute_see
       ~Restraint.compute_neglog_exp_ref
       ~Restraint.compute_neglog_gaussian_ref

.. autoclass:: Restraint_cs_Ca

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

       ~Restraint_cs_Ca.__init__
       ~Restraint_cs_Ca.prep_observable

.. autoclass:: Restraint_cs_H

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

       ~Restraint_cs_H.__init__
       ~Restraint_cs_H.prep_observable

.. autoclass:: Restraint_cs_Ha

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

       ~Restraint_cs_Ha.__init__
       ~Restraint_cs_Ha.prep_observable

.. autoclass:: Restraint_cs_N

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

       ~Restraint_cs_N.__init__
       ~Restraint_cs_N.prep_observable

.. autoclass:: Restraint_cs_J

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

       ~Restraint_cs_J.__init__
       ~Restraint_cs_J.prep_observable
       ~Restraint_J.adjust_weights

.. autoclass:: Restraint_cs_noe

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

       ~Restraint_cs_noe.__init__
       ~Restraint_cs_noe.prep_observable
       ~Restraint_noe.adjust_weights

.. autoclass:: Restraint_cs_pf

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

       ~Restraint_cs_pf.__init__
       ~Restraint_cs_pf.prep_observable



PosteriorSampler
----------------
.. currentmodule:: biceps

.. autoclass:: PosteriorSampler

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

       ~PosteriorSampler.__init__
       ~PosteriorSampler.compute_logZ
       ~PosteriorSampler.build_exp_ref
       ~PosteriorSampler.build_gaussian_ref
       ~PosteriorSampler.compile_nuisance_parameters
       ~PosteriorSampler.sample
       ~PosteriorSampler.neglogP
       ~PosteriorSampler.logspaced_array
       ~PosteriorSampler.write_results
       ~PosteriorSampler.read_results


.. autoclass:: PosteriorSamplingTrajectory

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

       ~PosteriorSamplingTrajectory.__init__
       ~PosteriorSamplingTrajectory.process

Analysis
--------
.. currentmodule:: biceps

.. autoclass:: Analysis

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

       ~Analysis.__init__
       ~Analysis.list_scheme
       ~Analysis.plot
       ~Analysis.load_data
       ~Analysis.MBAR_analysis
       ~Analysis.save_MBAR


Analysis
--------
.. currentmodule:: biceps

.. autoclass:: Analysis

   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::

       ~Analysis.__init__
       ~Analysis.list_scheme
       ~Analysis.plot
       ~Analysis.load_data
       ~Analysis.MBAR_analysis
       ~Analysis.save_MBAR


toolbox
-------
.. currentmodule:: biceps

.. autofunction:: sort_data
.. autofunction:: list_res
.. autofunction:: write_results
.. autofunction:: read_results
.. autofunction:: convert_pop_to_energy
.. autofunction:: get_J3_HN_HA
.. autofunction:: dihedral_angle
.. autofunction:: compute_nonaa_Jcoupling
.. autofunction:: plot_ref
.. autofunction:: get_rest_type
.. autofunction:: get_allowed_parameters
.. autofunction:: autocorr_valid
.. autofunction:: compute_ac
.. autofunction:: plot_ac
.. autofunction:: compute_JSD
.. autofunction:: plot_conv
































