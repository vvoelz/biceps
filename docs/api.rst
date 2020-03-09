.. _api:


API Reference
=============

Preparation
-----------
.. currentmodule:: biceps

.. autoclass:: Preparation

   .. rubric:: Methods

   .. autosummary::

       ~Preparation.write

Observables
-----------
.. currentmodule:: biceps

.. autoclass:: NMR_Chemicalshift

   .. rubric:: Methods

   .. autosummary::

       ~Observable.NMR_Chemicalshift.__init__

.. autoclass:: NMR_Dihedral

   .. rubric:: Methods

   .. autosummary::

       ~Observable.NMR_Dihedral.__init__

.. autoclass:: NMR_Distance

   .. rubric:: Methods

   .. autosummary::

       ~Observable.NMR_Distance.__init__


.. autoclass:: NMR_Protectionfactor

   .. rubric:: Methods

   .. autosummary::

       ~Observable.NMR_Protectionfactor.__init__




Restraint
---------
.. currentmodule:: biceps

.. autoclass:: Restraint


   .. rubric:: Methods

   .. autosummary::

       ~Restraint.load_data
       ~Restraint.add_restraint
       ~Restraint.compute_see
       ~Restraint.compute_neglog_exp_ref
       ~Restraint.compute_neglog_gaussian_ref

.. autoclass:: Restraint_cs_Ca

   .. rubric:: Methods

   .. autosummary::

       ~Restraint_cs_Ca.prep_observable

.. autoclass:: Restraint_cs_H

   .. rubric:: Methods

   .. autosummary::

       ~Restraint_cs_H.prep_observable

.. autoclass:: Restraint_cs_Ha

   .. rubric:: Methods

   .. autosummary::

       ~Restraint_cs_Ha.prep_observable

.. autoclass:: Restraint_cs_N


   .. rubric:: Methods

   .. autosummary::

       ~Restraint_cs_N.prep_observable

.. autoclass:: Restraint_cs_J


   .. rubric:: Methods

   .. autosummary::

       ~Restraint_cs_J.prep_observable
       ~Restraint_J.adjust_weights

.. autoclass:: Restraint_cs_noe


   .. rubric:: Methods

   .. autosummary::

       ~Restraint_cs_noe.prep_observable
       ~Restraint_noe.adjust_weights

.. autoclass:: Restraint_cs_pf


   .. rubric:: Methods

   .. autosummary::

       ~Restraint_cs_pf.prep_observable



PosteriorSampler
----------------
.. currentmodule:: biceps

.. autoclass:: PosteriorSampler

   .. rubric:: Methods

   .. autosummary::

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

   .. rubric:: Methods

   .. autosummary::

       ~PosteriorSamplingTrajectory.process

Analysis
--------
.. currentmodule:: biceps

.. autoclass:: Analysis

   .. rubric:: Methods

   .. autosummary::

       ~Analysis.list_scheme
       ~Analysis.plot
       ~Analysis.load_data
       ~Analysis.MBAR_analysis
       ~Analysis.save_MBAR

Convergence
------------

.. currentmodule:: biceps

.. autofunction:: autocorrelation

.. autoclass:: Convergence

   .. rubric:: Methods

   .. autosummary::

       ~Convergence.get_sampled_parameters
       ~Convergence.get_labels
       ~Convergence.plot_traces
       ~Convergence.plot_auto_curve
       ~Convergence.single_exp_decay
       ~Convergence.double_exp_decay
       ~Convergence.exponential_fit
       ~Convergence.process
       ~Convergence.plot_block_avg
       ~Convergence.compute_JSD
       ~Convergence.plot_JSD_conv



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


