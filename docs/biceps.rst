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

Observable
-----------
.. currentmodule:: biceps

.. automodule:: biceps.Observable

   .. contents::
      :local:

.. currentmodule:: biceps.Observable

   .. autosummary::
.. autoclass:: Observable.NMR_Chemicalshift
.. autoclass:: Observable.NMR_Dihedral
.. autoclass:: Observable.NMR_Distance
.. autoclass:: Observable.NMR_Protectionfactor

Restraint
---------

.. currentmodule:: biceps

.. automodule:: biceps.Restraint

   .. contents::
      :local:

.. currentmodule:: biceps.Restraint

   .. autosummary::
.. autoclass:: Restraint.Restraint_cs_Ca
.. autoclass:: Restraint.Restraint_cs_H
.. autoclass:: Restraint.Restraint_cs_Ha
.. autoclass:: Restraint_Restraint_N
.. autoclass:: Restraint_Restraint_J
.. autoclass:: Restraint_Restraint_noe
.. autoclass:: Restraint_Restraint_PF




PosteriorSampler
----------------

.. currentmodule:: biceps

.. automodule:: biceps.PosteriorSampler

   .. contents::
      :local:

.. currentmodule:: biceps.PosteriorSampler

   .. rubric:: Methods

   .. autosummary::

.. currentmodule:: biceps.PosteriorSampler


.. automodule:: biceps.PosteriorSamplingTrajectory

   .. contents::
      :local:

.. currentmodule:: biceps.PosteriorSamplingTrajectory

   .. rubric:: Methods

   .. autosummary::

.. currentmodule:: biceps.PosteriorSamplingTrajectory



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

.. automodule:: biceps.toolbox

   .. contents::
      :local:

.. currentmodule:: biceps.toolbox

   .. autosummary::

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




