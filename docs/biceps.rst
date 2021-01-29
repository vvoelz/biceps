.. _api:


API Reference
=============

Preparation
-----------

.. currentmodule:: biceps
.. autoclass:: biceps.Restraint.Preparation
.. automethod:: biceps.Restraint.Preparation.prep_cs
.. automethod:: biceps.Restraint.Preparation.prep_J
.. automethod:: biceps.Restraint.Preparation.prep_noe
.. automethod:: biceps.Restraint.Preparation.prep_pf


Ensemble
-----------

.. currentmodule:: biceps
.. autoclass:: biceps.Ensemble
.. automethod:: biceps.Ensemble.initialize_restraints
.. automethod:: biceps.Ensemble.to_list



Restraint
---------

.. currentmodule:: biceps
.. autoclass:: biceps.Restraint.Restraint
.. automethod:: biceps.Restraint.Restraint_cs.init_restraint
.. automethod:: biceps.Restraint.Restraint_J.init_restraint
.. automethod:: biceps.Restraint.Restraint_noe.init_restraint
.. automethod:: biceps.Restraint.Restraint_pf.init_restraint



PosteriorSampler
----------------

.. currentmodule:: biceps
.. autoclass:: biceps.PosteriorSampler
.. automethod:: biceps.PosteriorSampler.neglogP
.. automethod:: biceps.PosteriorSampler.sample
.. autoclass:: biceps.PosteriorSamplingTrajectory
.. automethod:: biceps.PosteriorSamplingTrajectory.process_results


.. raw:: html

    <table>
    <caption>Trajectory information</caption>
    <thead>
    <tr class="header">
    <th style="text-align: left;"><strong>Key</strong></th>
    <th style="text-align: left;"><strong>Short Description</strong></th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td style="text-align: left;"><code><a>rest_type</a></code></td>
    <td style="text-align: left;">list of strings for each restraint type.</td>
    </tr>
    <tr class="even">
    <td style="text-align: left;"><code><a>ref</a></code></td>
    <td style="text-align: left;">list of strings for each reference potential types</td>
    </tr>
    <tr class="odd">
    <td style="text-align: left;"><code>allowed_parameters</code></td>
    <td style="text-align: left;">list of numpy arrays containing the allowed range of nuisance parameters with shape (m,n)</td>
    </tr>
    <tr class="even">
    <td style="text-align: left;"><code>sampled_parameters</code></td>
    <td style="text-align: left;">list of numpy arrays containing the counts of nuisance parameters sampled for each restraint  with shape (m,n)</td>
    </tr>
    <tr class="odd">
    <td style="text-align: left;"><code>trajectory_headers</code></td>
    <td style="text-align: left;">e.g., [step, energy, accept, state, [nuisance parameter index]]</td>
    </tr>
    <tr class="even">
    <td style="text-align: left;"><code>trajectory</code></td>
    <td style="text-align: left;">list of values—see <code>trajectory_headers</code></td>
    </tr>
    <tr class="odd">
    <td style="text-align: left;"><code>sep_accept</code></td>
    <td style="text-align: left;">list of separated acceptance ratios with shape (n+1,)</td>
    </tr>
    <tr class="even">
    <td style="text-align: left;"><code>traces</code></td>
    <td style="text-align: left;">list of sampled nuisance parameters with shape (n)</td>
    </tr>
    <tr class="odd">
    <td style="text-align: left;"><code>state_trace</code></td>
    <td style="text-align: left;">list of sampled conformational state index</td>
    </tr>
    </tbody>
    </table>
    <p style="font-size: 10pt">n is the number of allowed parameters<br />
    m is the number of restraints<br />
    </p>



Analysis
--------

.. currentmodule:: biceps
.. autoclass:: biceps.Analysis
.. automethod:: biceps.Analysis.plot


Convergence
------------

.. currentmodule:: biceps
.. autoclass:: biceps.Convergence
.. automethod:: biceps.Convergence.plot_traces
.. automethod:: biceps.Convergence.plot_auto_curve
.. automethod:: biceps.Convergence.plot_block_avg
.. automethod:: biceps.Convergence.get_autocorrelation_curves
.. automethod:: biceps.Convergence.process


toolbox
-------

.. currentmodule:: biceps
.. automethod:: biceps.toolbox.sort_data
.. automethod:: biceps.toolbox.get_files
.. automethod:: biceps.toolbox.list_res
.. automethod:: biceps.toolbox.list_possible_restraints
.. automethod:: biceps.toolbox.list_extensions
.. automethod:: biceps.toolbox.list_possible_extensions
.. automethod:: biceps.toolbox.npz_to_DataFrame
.. automethod:: biceps.toolbox.save_object




