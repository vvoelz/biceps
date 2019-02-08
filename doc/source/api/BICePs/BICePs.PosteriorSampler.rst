.. raw:: html

   <div class="wy-grid-for-nav">

.. raw:: html

   <div class="wy-side-scroll">

.. raw:: html

   <div class="wy-side-nav-search">

`BICePs <../index.html>`__

.. raw:: html

   <div class="version">

2.0-beta

.. raw:: html

   </div>

.. raw:: html

   <div role="search">

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div class="wy-menu wy-menu-vertical" data-spy="affix"
   role="navigation" aria-label="main navigation">

-  `Installation <../installation.html>`__
-  `Theory <../theory.html>`__
-  `Workflow <../workflow.html>`__
-  `Examples <../examples/index.html>`__
-  `Reference <../reference/index.html>`__

   -  ```BICePs`` <BICePs.html>`__

      -  `Submodules <BICePs.html#submodules>`__

         -  ```BICePs.Analysis`` <BICePs.Analysis.html>`__
         -  ```BICePs.J_coupling`` <BICePs.J_coupling.html>`__
         -  ```BICePs.KarplusRelation`` <BICePs.KarplusRelation.html>`__
         -  ```BICePs.Observable`` <BICePs.Observable.html>`__
         -  ```BICePs.PosteriorSampler`` <#>`__
         -  ```BICePs.Preparation`` <BICePs.Preparation.html>`__
         -  ```BICePs.Restraint`` <BICePs.Restraint.html>`__
         -  ```BICePs.init_res`` <BICePs.init_res.html>`__
         -  ```BICePs.prep_J`` <BICePs.prep_J.html>`__
         -  ```BICePs.prep_cs`` <BICePs.prep_cs.html>`__
         -  ```BICePs.prep_noe`` <BICePs.prep_noe.html>`__
         -  ```BICePs.prep_pf`` <BICePs.prep_pf.html>`__
         -  ```BICePs.toolbox`` <BICePs.toolbox.html>`__

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div class="section wy-nav-content-wrap" data-toggle="wy-nav-shift">

 `BICePs <../index.html>`__

.. raw:: html

   <div class="wy-nav-content">

.. raw:: html

   <div class="rst-content">

.. raw:: html

   <div role="navigation" aria-label="breadcrumbs navigation">

-  `Docs <../index.html>`__ »
-  `Reference <../reference/index.html>`__ »
-  ```BICePs`` <BICePs.html>`__ »
-  ``BICePs.PosteriorSampler``
-  `View page
   source <../_sources/BICePs/BICePs.PosteriorSampler.rst.txt>`__

--------------

.. raw:: html

   </div>

.. raw:: html

   <div class="document" role="main" itemscope="itemscope"
   itemtype="http://schema.org/Article">

.. raw:: html

   <div itemprop="articleBody">

.. raw:: html

   <div id="module-BICePs.PosteriorSampler" class="section">

.. rubric:: ``BICePs.PosteriorSampler``\ `¶ <#module-BICePs.PosteriorSampler>`__
   :name: biceps.posteriorsampler

.. raw:: html

   <div id="contents" class="contents local topic">

-  `Classes <#classes>`__

.. raw:: html

   </div>

 *class* ``BICePs.PosteriorSampler.``\ ``PosteriorSampler``\ (\ *ensemble*, *freq_write_traj=1000*, *freq_print=1000*, *freq_save_traj=100*\ )\ `¶ <#BICePs.PosteriorSampler.PosteriorSampler>`__
   A class to perform posterior sampling of conformational populations

   ensemble: list
      a list of lists of Restraint objects, one list for each
      conformation.
   freq_write_traj: int
      the frequency (in steps) to write the MCMC trajectory
   freq_print: int
      the frequency (in steps) to print status
   freq_save_traj: int
      the frequency (in steps) to store the MCMC trajectory

   Initialize PosteriorSampler Class.

    ``build_exp_ref``\ (\ *rest_index*, *verbose=False*\ )\ `¶ <#BICePs.PosteriorSampler.PosteriorSampler.build_exp_ref>`__
      Look at all the structures to find the average observables r_j

      >> beta_j =
      np.array(distributions[j]).sum()/(len(distributions[j])+1.0)

      then store this reference potential info for all Restraints of
      this type for each structure

    ``build_gaussian_ref``\ (\ *rest_index*, *use_global_ref_sigma=False*, *verbose=False*\ )\ `¶ <#BICePs.PosteriorSampler.PosteriorSampler.build_gaussian_ref>`__
      Look at all the structures to find the mean (mu) and std (sigma)
      of observables r_j then store this reference potential info for
      all Restraints of this type for each structure

    ``compile_nuisance_parameters``\ (\ *verbose=False*\ )\ `¶ <#BICePs.PosteriorSampler.PosteriorSampler.compile_nuisance_parameters>`__
      Compiles arrays into a list for each nuisance parameter.

      [[allowed_sigma_cs_H],[allowed_sigma_noe,allowed_gamma_noe],…,[Nth_restraint]]

    ``compute_logZ``\ ()\ `¶ <#BICePs.PosteriorSampler.PosteriorSampler.compute_logZ>`__
      Compute reference state logZ for the free energies to normalize.

    ``neglogP``\ (\ *new_state*, *parameters*, *parameter_indices*, *verbose=True*\ )\ `¶ <#BICePs.PosteriorSampler.PosteriorSampler.neglogP>`__
      Return -ln P of the current configuration.

      new_state: int
         the new conformational state from Sample()
      parameters: list
         a list of the new parameters for each of the restraints
      parameter_indices: list
         a list of the new indices for each of the parameters

      Energy
         the energy

    ``sample``\ (\ *nsteps*, *verbose=True*\ )\ `¶ <#BICePs.PosteriorSampler.PosteriorSampler.sample>`__
      Perform n number of steps (nsteps) of posterior sampling, where
      Monte Carlo moves are accepted or rejected according to Metroplis
      criterion.

 *class* ``BICePs.PosteriorSampler.``\ ``PosteriorSamplingTrajectory``\ (\ *ensemble*\ )\ `¶ <#BICePs.PosteriorSampler.PosteriorSamplingTrajectory>`__
   A container class to store and perform operations on the trajectories
   of sampling runs.

   Initialize the PosteriorSamplingTrajectory container class.

    ``process``\ ()\ `¶ <#BICePs.PosteriorSampler.PosteriorSamplingTrajectory.process>`__
      Process the trajectory, computing sampling statistics,
      ensemble-average NMR observables.

    ``read_results``\ (\ *filename*\ )\ `¶ <#BICePs.PosteriorSampler.PosteriorSamplingTrajectory.read_results>`__
      Reads a npz file

    ``write_results``\ (\ *outfilename='traj.npz'*\ )\ `¶ <#BICePs.PosteriorSampler.PosteriorSamplingTrajectory.write_results>`__
      Writes a compact file of several arrays into binary format.
      Standardized: Yes ; Binary: Yes; Human Readable: No;

.. raw:: html

   <div id="classes" class="section">

.. rubric:: `Classes <#id1>`__\ `¶ <#classes>`__
   :name: classes

-  ```PosteriorSampler`` <#BICePs.PosteriorSampler.PosteriorSampler>`__:
   A class to perform posterior sampling of conformational populations
-  ```PosteriorSamplingTrajectory`` <#BICePs.PosteriorSampler.PosteriorSamplingTrajectory>`__:
   A container class to store and perform operations on the trajectories
   of

 *class* ``BICePs.PosteriorSampler.``\ ``PosteriorSampler``\ (\ *ensemble*, *freq_write_traj=1000*, *freq_print=1000*, *freq_save_traj=100*\ )
   A class to perform posterior sampling of conformational populations

   ensemble: list
      a list of lists of Restraint objects, one list for each
      conformation.
   freq_write_traj: int
      the frequency (in steps) to write the MCMC trajectory
   freq_print: int
      the frequency (in steps) to print status
   freq_save_traj: int
      the frequency (in steps) to store the MCMC trajectory

   Initialize PosteriorSampler Class.

   Inheritance

   digraph inheritancecf9a2b349f { rankdir=LR; size="8.0, 12.0";
   "PosteriorSampler"
   [URL="#BICePs.PosteriorSampler.PosteriorSampler",fontname="Vera Sans,
   DejaVu Sans, Liberation Sans, Arial, Helvetica,
   sans",fontsize=10,height=0.25,shape=box,style="setlinewidth(0.5)",target="_top",tooltip="A
   class to perform posterior sampling of conformational populations"];
   }

    ``build_exp_ref``\ (\ *rest_index*, *verbose=False*\ )
      Look at all the structures to find the average observables r_j

      >> beta_j =
      np.array(distributions[j]).sum()/(len(distributions[j])+1.0)

      then store this reference potential info for all Restraints of
      this type for each structure

    ``build_gaussian_ref``\ (\ *rest_index*, *use_global_ref_sigma=False*, *verbose=False*\ )
      Look at all the structures to find the mean (mu) and std (sigma)
      of observables r_j then store this reference potential info for
      all Restraints of this type for each structure

    ``compile_nuisance_parameters``\ (\ *verbose=False*\ )
      Compiles arrays into a list for each nuisance parameter.

      [[allowed_sigma_cs_H],[allowed_sigma_noe,allowed_gamma_noe],…,[Nth_restraint]]

    ``compute_logZ``\ ()
      Compute reference state logZ for the free energies to normalize.

    ``neglogP``\ (\ *new_state*, *parameters*, *parameter_indices*, *verbose=True*\ )
      Return -ln P of the current configuration.

      new_state: int
         the new conformational state from Sample()
      parameters: list
         a list of the new parameters for each of the restraints
      parameter_indices: list
         a list of the new indices for each of the parameters

      Energy
         the energy

    ``sample``\ (\ *nsteps*, *verbose=True*\ )
      Perform n number of steps (nsteps) of posterior sampling, where
      Monte Carlo moves are accepted or rejected according to Metroplis
      criterion.

 *class* ``BICePs.PosteriorSampler.``\ ``PosteriorSamplingTrajectory``\ (\ *ensemble*\ )
   A container class to store and perform operations on the trajectories
   of sampling runs.

   Initialize the PosteriorSamplingTrajectory container class.

   Inheritance

   digraph inheritancedb5176b91b { rankdir=LR; size="8.0, 12.0";
   "PosteriorSamplingTrajectory"
   [URL="#BICePs.PosteriorSampler.PosteriorSamplingTrajectory",fontname="Vera
   Sans, DejaVu Sans, Liberation Sans, Arial, Helvetica,
   sans",fontsize=10,height=0.25,shape=box,style="setlinewidth(0.5)",target="_top",tooltip="A
   container class to store and perform operations on the trajectories
   of"]; }

    ``process``\ ()
      Process the trajectory, computing sampling statistics,
      ensemble-average NMR observables.

    ``read_results``\ (\ *filename*\ )
      Reads a npz file

    ``write_results``\ (\ *outfilename='traj.npz'*\ )
      Writes a compact file of several arrays into binary format.
      Standardized: Yes ; Binary: Yes; Human Readable: No;

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div class="rst-footer-buttons" role="navigation"
   aria-label="footer navigation">

`Next <BICePs.Preparation.html>`__ `Previous <BICePs.Observable.html>`__

.. raw:: html

   </div>

--------------

.. raw:: html

   <div role="contentinfo">

© Copyright 2018, Temple University

.. raw:: html

   </div>

Built with `Sphinx <http://sphinx-doc.org/>`__ using a
`theme <https://github.com/rtfd/sphinx_rtd_theme>`__ provided by `Read
the Docs <https://readthedocs.org>`__.

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   </div>
