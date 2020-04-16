.. _tutorials and examples:


********************
Tutorials & Examples
********************


.. raw:: html

    <h3 style="align: justify;font-size: 12pt"># <span
    style="color:red;">NOTE</span>: Each of of the following jupyter notebooks can be downloaded
    <a href="https://github.com/vvoelz/biceps/">here</a>.</h3>
    <hr style="height:2.5px">

.. raw:: html

    <h1 style="font-size: 16pt;">Tutorials: understanding the workflow</h1>

    <h3 style="align: justify;font-size: 12pt">
    <ul>
        <li style="list-style-type: none;font-size: 12pt;">
            <a href="Tutorials/Prep_Rest_Post_Ana/preparation.html">Preparation</a> -
            prepare input files for <code>biceps</code>
        </li>
        <li style="list-style-type: none;font-size: 12pt;">
            <a href="Tutorials/Prep_Rest_Post_Ana/restraint.html">Restraint</a> -
            constructing the ensemble and initializing experimental restraints
        </li>
        <li style="list-style-type: none;font-size: 12pt;">
            <a href="Tutorials/Prep_Rest_Post_Ana/posteriorsampler.html">PosterSampler</a> -
            sampling the posterior distribution for a given ensemble
        </li>
        <li style="list-style-type: none;font-size: 12pt;">
            <a href="Tutorials/Prep_Rest_Post_Ana/analysis.html">Analysis</a> -
            predict populations of conformational states and compute the BICePs score
        </li>
        <li style="list-style-type: none;font-size: 12pt;">
            <a href="Tutorials/Convergence/convergence.html">Convergence</a> -
            check the convergence of MCMC sampling
        </li>
        <li style="list-style-type: none;font-size: 12pt;">
            <a href="Tutorials/MP_Lambdas/mp_lambdas.html">Multiprocessing Lambdas</a> -
            run biceps in a fraction of the time by parallelizing lambda values
        </li>
        <li style="list-style-type: none;font-size: 12pt;">
            <a href="Tutorials/Tools/toolbox.html">Toolbox</a> -
            collection of methods outside the typical <a
            href="https://biceps.readthedocs.io/en/latest/workflow.html">workflow</a>
        </li>
    </ul>
    </h3>
    <hr style="height:2.5px">
    <h1 style="font-size: 16pt;">Full Examples</h1>
    </h3>
    <ul>
        <li style="list-style-type: none;font-size: 12pt;">
            <a href="full_examples/cineromycinB/CineromycinB.html">Cineromycin B</a> -
            determine the solution-state conformational populations of a 14-membered macrolide
            antibiotic. This example contains the use of experimental scalar coupling
            constant alongside NOE data and is based on a previously
            published work by Voelz et al
            (DOI: <a href="https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.23738">10.1002/jcc.23738</a>).
        </li>
        <li style="list-style-type: none;font-size: 12pt;">
            <a href="full_examples/albocycline/albocycline.html">Albocycline</a> -
            compute conforamtional populations of a 14-membered macrolactone. Multiprocess
            lambda values and sample the posterior distribution for each simultaneously.
            This example is based on a previously published work by Zhou et al
            (DOI: <a href="https://www.sciencedirect.com/science/article/pii/S0968089618303389">10.1016/j.bmc.2018.05.017</a>).
        </li>
        <li style="list-style-type: none;font-size: 12pt;">
            <a href="full_examples/apomyoglobin/apomyoglobin.html">Apomyoglobin</a> -
            an example using experimental HDX protetion factors and
            chemical shift data after obtaining the forward model. This
            example is base on a previously published work by Wan et al
            (DOI: <a href="https://doi.org/10.1021/acs.jctc.9b01240">10.1021/acs.jctc.9b01240</a>).
            Posterior sampling of ln PF forward model parameters were calculated
            for ubiquitin and BPTI experimental HDX protection
            factors (ln PF) data for ubiquitin and BPTI. More
            information regarding the forward model can be found
            <a href="https://github.com/vvoelz/HDX-forward-model">here</a>.
        </li>
    </ul>
    <br>
    </h3>

    <h3 style="align: justify;font-size: 12pt">
   You are welcome to contribute your own examples and please let us know
   by simply submit a pull request on our <a href="https://github.com/vvoelz/biceps">Github</a>!</h3>





.. vim: tw=75
