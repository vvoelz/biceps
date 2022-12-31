import biceps
import numpy as np
import pandas as pd
import os, pickle, string
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()


    # Compute NOE model_data
    data_dir = "cineromycin_B/"
    outdir = "_NOE/"
    biceps.toolbox.mkdir(outdir)
    states = biceps.toolbox.get_files(data_dir+"cineromycinB_pdbs/*.pdb")
    ind_noe = data_dir+'atom_indice_noe.txt'
    biceps.toolbox.compute_distances(states, ind_noe, outdir)
    model_data_NOE = str(outdir+"*.txt")
    exp_data_NOE = data_dir+"noe_distance.txt"

    # Compute J-coupling model_data
    ind = np.load(data_dir+'ind.npy')
    ind_J = data_dir+'atom_indice_J.txt'
    outdir = "_J/"
    biceps.toolbox.mkdir(outdir)
    karplus_key=np.loadtxt(data_dir+'Karplus.txt', dtype=str)
    #print('Karplus relations:', karplus_key)
    biceps.toolbox.compute_nonaa_scalar_coupling(states,
            indices=ind, karplus_key=karplus_key, outdir=outdir)
    exp_data_J = data_dir+'exp_Jcoupling.txt'
    model_data_J = data_dir+"J_coupling/*.txt"

    # Now using biceps Preparation submodule
    outdir = "J_NOE/"
    biceps.toolbox.mkdir(outdir)
    prep = biceps.Preparation(nstates=len(states), top_file=states[0], outdir=outdir)
    prep.prepare_noe(exp_data_NOE, model_data_NOE, indices=ind_noe, verbose=False)
    prep.prepare_J(exp_data_J, model_data_J, indices=ind_J, verbose=False)


    input_data = prep.to_sorted_list()
    #print(input_data)

    # Let's look at J-coupling input file for state 0
    pd.read_pickle(input_data[0][0])[["restraint_index", "atom_index1", "exp", "model"]]


    ####### Data and Output Directories #######
    energies = np.loadtxt('cineromycin_B/cineromycinB_QMenergies.dat')*627.509  # convert from hartrees to kcal/mol
    energies = energies/0.5959   # convert to reduced free energies F = f/kT
    energies -= energies.min()  # set ground state to zero, just in case

    # REQUIRED: specify directory of input data (BICePs readable format)
    input_data = biceps.toolbox.sort_data('cineromycin_B/J_NOE')

    # REQUIRED: specify outcome directory of BICePs sampling
    outdir = 'blah'
    # Make a new directory if we have to
    biceps.toolbox.mkdir(outdir)

    fig = plt.figure(figsize=(6,8))
    gs = gridspec.GridSpec(2, 1)
    ax1 = plt.subplot(gs[0,0])
    data1 = pd.concat([pd.read_pickle(i) for i in biceps.toolbox.get_files('cineromycin_B/J_NOE/*.noe')])
    ax1 = data1["model"].plot.hist(alpha=0.5, bins=100,
        edgecolor='black', linewidth=1.2, color="b", label="Model")
    ax1.axvline(list(set(data1["exp"].to_numpy())), c="orange", linewidth=3, label="Experiment")
    ax1.legend(fontsize=14)
    ax1.set_xlabel(r"NOE distance ($\AA$)", size=16)
    ax1.set_ylabel("Counts", size=16)
    ax1.axes.get_yaxis().set_ticks([])

    ax2 = plt.subplot(gs[1,0])
    data2 = pd.concat([pd.read_pickle(i) for i in biceps.toolbox.get_files('cineromycin_B/J_NOE/*.J')])
    ax2 = data2["model"].plot.hist(alpha=0.5, bins=100,
        edgecolor='black', linewidth=1.2, color="b", label="Model")
    data2["exp"].plot.hist(alpha=0.5, bins=40,
        linewidth=1.2, color="orange", label="Experiment", ax=ax2)
    ax2.set_ylim(0,100)
    ax2.axes.get_yaxis().set_ticks([])
    ax2.set_xlabel(r"J coupling (Hz)", size=16)
    ax2.set_ylabel("Counts", size=16)

    axs = [ax1,ax2]
    for n, ax in enumerate(axs):
        ax.text(-0.05, 1.0, string.ascii_lowercase[n], transform=ax.transAxes,
                size=20, weight='bold')
        ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),]
        xmarks = [ax.get_xticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(16)
        for k in range(0,len(xmarks)):
            for mark in xmarks[k]:
                mark.set_size(fontsize=16)
                mark.set_rotation(s=0)
    fig.tight_layout()
    fig.savefig(outdir+'/histogram_of_observables.pdf', dpi=600)

    # REQUIRED: number of MCMC steps for each lambda
    nsteps = 100000000 #100000000 # 100 M steps for production
    #nsteps = 100000 #100000000 # 100 M steps for production
    # REQUIRED: specify how many lambdas to sample (more lambdas will provide higher
    # accuracy but slower the whole process, lambda=0.0 and 1.0 are necessary)
    n_lambdas = 2
    lambda_values = np.linspace(0.0, 1.0, n_lambdas)

    #pd.DataFrame(biceps.get_restraint_options())
    options = biceps.get_restraint_options(input_data)
    pd.DataFrame(options)

    df = pd.DataFrame(options)
    df.to_latex("restraint_options.tex")

    # Change NOE reference potential from uniform to exponential
    options[1]["ref"] = 'exponential'
    # Change the sigma-space to a smaller range of allowed sample space
    options[1]["sigma"] = (0.05, 5.0, 1.02)
    # Alter gamma spacing to have larger width
    options[1]["gamma"] = (0.2, 5.0, 1.02)
    #print(options[1])
    df = pd.DataFrame(options)

    check_convergence = 0
    if check_convergence:
        lam = 1.0
        ensemble = biceps.Ensemble(lam, energies)
        ensemble.initialize_restraints(input_data, options)
        sampler = biceps.PosteriorSampler(ensemble)
        sampler.sample(nsteps=nsteps, print_freq=1000, verbose=False)
        convergence = biceps.Convergence(sampler.traj, outdir=outdir)
        convergence.plot_traces(figname="traces.pdf", xlim=(0, sampler.traj.__dict__["trajectory"][-1][0]))
        convergence.get_autocorrelation_curves(method="block-avg", maxtau=500, nblocks=5)

        init, frac = biceps.find_all_state_sampled_time(sampler.traj.__dict__['state_trace'], len(energies))
        exit()

    # Multiprocess trajectories for each $\lambda$-value with a built-in decorator
    @biceps.multiprocess(iterable=lambda_values)
    def mp_lambdas(lam):
        ensemble = biceps.Ensemble(lam, energies)
        ensemble.initialize_restraints(input_data, options)
        sampler = biceps.PosteriorSampler(ensemble)
        sampler.sample(nsteps=nsteps, print_freq=1000, verbose=False)
        filename = os.path.join(outdir,'traj_lambda%2.2f.npz'%(lam))
        sampler.traj.process_results(filename)
        biceps.toolbox.save_object(sampler, filename.replace(".npz", ".pkl"))

    ############ MBAR and Figures ###########
    # Let's do analysis using MBAR algorithm and plot figures
    A = biceps.Analysis(outdir, nstates=len(energies))
    #biceps.toolbox.save_object(A, "analysis_object.pkl")
    #pops = A.P_dP[:,n_lambdas-1]
    pops, BS = A.P_dP, A.f_df
    print(f"BICePs Scores = {BS[:,0]}")

    pops, BS = A.P_dP, A.f_df
    pops0,pops1 = pops[:,0], pops[:,len(lambda_values)-1]

    legend_fontsize=16
    #fig = A.plot(plottype="hist", figname="BICePs.pdf", figsize=(14,8),
    fig = A.plot(plottype="step", figname="BICePs_.pdf", figsize=(14,8),
           label_fontsize=18, legend_fontsize=legend_fontsize)
    ax = fig.axes[0]
    high_E_confs = [79, 21] #87
    for i in high_E_confs:
        #print(pops0[i], pops1[i], str(i))
        ax.text( pops0[i], pops1[i], str(i), color='r' , fontsize=legend_fontsize)
    fig.savefig(os.path.join(outdir,"BICePs.pdf"), dpi=600)


    mlp = pd.concat([A.get_max_likelihood_parameters(model=i) for i in range(len(lambda_values))])
    mlp.reset_index(inplace=True, drop=True)
    print(mlp)

    # NOTE: Get Prior MSM populations
    prior_pops = np.loadtxt(data_dir+"prior_pops.txt")
    prior_pops /= prior_pops.sum()

    noe = [pd.read_pickle(i) for i in biceps.toolbox.get_files(data_dir+"J_NOE/*.noe")]
    #  Get the ensemble average observable
    noe_Exp = noe[0]["exp"].to_numpy()
    noe_model = [i["model"].to_numpy() for i in noe]

    noe_prior = np.array([w*noe_model[i] for i,w in enumerate(prior_pops)]).sum(axis=0)
    noe_reweighted = np.array([w*noe_model[i] for i,w in enumerate(pops[:,n_lambdas-1])]).sum(axis=0)

    distance_labels = [f"{row[1]['atom_name1']}-{row[1]['atom_name2']}" for row in noe[0].iterrows()]
    distance_label_indices = np.array([[row[1]['atom_index1'], row[1]['atom_index2']] for row in noe[0].iterrows()])


    J = [pd.read_pickle(file) for file in biceps.toolbox.get_files(data_dir+'J_NOE/*.J')]
    #  Get the ensemble average observable
    J_Exp = J[0]["exp"].to_numpy()
    J_model = [i["model"].to_numpy() for i in J]

    J_prior = np.array([w*J_model[i] for i,w in enumerate(prior_pops)]).sum(axis=0)
    J_reweighted = np.array([w*J_model[i] for i,w in enumerate(pops[:,n_lambdas-1])]).sum(axis=0)

    J_labels = [f"{row[1]['atom_name1']}-{row[1]['atom_name2']}-{row[1]['atom_name3']}-{row[1]['atom_name4']}" for row in J[0].iterrows()]
    J_label_indices = np.array([[row[1]['atom_index1'], row[1]['atom_index2'], row[1]['atom_index3'], row[1]['atom_index4']] for row in J[0].iterrows()])




    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(2, 1)


    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    data = []
    for i in range(len(noe_reweighted)):
        data.append({"index":i,
            "reweighted noe":noe_reweighted[i], "prior noe":noe_prior[i],
            "exp noe":noe_Exp[i]*mlp['gamma_noe'].to_numpy()[-1], "label":distance_labels[i]
            })
    data1 = pd.DataFrame(data)

    _data1 = data1.sort_values(["prior noe"])
    _data1 = _data1.reset_index()
    #print(_data1)


    #ax1 = data1.plot.scatter(x='index', y="reweighted noe", s=5, edgecolor='black', color="b", label="BICePs")
    ax1.scatter(x=_data1["label"].to_numpy(), y=_data1["prior noe"].to_numpy(),
                s=45, color="orange", label="Prior", edgecolor='black',)
    ax1.scatter(x=_data1["label"].to_numpy(), y=_data1["exp noe"].to_numpy(),
                s=100, marker="_", color="k", label="Exp")
    ax1.scatter(x=_data1["label"].to_numpy(), y=_data1["reweighted noe"].to_numpy(),
                s=40, color="c", label="BICePs", edgecolor='black')
    ax1.legend(fontsize=14)
    #ax1.set_xlabel(r"Index", size=16)
    ax1.set_ylabel(r"NOE distance ($\AA$)", size=16)


    data = []
    for i in range(len(J_reweighted)):
        data.append({"index":i,
            "reweighted J":J_reweighted[i], "prior J":J_prior[i],
            "exp J":J_Exp[i], "label":J_labels[i]
            })
    data1 = pd.DataFrame(data)

    ax2.scatter(x=data1['label'].to_numpy(), y=data1["prior J"].to_numpy(),
                s=45, color="orange", label="Prior", edgecolor='black',)
    ax2.scatter(x=data1['label'].to_numpy(), y=data1["exp J"].to_numpy(),
                s=100, marker="_", color="k", label="Exp")
    ax2.scatter(x=data1['label'].to_numpy(), y=data1["reweighted J"].to_numpy(),
                s=40, color="c", label="BICePs", edgecolor='black')
    ax2.legend(fontsize=14)
    #ax2.set_xlabel(r"Index", size=16)
    ax2.set_ylabel(r"J-coupling (Hz)", size=16)

    ticks = [
             ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),]
    xmarks = [ax1.get_xticklabels(),
            ]
    ymarks = [ax1.get_yticklabels(),
            ]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(xmarks)):
        for mark in xmarks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=65)
    for k in range(0,len(ymarks)):
        for mark in ymarks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=0)


    ticks = [
             ax2.xaxis.get_minor_ticks(),
             ax2.xaxis.get_major_ticks(),]
    xmarks = [
             ax2.get_xticklabels(),

            ]
    ymarks = [
             ax2.get_yticklabels(),
            ]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(xmarks)):
        for mark in xmarks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=70)
    for k in range(0,len(ymarks)):
        for mark in ymarks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=0)


    axs = [ax1,ax2]
    for n, ax in enumerate(axs):
        ax.text(-0.1, 1.0, string.ascii_lowercase[n], transform=ax.transAxes,
                size=20, weight='bold')
    fig.tight_layout()
    fig.savefig(f"{outdir}/reweighted_observables.pdf", dpi=500)


    #lam = 1.0
    #nsteps=1000000
    #ensemble = biceps.Ensemble(lam, energies)
    #ensemble.initialize_restraints(input_data, options)
    #sampler = biceps.PosteriorSampler(ensemble)
    #sampler.sample(nsteps=nsteps, print_freq=1000, verbose=False)
    #traj = sampler.traj.__dict__
    #C = biceps.Convergence(traj)
    #C.plot_traces(figname="test.pdf", xlim=(0, nsteps))
    #C.get_autocorrelation_curves(method="auto", maxtau=5000)
    #C.process()
    #




