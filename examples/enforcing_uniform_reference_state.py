'''
A notebook of a simple three state toy model system to demonstrate four different
approaches of computing the BICePs score, $f_{xi=0 -> 1}$ represented
as the free energy of "turning on" the data restraints.

Please see the Jupyter notebook for more details and figures.
(examples/enforcing_uniform_reference_state.ipynb)
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import biceps
from biceps.PosteriorSampler import u_kln_and_states_kn
from pymbar import MBAR
from scipy import interpolate, integrate
import sys
import FwdModelOpt_routines as fmo


#trial = int(sys.argv[1])
trial = 3

def write_noe_files(weights, x, exp, dir):
    for i in range(len(weights)):
        model = pd.read_pickle("template.noe")
        _model = pd.DataFrame()
        for j in range(len(exp)):
            model["restraint_index"], model["model"], model["exp"] = 1+j, x[i][j], exp[j]
            _model = pd.concat([_model,model], ignore_index=True)
        _model.to_pickle(dir+"/%s.noe"%i)

####### BICePs Parameters #######
nStates,Nd = 3,2
n_xis,n_lambdas,nreplicas,nsteps,swap_every,write_every=1,2,8,1000000,0,100
stat_model,data_uncertainty="Students","single"
expanded_values = [(0.0,0.00), (0.0,0.001), (0.0,0.0025),
                   (0.0,0.01), (0.0,0.025), (0.0,0.05),
                   (0.0,0.1), (0.0,0.15), (0.0,0.20),
                   (0.0,0.25), (0.0,0.3), (0.0,0.35), (0.0,0.4),
                   (0.0,0.45), (0.0,0.5), (0.0,0.55), (0.0,0.6),
                   (0.0,0.65), (0.0,0.7), (0.0,0.75),
                   (0.0,0.8), (0.0,0.85),
                   (0.0,0.9), (0.0,1.0), (1.0,1.0)]
multiprocess=1
move_state_every,move_sigma_every = 1,1
burn = 1000
lambda_values = np.linspace(0.0, 1.0, n_lambdas)
sigMin,sigMax,dsig = 0.001,100,1.02
arr = np.exp(np.arange(np.log(sigMin), np.log(sigMax), np.log(dsig)))
sigma_index = round(len(arr)*0.73)
options = [dict(stat_model=stat_model,
            sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index,
            data_uncertainty=data_uncertainty,
            )]
print(f"nSteps of sampling: {nsteps}\nnReplicas: {nreplicas}")

####### Toy model data #######
populations = np.array([0.5, 0.3, 0.2])
energies = -np.log(populations)
forward_model_data = np.array([[1.0, 1.1],
                               [1.2, 1.3],
                               [1.4, 1.5],
                              ])
experiment = [0.0, 0.0]
# Force fields yield different populations. Here, Model1 and Model2 give
# state0 and state1 smaller populations (higher energy).
energy_perturbations = np.array([
                        [0.00000000, 0.00000000, 0.00000000],
                        [0.00000000, 0.00000000, 0.00000000],
                        [0.00000000, 0.00000000, 0.00000000]])
# Perurb the 2nd distance of the forward model by for each model by [0,1,2].
# By setting these values to zero, we will find the correction virtually goes to 0.
forward_model_perturbations = [0, 1.0, 2.0]

ensembles,trajs,logZs,results = [],[],[],[]

####### Create directories and write to files #######
dir = "results"
biceps.toolbox.mkdir(dir)
state_dir = f"{dir}/{nStates}_state_toy_model"
biceps.toolbox.mkdir(state_dir)
dir = f"{state_dir}/{nStates}_state_{Nd}_datapoints"
data_dir = dir+"/NOE"
biceps.toolbox.mkdir(data_dir)
write_noe_files(weights=energies, x=forward_model_data, exp=experiment, dir=data_dir)

input_data = biceps.toolbox.sort_data(data_dir)
print(f"Input data: {biceps.toolbox.list_extensions(input_data)}")
outdir = f'{dir}/{stat_model}_{data_uncertainty}_sigma/{nsteps}_steps_{nreplicas}_replicas_{n_lambdas}_lam__swap_every_{swap_every}'
biceps.toolbox.mkdir(outdir)


# other methods:{{{

for i in range(len(energy_perturbations)):

    ####### Data for i'th model #######
    new_energies = energies + energy_perturbations[i]
    new_populations = np.exp(-new_energies)
    new_populations /= new_populations.sum()
    print(f"Prior populations: {new_populations}")
    # Shift the 2nd data point for each state by +1 for each new model
    model_perturbations = np.zeros(forward_model_data.shape)
    model_perturbations[:, 1] = model_perturbations[:, 1]+forward_model_perturbations[i]
    new_forward_model_data = forward_model_data + model_perturbations

    data_dir = dir+f"/NOE_{i}"
    biceps.toolbox.mkdir(data_dir)
    write_noe_files(weights=energies, x=new_forward_model_data, exp=experiment, dir=data_dir)
    input_data = biceps.toolbox.sort_data(data_dir)

    ####### Run MCMC #######
    ensemble = biceps.ExpandedEnsemble(expanded_values=expanded_values, energies=new_energies)
    ensemble.initialize_restraints(input_data, options, verbose=0)
    sampler = biceps.PosteriorSampler(ensemble, nreplicas, write_every=write_every)
    sampler.sample(nsteps, attempt_lambda_swap_every=swap_every, swap_sigmas=1,
            attempt_move_state_every=move_state_every,
            attempt_move_sigma_every=move_sigma_every,
            progress=1, multiprocess=multiprocess, capture_stdout=0)

    ####### store data #######
    expanded_values = sampler.expanded_values
    trajs.append([sampler.traj[k].__dict__ for k in range(len(sampler.traj))])

    ####### Calculate BICePs scores #######
    A = biceps.Analysis(sampler, outdir=outdir)
    # get BICePs scores w/ MBAR
    BS, pops = A.f_df, A.P_dP[:,len(expanded_values[:])-1]
    print(f"Reweighted populations: {pops}\n\n")
    BS /= sampler.nreplicas
    K = len(expanded_values[:])-1
    pops_std = A.P_dP[:,2*K]
    # approximate scores w/ exponential averaging
    approx_scores = A.approximate_scores(burn)
    approx_scores["exp_avg"] /= sampler.nreplicas
    approx_scores["BS"] = BS[:,0] # add BICePs scores to DataFrame
    results.append(approx_scores)




extended = []
for i in range(len(results)):
    res = results[i]
    ref = res.iloc[np.where((res["lambda"]==0.0) & (res["xi"]==0.0))[0]]
    data_rest = res.iloc[np.where((res["lambda"]==0.0) & (res["xi"]==1.0))[0]]
    full = res.iloc[np.where(res["lambda"]==1.0)[0]]
    _df = pd.concat([ref,data_rest,full])
    _df["Model"] = np.ones(len(_df), dtype=int)*i
    extended.append(_df)
extended = pd.concat(extended).reset_index()
extended.drop(["index"], axis=1,inplace=True)
abs_FE_ref_lam0 = extended.iloc[np.where((extended["lambda"]==0.0) & (extended["xi"]==1.0) &
                                    (extended["Model"]==0))[0]]["BS"].to_numpy()





ensembles,trajs,logZs,results = [],[],[],[]
for i in range(len(energy_perturbations)):

    ####### Data for i'th model #######
    new_energies = energies + energy_perturbations[i]
    new_populations = np.exp(-new_energies)
    new_populations /= new_populations.sum()
    print(f"Prior populations: {new_populations}")
    # Shift the 2nd data point for each state by +1 for each new model
    model_perturbations = np.zeros(forward_model_data.shape)
    model_perturbations[:, 1] = model_perturbations[:, 1]+forward_model_perturbations[i]
    new_forward_model_data = forward_model_data + model_perturbations

    data_dir = dir+f"/NOE_{i}"
    biceps.toolbox.mkdir(data_dir)
    write_noe_files(weights=energies, x=new_forward_model_data, exp=experiment, dir=data_dir)
    input_data = biceps.toolbox.sort_data(data_dir)

    ####### Run MCMC #######
    ensemble = biceps.ExpandedEnsemble(lambda_values=lambda_values, energies=new_energies)
    ensemble.initialize_restraints(input_data, options, verbose=0)
    sampler = biceps.PosteriorSampler(ensemble, nreplicas, write_every=write_every)
    sampler.sample(nsteps, attempt_lambda_swap_every=swap_every, swap_sigmas=1,
            attempt_move_state_every=move_state_every,
            attempt_move_sigma_every=move_sigma_every,
            progress=1, multiprocess=multiprocess, capture_stdout=0)

    ####### store data #######
    expanded_values = sampler.expanded_values
    # get ensemble and normalization
    for k in range(len(sampler.ensembles)):
        ensembles.append(sampler.ensembles[k])
        logZs.append(sampler.logZs[k])
    # get trajectories
    trajs.append([sampler.traj[k].__dict__ for k in range(len(sampler.traj))])

    ####### Calculate BICePs scores #######
    A = biceps.Analysis(sampler, outdir=outdir)
    # get BICePs scores w/ MBAR
    BS, pops = A.f_df, A.P_dP[:,len(expanded_values[:])-1]
    print(f"Reweighted populations: {pops}\n\n")
    K = len(expanded_values[:])-1
    pops_std = A.P_dP[:,2*K]
    # approximate scores w/ exponential averaging
    approx_scores = A.approximate_scores(burn)
    approx_scores["exp_avg"] /= sampler.nreplicas
    approx_scores["BS"] = BS[:,0]/sampler.nreplicas # add BICePs scores to DataFrame
    results.append(approx_scores)


# In[11]:


####### Calculate BICePs #######
trajectories = np.concatenate(trajs)
K = len(ensembles)   # number of thermodynamic ensembles
N_k = np.array( [len(trajectories[i]['trajectory']) for i in range(len(ensembles))] )
u_kln, states_kn, Nr_array = u_kln_and_states_kn(ensembles, trajectories,
                nstates=len(energies), logZs=logZs)
mbar = MBAR(u_kln, N_k)
_results = mbar.compute_free_energy_differences(uncertainty_method='approximate', return_theta=True)
Deltaf_ij, dDeltaf_ij, Theta_ij = _results["Delta_f"], _results["dDelta_f"], _results["Theta"]
beta = 1.0 # keep in units kT
f_df = np.zeros( (len(ensembles), 2) )  # first column is Deltaf_ij[0,:], second column is dDeltaf_ij[0,:]
Deltaf_ij /= nreplicas
f_df[:,0] = Deltaf_ij[0,:]  # biceps score
f_df[:,1] = dDeltaf_ij[0,:] # biceps score std

# NOTE: https://pymbar.readthedocs.io/en/master/mbar.html#pymbar.MBAR.compute_overlap
overlap = mbar.compute_overlap()
fig, ax = plt.subplots(figsize=(6, 5))  # Adjust the figsize as desired
im = ax.pcolor(overlap['matrix'], edgecolors='k', linewidths=2)

df = pd.concat(results)
model_ids = [i for i in range(len(trajs)) for k in range(len(trajs[i]))]
df["Model"] = model_ids
df["FF"] = [f"Model{model_ids[m]} "+f"{(lam,xi)}"
            for m,(lam,xi) in enumerate(zip(df["lambda"].to_numpy(),df["xi"].to_numpy()))]
ff = df["FF"].to_numpy()



col_name = "BS w/ ref=Model"
df = pd.concat(results)
model_ids = [i for i in range(len(trajs)) for k in range(len(trajs[i]))]
df["Model"] = model_ids
df[col_name+"0"] = f_df[:,0]
df.to_csv(f"{outdir}/scores.csv", index=False)
columns = [col for col in df.columns.to_list() if col_name in col]
lam0 = df.iloc[np.where(df["lambda"]==0.0)[0]]
lam0 = lam0[columns]
lam1 = df.iloc[np.where(df["lambda"]==1.0)[0]]
lam1 = lam1[columns]
values = []
for i in range(len(lam0)):
    values.append(lam0.to_numpy()[i]+abs_FE_ref_lam0)
    values.append(lam1.to_numpy()[i]+abs_FE_ref_lam0)
df["Abs. BS w/ ref=Model0"] = np.array(values)
indices = np.where((extended["lambda"]==0.0) & (extended["xi"]==0.0))[0]
df["Abs. BS"] = extended.iloc[[i for i in range(len(extended)) if i not in indices]]["BS"].to_numpy()




_scores = []
figs = []
ensembles,trajs,logZs,results = [],[],[],[]
for i in range(len(energy_perturbations)):

    ####### Data for i'th model #######
    new_energies = energies + energy_perturbations[i]
    new_populations = np.exp(-new_energies)
    new_populations /= new_populations.sum()
    print(f"Prior populations: {new_populations}")
    # Shift the 2nd data point for each state by +1 for each new model
    model_perturbations = np.zeros(forward_model_data.shape)
    model_perturbations[:, 1] = model_perturbations[:, 1]+forward_model_perturbations[i]
    new_forward_model_data = forward_model_data + model_perturbations

    data_dir = dir+f"/NOE_{i}"
    biceps.toolbox.mkdir(data_dir)
    write_noe_files(weights=energies, x=new_forward_model_data, exp=experiment, dir=data_dir)
    input_data = biceps.toolbox.sort_data(data_dir)

    burn = 10000
    num_xi_values = 11
    change_xi_every = round(nsteps/num_xi_values)
    dXi = 1 / (num_xi_values - 1)

    ####### Run MCMC #######
    ensemble = biceps.ExpandedEnsemble(lambda_values=[lambda_values[0]], energies=new_energies)
    ensemble.initialize_restraints(input_data, options, verbose=0)
    sampler = biceps.PosteriorSampler(ensemble, nreplicas, write_every=write_every,
            change_xi_every=change_xi_every, dXi=dXi, xi_integration=1)
    sampler.sample(nsteps, attempt_lambda_swap_every=swap_every, swap_sigmas=1,
            burn=burn,
            attempt_move_state_every=move_state_every,
            attempt_move_sigma_every=move_sigma_every,
            progress=1, multiprocess=multiprocess, capture_stdout=0)

    # get ensemble and normalization
    for k in range(len(sampler.ensembles)):
        ensembles.append(sampler.ensembles[k])
        logZs.append(sampler.logZs[k])
    # get trajectories
    trajs.append([sampler.traj[k].__dict__ for k in range(len(sampler.traj))])

    ti_info = sampler.ti_info

    mbar = sampler.integrate_xi_ensembles(multiprocess=1)
    #print(mbar.f_k)
    overlap = mbar.compute_overlap()
    overlap_matrix = overlap['matrix']

    _results = mbar.compute_free_energy_differences(uncertainty_method='approximate', return_theta=True)
    Deltaf_ij, dDeltaf_ij, Theta_ij = _results["Delta_f"], _results["dDelta_f"], _results["Theta"]
    #print(Deltaf_ij)
    f_df = np.zeros( (len(overlap_matrix), 2) )  # first column is Deltaf_ij[0,:], second column is dDeltaf_ij[0,:]
    f_df[:,0] = Deltaf_ij[0,:]  # NOTE: biceps score
    f_df[:,1] = dDeltaf_ij[0,:] # NOTE: biceps score std
    BS = -f_df[:,0]/sampler.nreplicas
    #print(BS)
    print("Integral from MBAR:", BS[-1])

    force_constants = list(range(len(overlap_matrix)))
    fig, ax = plt.subplots(figsize=(14, 10))  # Adjust the figsize as desired
    im = ax.pcolor(overlap_matrix, edgecolors='k', linewidths=2)

    # Add annotations
    for i in range(len(overlap_matrix)):
        for j in range(len(overlap_matrix[i])):
            value = overlap_matrix[i][j]
            if value > 0.01:
                text_color = 'white' if value < 0.5 else 'black'
                ax.text(j + 0.5, i + 0.5, f"{value:.2f}", ha='center', va='center', color=text_color,
                        fontsize=8)  # Adjust fontsize as desired
    ax.set_xticks(np.array(list(range(len(force_constants)))))
    ax.set_yticks(np.array(list(range(len(force_constants)))))
    ax.tick_params(axis='x', direction='inout')
    ax.tick_params(axis='y', direction='inout')
    ax.grid()

    try:
        ax.set_xticklabels([str(tick) for tick in force_constants], rotation=90)
        ax.set_yticklabels([str(tick) for tick in force_constants])
    except Exception as e:
        print(e)

    ax.set_xticklabels(ax.get_xticklabels(), ha='left')
    ax.set_yticklabels(ax.get_yticklabels(), va='bottom')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label("Transition probability")  # Set the colorbar label
    fig.tight_layout()
    fig.savefig(f"{outdir}/contour.png")



    #integral = sampler.get_score_using_TI()
    #print("Integral:", integral)
    ####### store data #######
    expanded_values = sampler.expanded_values
    A = biceps.Analysis(sampler, outdir=outdir, nstates=len(energies),
                    scale_energies=0, MBAR=False, multiprocess=1)
    A.plot_energy_trace()


    _scores.append(BS[-1])
    A = biceps.Analysis(sampler, outdir=outdir, MBAR=False)
    figs.append(A.plot_energy_trace())



col_name = "BS"
columns = [col for col in df.columns.to_list() if col_name in col]
lam0 = df.iloc[np.where(df["lambda"]==0.0)[0]]
lam0 = lam0[columns]
lam1 = df.iloc[np.where(df["lambda"]==1.0)[0]]
lam1 = lam1[columns]

values = []
for i in range(len(lam0)):
    values.append((lam0.to_numpy()[i]+_scores[i])[0])
    values.append((lam1.to_numpy()[i]+_scores[i])[0])

df["Abs. BS (xi integration)"] = values

# }}}


###############################################################################


_scores = []
figs = []
ensembles,trajs,logZs,results = [],[],[],[]
for i in range(len(energy_perturbations)):

    ####### Data for i'th model #######
    new_energies = energies + energy_perturbations[i]
    new_populations = np.exp(-new_energies)
    new_populations /= new_populations.sum()
    print(f"Prior populations: {new_populations}")
    # Shift the 2nd data point for each state by +1 for each new model
    model_perturbations = np.zeros(forward_model_data.shape)
    model_perturbations[:, 1] = model_perturbations[:, 1]+forward_model_perturbations[i]
    new_forward_model_data = forward_model_data + model_perturbations

    data_dir = dir+f"/NOE_{i}"
    biceps.toolbox.mkdir(data_dir)
    write_noe_files(weights=energies, x=new_forward_model_data, exp=experiment, dir=data_dir)
    input_data = biceps.toolbox.sort_data(data_dir)

    ####### Run MCMC #######
    _ensemble = biceps.ExpandedEnsemble(lambda_values=[lambda_values[0]], energies=new_energies)
    _ensemble.initialize_restraints(input_data, options, verbose=0)

    _PSkwargs = biceps.toolbox.get_PSkwargs()
    _sample_kwargs = biceps.toolbox.get_sample_kwargs()

    _sample_kwargs["progress"] = 1
    _sample_kwargs["verbose"] = 0
#    _sample_kwargs["burn"] = 1000
    _sample_kwargs["burn"] = 10000
    _sample_kwargs["nsteps"] = nsteps
    _sample_kwargs["attempt_move_fm_prior_sigma_every"] = 0
    _PSkwargs["fmo"] = 0
    _PSkwargs["write_every"] = write_every
    _PSkwargs["xi_integration"] = 1
    _PSkwargs["nreplicas"] = nreplicas
    _PSkwargs["num_xi_values"] = 11
#
    score = fmo.xi_integration(_ensemble, _PSkwargs, _sample_kwargs, plot_overlap=True, outdir=f"{outdir}",
                   optimize_xi_values=1, xi_opt_steps=2000000, tol=1e-7, alpha=1e-5, progress=1,
                   max_attempts=4, print_every=1000, scale_energies=False, verbose=False)
    _scores.append(score)

    print(f"BICePs score: {score}")

#exit()

col_name = "BS"
columns = [col for col in df.columns.to_list() if col_name in col]
lam0 = df.iloc[np.where(df["lambda"]==0.0)[0]]
lam0 = lam0[columns]
lam1 = df.iloc[np.where(df["lambda"]==1.0)[0]]
lam1 = lam1[columns]

values = []
for i in range(len(lam0)):
    values.append((lam0.to_numpy()[i]+_scores[i])[0])
    values.append((lam1.to_numpy()[i]+_scores[i])[0])

df["Abs. BS (xi integration w/ xi Opt)"] = values

###############################################################################

df.to_csv(f"{outdir}/trial{trial}.csv")


files = biceps.toolbox.get_files(outdir+"/trial*.csv")
dfs = []
for file in files:
    dfs.append(pd.read_csv(file, index_col=0))
df = pd.concat(dfs)
grouped = df.groupby(["lambda","xi", "Model"])
mean = grouped.agg("mean")
std = grouped.agg("std")
mean = mean.reset_index()
std = std.reset_index()
print(mean)


figsize = (7, 5)
label_fontsize=12
legend_fontsize=10
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(1, 1, wspace=0.02)
ax = fig.add_subplot(gs[0,0])

if all(np.isnan(std["Abs. BS"].to_numpy())):
    std["Abs. BS w/ ref=Model0"] = np.zeros(std["Abs. BS w/ ref=Model0"].to_numpy().shape)
    std["Abs. BS"] = np.zeros(std["Abs. BS"].to_numpy().shape)
    std["Abs. BS (xi integration)"] = np.zeros(std["Abs. BS (xi integration)"].to_numpy().shape)
    std["Abs. BS (xi integration w/ xi Opt)"] = np.zeros(std["Abs. BS (xi integration w/ xi Opt)"].to_numpy().shape)

ax.errorbar(x=mean["Model"], y=mean["Abs. BS w/ ref=Model0"].to_numpy(), yerr=std["Abs. BS w/ ref=Model0"].to_numpy(),
            color="k", fmt='o', capsize=6, label="Relative")
ax.errorbar(x=mean["Model"], y=mean["Abs. BS"].to_numpy(), yerr=std["Abs. BS"].to_numpy(),
            color="r", fmt='o', capsize=6, label="Absolute (intermediates)")
ax.errorbar(x=mean["Model"], y=mean["Abs. BS (xi integration)"].to_numpy(), yerr=std["Abs. BS (xi integration)"].to_numpy(),
            color="blue", fmt='o', capsize=6, label=r"Absolute ($\xi$ integration)")
ax.errorbar(x=mean["Model"], y=mean["Abs. BS (xi integration w/ xi Opt)"].to_numpy(), yerr=std["Abs. BS (xi integration w/ xi Opt)"].to_numpy(),
            color="green", fmt='o', capsize=6, label=r"Absolute ($\xi$ integration w/ $\xi$ Opt)")
xticks = ax.get_xticks()
xticks = np.array(xticks, dtype=int)
xticks = [str(tick) for tick in xticks]
for i in range(len(xticks)):
    if i%2 != 1: xticks[i] = ""
ax.set_xticklabels(xticks, size=14)
yticks = ax.get_yticks()
ax.set_yticklabels(yticks, size=14)
ax.set_xlabel("Model", fontsize=16)
ax.set_ylabel("Score", fontsize=16)
ax.set_ylim(4.1, 5.9)
legend = ax.legend(fontsize=14)
fig.savefig(f"{outdir}/biceps_scores_for_each_model.png")


exit()







