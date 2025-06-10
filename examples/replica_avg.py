"""
This script is slightly more comprehensive as it demonstrates the broad functionality
of the BICePs module with many code blocks.

Here, we use a 10 state toy model system with 20 syntehtic experimental observables.

To test the performance of the various likelihood models, the user can also add
random and systematic error to the experimental observables.
In addition, you can add prior error and see how the accuray of BICePs reweighting
relies on a good prior.

In this script, you can check convergence with our built-in module.
(see this paper for more details: https://doi.org/10.1021/acs.jcim.2c01296)
The user can obtain the maximum a posteriori parameters, which can then be used
to compute an effective chi-squared using the inferred sigma from sampling the
posterior.

"""

# Python libraries:{{{
import numpy as np
import sys, time, os, gc, psutil, string, re
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import scipy
from scipy.stats import maxwell
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import biceps
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import repXroutines as rxr
import uncertainties as u
#:}}}

# Methods:{{{
def matprint(mat, fmt="g"):
    """
    https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
    """
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


# }}}


################################################################################
####### Parameters #######
nStates,Nd = 10,20
n_xis,n_lambdas,nreplicas,nsteps,change_Nr_every,swap_every=1,2,128,100000,0,0
n_xis,n_lambdas,nreplicas,nsteps,change_Nr_every,swap_every=1,2,128,50000,0,0


#NOTE: [(λ,ξ), (λ,ξ)]
expanded_values = [(0.0,1.0), (1.0,1.0)]

#burn = int(nsteps)
#burn = int(nsteps/4)
burn = 0
scale_energies = 0
find_optimal_nreplicas = 0
multiprocess=1

#σ_prior=0.16 # 0.08, 0.16
σ_prior=0.161 # 0.08, 0.16
#σ_prior=0.000 # 0.08, 0.16

σ_data=0.0   # 0.0, 0.5
μ_data=0.0   # 0.0, 4.0

#σ_data=0.5   # 0.0, 0.5
#μ_data=4.0   # 0.0, 4.0
verbose=True

#stat_model,data_uncertainty="Bayesian","single"
stat_model,data_uncertainty="GB","single"
#stat_model,data_uncertainty="Students","single"
#stat_model,data_uncertainty="Gaussian","multiple"

data_likelihood = "gaussian" #"log normal" # "gaussian"
#data_likelihood = "log normal" #"gaussian" #"log normal" # "gaussian"

verbose = 0

if verbose:
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

continuous_space=0
walk_in_all_dim=continuous_space
# Good parameters
#dsigma=0.055
dsigma=None
move_sigma_std=1.0
sigma_batch_size=Nd
#sigma_batch_size=1
write_every = 1
#write_every = 10

attempt_move_state_every = 1
attempt_move_sigma_every = 1
#attempt_move_state_every = 1
#attempt_move_sigma_every = 1

scale_and_offset = 0
move_ftilde_every = 0
dftilde = 1.10 #1.0 #0.1
ftilde_sigma = 1.0 #2.0 #1.0
plottype="step"
##########################
state_dir = f"{nStates}_state"
biceps.toolbox.mkdir(state_dir)
dir = f"{state_dir}/{nStates}_state_{Nd}_datapoints/Prior_error_{σ_prior}"
# TODO: Do once, then comment out...
boltzmann_scale=1.
boltzmann_scale=0.5
boltzmann_domain=(0.12,0.6)   # for 3 state model
#boltzmann_domain=(0.15, 0.35) # for 5 state model
#boltzmann_domain=(-0.01, .35) # for 5 state model
#boltzmann_domain=(-0.2,0.038) # for 50 state model
#boltzmann_domain=(-0.3,0.01) # for 100 state model
boltzmann_loc=0.0
rxr.gen_synthetic_data(dir, nStates, Nd, σ_prior=σ_prior, σ_data=σ_data, μ_data=μ_data,
        boltzmann_scale=boltzmann_scale, boltzmann_domain=boltzmann_domain, boltzmann_loc=boltzmann_loc,
                       verbose=verbose, as_intensities=0)

# true pops
populations = np.array(np.loadtxt(f"{dir}/pops.txt"), ndmin=1)
print("True pops: ", populations)
# perturbed populations
prior_populations = np.array(np.loadtxt(f"{dir}/prior_pops.txt"), ndmin=1)
energies = np.array(np.loadtxt(f"{dir}/prior.txt"), ndmin=1)
energies -= energies.min()
data_sigma = np.array(np.loadtxt(f"{dir}/data_sigma.txt"), ndmin=1)
if np.isnan(data_sigma): data_sigma = 0.0
data_deviations = np.array(np.loadtxt(f"{dir}/data_deviations.txt"), ndmin=1)
################################################################################
data_dir = dir+"/NOE"
#input_data = biceps.toolbox.sort_data(data_dir)
#input_data = biceps.toolbox.sort_data(data_dir+"/*.noe*")
#input_data = biceps.toolbox.sort_data(data_dir+"/*.noe*")
input_data = [[file] for file in biceps.toolbox.get_files(data_dir+"/*.noe*")]
#print(input_data)
#exit()

#if rdc_only: input_data = biceps.toolbox.sort_data(f'{data_dir}/CS_J_NOE/*.rdc*')
print(f"Input data: {biceps.toolbox.list_extensions(input_data)}")
model = pd.concat([pd.read_pickle(i) for i in biceps.toolbox.get_files(f"{data_dir}/*.noe")])
ax2 = model["model"].plot.hist(alpha=0.5, bins=30, edgecolor='black', linewidth=1.2, density=1, color="b", figsize=(14, 6), label="data1", ax=None)
model["exp"].plot.hist(bins=20, alpha=0.5, ax=ax2, color="orange", density=True)
fig2 = ax2.get_figure()
fig2.savefig(f"{dir}/synthetic_experimental_NOEs.png")
plt.close()
#exit()
#outdir = f'{dir}/{nsteps}_steps_{nreplicas}_replica_{n_lambdas}_lam'
outdir = f'{dir}/{stat_model}_{data_uncertainty}_sigma/{nsteps}_steps_{nreplicas}_replicas_{n_lambdas}_lam__swap_every_{swap_every}'
biceps.toolbox.mkdir(outdir)
print(f"nSteps of sampling: {nsteps}\nnReplicas: {nreplicas}")
lambda_values = np.linspace(0.0, 1.0, n_lambdas)


#noe = [pd.read_pickle(i) for i in biceps.toolbox.get_files(f"{data_dir}/*.noe")]
#print(noe)
#exit()


if n_xis > 1: xi_values = np.linspace(0.0, 1.0, n_xis)
else: xi_values = [1.0]

#parameters = [dict(ref="uniform", sigma=(0.00001, 200, 1.009), gamma=(1.0, 2.0, np.e)),]
#dsig = 1.01
#dsig = 1.005
dsig = 1.02
sigMin = 0.001
sigMax = 50

#sigMin = 0.00001
#sigMax = 50
sigMax = 100
arr = np.exp(np.arange(np.log(sigMin), np.log(sigMax), np.log(dsig)))
#print(arr)
#exit()
l = len(arr)
#sigma_index = round(l*0.81)#1689 # 0.80
sigma_index = round(l*0.73)#1689 # 0.80

if dsigma == None: dsigma=0.05*(arr[sigma_index] - arr.min())



if (stat_model == "GaussianGB") or (stat_model == "GB"):
    # NOTE: acting as gamma for Good-and-Bad
    #alpha=(1., 2, 1000)
    beta,beta_index=(1., 2.0, 1),0
    #phi,phi_index=(1., 100.0, 10000),0
    phi,phi_index=(1., 2.0, 1),0
    gamma,gamma_index=(1.0, 2.0, np.e),0
    _arr = np.linspace(*phi)
    _l = len(_arr)
    print("Alpha starts here: ",_arr[phi_index])

elif (stat_model == "Students") :
    beta,beta_index=(1., 100.0, 10000),0
    #beta,beta_index=(1., 10.0, 1),0
    _arr = np.linspace(*beta)
    _l = len(_arr)
    print("Alpha starts here: ",_arr[beta_index])
    phi,phi_index=(1., 2.0, 1),0
    gamma,gamma_index=(1.0, 2.0, np.e),0

else:
    beta,beta_index=(1., 2.0, 1),0
    phi,phi_index=(1., 2.0, 1),0
    gamma,gamma_index=(1.0, 2.0, np.e),0


options = biceps.get_restraint_options(input_data)
for i in range(len(options)):
    options[i].update(dict(ref="uniform",
         sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index,
         #sigma=(1.0, 2.0, np.e), sigma_index=0,
         phi=phi, phi_index=phi_index, beta=beta, beta_index=beta_index,
#         omega=(0.0001, 1, 1000), omega_index=999,
         #convert_to_intensity=0,
         stat_model=stat_model, data_uncertainty=data_uncertainty,
         data_likelihood=data_likelihood))


print(pd.DataFrame(options))
#np.logspace(start=0, stop=-1, num=number_of_replica)
#print(arr)
print("len = ",l)
print("simga = ",arr[sigma_index])

energies = np.ones(len(input_data))/len(input_data)
print("energies.shape = ",energies.shape)

ensemble = biceps.ExpandedEnsemble(energies, expanded_values=expanded_values)
ensemble.initialize_restraints(input_data, options)
sampler = biceps.PosteriorSampler(ensemble, nreplicas, write_every=write_every)
sampler.sample(nsteps, attempt_lambda_swap_every=swap_every, swap_sigmas=1,
        find_optimal_nreplicas=find_optimal_nreplicas,
        attempt_move_state_every=attempt_move_state_every,
        attempt_move_sigma_every=attempt_move_sigma_every,
        burn=10000, print_freq=100,
        verbose=0, progress=1, multiprocess=multiprocess, capture_stdout=0)
#print(sampler.traj[1].__dict__["trajectory"])
print("sampler.acceptance_info = \n",sampler.acceptance_info)
biceps.toolbox.save_object(sampler, f"{outdir}/sampler_obj.pkl")

sampler.plot_exchange_info(xlim=(-100, nsteps), figname=f"{outdir}/lambda_swaps.png")

A = biceps.Analysis(sampler, outdir=outdir, MBAR=True)
A.plot_acceptance_trace()
try:
   A.plot_energy_trace()
except(Exception) as e: print(e)

####### Posterior Analysis #######
BS, pops = A.f_df, A.P_dP[:,len(expanded_values)-1]
K = len(expanded_values)-1
pops_std = A.P_dP[:,2*K]


if scale_energies == False:
    BS /= sampler.nreplicas
#print("from lam=0 to lam=1")

print("Predicted populations: ",pops)
print(f"BICePs Score = {BS[:,0]}")
print()

RMSE = np.sqrt(metrics.mean_squared_error(pops, populations))
print(f"\n\nRMSE = {RMSE}")
A.plot(plottype, figsize=(12,14), figname=f"BICePs.pdf", pad=0.35, plot_all_distributions=1)


check_convergence = 1
if check_convergence:

    converge_dir = f"{outdir}/convergence_results"
    biceps.toolbox.mkdir(converge_dir)
    traj = sampler.traj[-1].__dict__
    C = biceps.Convergence(traj, outdir=converge_dir, verbose=1)
    C.plot_traces(figname="traces.pdf", xlim=(0, traj["trajectory"][-1][0]), figsize=(8,4))
    C.get_autocorrelation_curves(method="block-avg-auto", nblocks=3, maxtau=500, figsize=(8,4))
    #C.process(plot=False)
    C.process(plot=True)

    #C = biceps.Convergence(traj, outdir=converge_dir)
    #C.plot_traces(figname="test.pdf", xlim=(0, traj["trajectory"][-1][0]), figsize=(8,30))
    #C.get_autocorrelation_curves(method="block-avg", nblocks=2, maxtau=1000, figsize=(8,30))
    #C.process()

    all_JSD = np.load(f"{converge_dir}/all_JSD.npy")
    all_JSDs = np.load(f"{converge_dir}/all_JSDs.npy")
    nblock=5
    nfold=10
    nround=100
    n_rest = all_JSDs.shape[0]
    JSD_dist = [[] for i in range(n_rest)]
    JSD_std = [[] for i in range(n_rest)]
    for rest in range(n_rest):
        for f in range(nfold):
            temp_JSD = []
            for r in range(nround):
                temp_JSD.append(all_JSDs[rest][f][r])
            JSD_dist[rest].append(np.mean(temp_JSD))
            JSD_std[rest].append(np.std(temp_JSD))
    lowers,uppers,values = [],[],[]
    for i in range(n_rest):
        index = -1 # at 100% of data
        values.append(all_JSD[i].transpose()[index])
        bounds = np.sort(all_JSDs[i]) # at 95% confidence interval
        # remove top 50 and lower 50
        print(bounds[:, int(nround*0.05)].shape)
        lowers.append(bounds[:, int(nround*0.05)][index])
        uppers.append(bounds[:, int(nround*0.95)][index])
    lowers,uppers,values = np.array(lowers),np.array(uppers),np.array(values)
    np.savetxt(f"{converge_dir}/uppers.txt", uppers)
    np.savetxt(f"{converge_dir}/lowers.txt", lowers)
    np.savetxt(f"{converge_dir}/values.txt", values)


    mid = (uppers + lowers)/2
    fig = plt.figure(figsize=(10,8))
    # the number of rows should be the data types
    n_rows, n_columns = len(options), 1
    gs = gridspec.GridSpec(n_rows, n_columns)
    ax = plt.subplot(gs[0])
    x = np.linspace(0, len(values)-1, len(values))
    ax.scatter(x=x, y=lowers, s=25, marker="_", color="k", edgecolor='black',)
    ax.scatter(x=x, y=uppers, s=50, marker="_", color="k", edgecolor='black',)
    ax.vlines(x=x, ymin=lowers, ymax=uppers, zorder=0)
    ax.scatter(x=x, y=values, s=25, color="r", edgecolor='black',)

    ax.set_ylabel("JSD", fontsize=16)
    ax.set_xlabel("Index", fontsize=16)
    ax.set_title("Using 100% of the dataset", fontsize=16)
    fig.tight_layout()
    fig.savefig(f"{converge_dir}/JSDs.png")


expanded_values = sampler.expanded_values




#exit()
try:
    A.plot_restraint_intensity()
    k = A.get_restraint_intensity()
except(Exception) as e:
    print(e)

model_scores = A.get_model_scores(verbose=0)
if scale_energies == False:
    BS /= sampler.nreplicas

approx_scores = A.approximate_scores(burn=1000)
print(approx_scores)


# NOTE: Get Prior MSM populations
noe = [pd.read_pickle(i) for i in biceps.toolbox.get_files(f"{data_dir}/*.noe")]
#  Get the ensemble average observable
noe_Exp = noe[0]["exp"].to_numpy()
noe_model = [i["model"].to_numpy() for i in noe]
#print(noe_model)

noe_prior = np.array([w*noe_model[i] for i,w in enumerate(prior_populations)]).sum(axis=0)
noe_reweighted = np.array([u.ufloat(w, pops_std[i])*noe_model[i] for i,w in enumerate(pops)]).sum(axis=0)

#print(noe_prior)
#print(noe_reweighted)

fig = plt.figure(figsize=(12,8))
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0,0])
#data1 = pd.concat([pd.read_pickle(i) for i in biceps.toolbox.get_files('cineromycin_B/J_NOE/*.noe')])
data = []
for i in range(len(noe_reweighted)):
    data.append({"index":i,
        "reweighted noe":noe_reweighted[i], "prior noe":noe_prior[i], "exp noe":noe_Exp[i],
        })
data1 = pd.DataFrame(data)

_data1 = data1.copy()
# NOTE: Sort by prior NOE,
#_data1 = data1.sort_values(["prior noe"])
#_data1 = _data1.reset_index()

reweighted_vals = np.array([val.nominal_value for val in _data1["reweighted noe"].to_numpy()])
reweighted_std = np.array([val.std_dev for val in _data1["reweighted noe"].to_numpy()])

#ax1 = data1.plot.scatter(x='index', y="reweighted noe", s=5, edgecolor='black', color="b", label="BICePs")
ax1.scatter(x=_data1.index.to_numpy(), y=_data1["prior noe"].to_numpy(), s=65, color="orange", label="Prior", edgecolor='black',)
ax1.scatter(x=_data1.index.to_numpy(), y=_data1["exp noe"].to_numpy(), s=45, color="r", label="Exp", edgecolor='black',)
#ax1.scatter(x=_data1.index.to_numpy(), y=_data1["reweighted noe"].to_numpy(), s=25, color="b", label="BICePs", edgecolor='black')
ax1.errorbar(x=_data1.index.to_numpy(), y=reweighted_vals, yerr=reweighted_std , fmt="o", capsize=5,
             markersize=5, markerfacecolor="b", label="BICePs", ecolor="k", markeredgecolor='black')
ax1.legend(fontsize=14)
ax1.set_xlabel(r"Index", size=16)
ax1.set_ylabel(r"NOE distance ($\AA$)", size=16)
axs = [ax1]
for n, ax in enumerate(axs):
    ax.text(-0.1, 1.0, string.ascii_lowercase[n], transform=ax.transAxes,
            size=20, weight='bold')
fig.tight_layout()
fig.savefig(f"{outdir}/reweighted_observables.png")

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
print(noe_reweighted)
print(noe_reweighted.shape)
_noe_reweighted = np.array([v.nominal_value for v in noe_reweighted])
chi2_exp = scipy.stats.chi2_contingency(np.stack([_noe_reweighted, noe_Exp], axis=1))
chi2_prior = scipy.stats.chi2_contingency(np.stack([_noe_reweighted, noe_prior], axis=1))
print("BICePs Vs Exp.  chi^2  %5.3f" % (chi2_exp[0]))
print("BICePs Vs Prior chi^2  %5.3f" % (chi2_prior[0]))
#sse = np.array([(noe_reweighted[i] - noe_Exp[i])**2/np.std([noe_reweighted[i],noe_Exp[i]]) for i in range(len(noe_Exp))]).sum()



ref_index = 0
print(f"data_deviations = {data_deviations}")
mlp = pd.concat([A.get_max_likelihood_parameters(model=i) for i in range(len(expanded_values[ref_index:]))])
mlp.reset_index(inplace=True, drop=True)
mlp.to_pickle(outdir+"/mlp.pkl")
print(mlp)


mlp = mlp.iloc[[1]]
columns = [col for col in mlp.columns.to_list() if "sigma" in col]
mlp = mlp[columns]
mlp.columns = [col.split("sigma_")[-1] for col in columns]
sigmas = mlp.to_numpy()[0]
if data_uncertainty == "single":
    total_sigma = sigmas[0]
    sigma_RMSE = np.abs(data_sigma - total_sigma)
else:
    total_sigma = np.sqrt(np.sum([sigma*sigma for sigma in sigmas])/(len(sigmas)-1))
    sigma_RMSE = np.sqrt(metrics.mean_squared_error(data_deviations, sigmas))
print(f"mlp =             {mlp}")
print(f"data_sigma = {data_sigma}")
print(f"total_sigma = {total_sigma}")
print(f"sigma_RMSE = {sigma_RMSE}")
# make figure
fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[(0,0)])
colors = ["c", "y", "r", "g"]
ticklabels = []
for c,col in enumerate(mlp.columns.to_list()):
    if "noe" in col:
        ax.bar(col, mlp[col], color=colors[0])
    elif "J" in col:
        ax.bar(col, mlp[col], color=colors[1])
    elif "H" in col:
        ax.bar(col, mlp[col], color=colors[2])
    #print(col)
    ticklabels.append(re.findall(r'\d+', col)[0])
if len(mlp.columns.to_list()) > 10:
    for r,row in enumerate(ax.get_xticklabels()):
        #row.set_text(row.get_text().split("sigma_")[-1])#.split("")[-1])
        if r %2 == 0:
            row.set_text(ticklabels[r])
        else:
            row.set_text("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8)
ax.set_xlim(-1, len(mlp.columns.to_list())+1)

ax.set_ylabel(r"$\hat{\sigma}$", fontsize=18, rotation=0, labelpad=20)
ax.set_xlabel(r"Data restraint index", fontsize=16)

visible_tick_labels = [ticklabels[i].split("noe")[-1].split("J")[-1].split("H")[-1] for i in range(len(ticklabels))]
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(visible_tick_labels, rotation=10, fontsize=10)


fig = ax.get_figure()
fig.tight_layout()
fig.savefig(f"{outdir}/max_aposteriori_sigmas.pdf")



columns = [col for col in mlp.columns.to_list() if "sigma_noe" in col]
sse = []
for k,col in enumerate(columns):
    sse.append(np.array([(noe_reweighted[i] - noe_Exp[i])**2/(mlp.iloc[0][col]**2 + mlp.iloc[-1][col]**2) for i in range(len(noe_Exp))]).sum())
#print(sse)

exit()


















