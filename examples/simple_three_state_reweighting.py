"""
A script implementing BICePs to reweight populations for a simple three state
toy model system.  Here, our prior comes from random generation of the Boltzmann
distribution and reweighting is performed using two experimental observables both set to 0.0 A.U.

For more details about this toy model systema and visual aids, please refer to
this notebook: `examples/enforcing_uniform_reference.ipynb`
"""

# Python libraries:{{{
import sys, os
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import biceps
from biceps.PosteriorSampler import u_kln_and_states_kn
from pymbar import MBAR
#:}}}

# Write noe files:{{{
def write_noe_files(weights, x, exp, dir):
    for i in range(len(weights)):
        model = pd.read_pickle("template.noe")
        _model = pd.DataFrame()
        for j in range(len(exp)):
            model["restraint_index"], model["model"], model["exp"] = 1+j, x[i][j], 0.0
            #model["restraint_index"], model["model"], model["exp"] = 1+j, x[i][j], exp[j]
            #_model = _model.append(model, ignore_index=True)
            _model = pd.concat([_model,model], ignore_index=True)
        _model.to_pickle(dir+"/%s.noe"%i)

#:}}}

################################################################################
####### Parameters #######
nStates,Nd = 3,2
n_xis,n_lambdas,nreplicas,nsteps,change_Nr_every,swap_every=1,2,8,1000000,0,0
multiprocess=1
σ_prior=0.161 # 0.08, 0.16
stat_model,data_uncertainty="Students","single"
data_likelihood = "gaussian" #"log normal" # "gaussian"

write_every = 10
attempt_move_state_every = 1
attempt_move_sigma_every = 1

##########################
# Make output directories
state_dir = f"{nStates}_state"
biceps.toolbox.mkdir(state_dir)

datapoints_dir = f"{state_dir}/{nStates}_state_{Nd}_datapoints"
biceps.toolbox.mkdir(datapoints_dir)

dir = f"{datapoints_dir}/Prior_error_{σ_prior}"
biceps.toolbox.mkdir(dir)

################################################################################
populations = np.array([0.5, 0.3, 0.2])
energies = -np.log(populations)

data_dir = dir+"/NOE"
biceps.toolbox.mkdir(data_dir)
forward_model_data = np.array([[1.0, 1.1],
                               [1.1, 1.2],
                               [2.0, 2.1]])
experiment = [0.0, 0.0]
write_noe_files(weights=energies, x=forward_model_data, exp=experiment, dir=data_dir)
################################################################################
input_data = biceps.toolbox.sort_data(data_dir)
print(f"Input data: {biceps.toolbox.list_extensions(input_data)}")
forward_model_data = np.array([pd.read_pickle(i)["model"].to_numpy() for i in biceps.toolbox.get_files(f"{data_dir}/*.noe")])
experiment = np.array([pd.read_pickle(i)["exp"].to_numpy() for i in biceps.toolbox.get_files(f"{data_dir}/0.noe")])[0]


outdir = f'{dir}/{stat_model}_{data_uncertainty}_sigma/{nsteps}_steps_{nreplicas}_replicas_{n_lambdas}_lam__swap_every_{swap_every}'
biceps.toolbox.mkdir(outdir)
print(f"nSteps of sampling: {nsteps}\nnReplicas: {nreplicas}")
lambda_values = np.linspace(0.0, 1.0, n_lambdas)

sigMin,sigMax,dsig = 0.001,200,1.02
arr = np.exp(np.arange(np.log(sigMin), np.log(sigMax), np.log(dsig)))
l = len(arr)
sigma_index = round(l*0.73)

beta,beta_index=(1., 2.0, 1),0
_arr = np.linspace(*beta)
_l = len(_arr)
print("Alpha starts here: ",_arr[beta_index])
phi,phi_index=(1., 2.0, 1),0
gamma,gamma_index=(1.0, 2.0, np.e),0

options = [dict(ref="uniform", stat_model=stat_model,
            sigma=(sigMin, sigMax, dsig), sigma_index=sigma_index, gamma=gamma,
            beta=beta, beta_index=beta_index, phi=phi, phi_index=phi_index,
            data_uncertainty=data_uncertainty, data_likelihood=data_likelihood,
            )]
print(pd.DataFrame(options))

ensemble = biceps.ExpandedEnsemble(lambda_values=lambda_values, energies=energies)
ensemble.initialize_restraints(input_data, options, verbose=1)
print("ensemble.expanded_values = ",ensemble.expanded_values)
sampler = biceps.PosteriorSampler(ensemble, nreplicas, change_Nr_every, write_every=write_every)
sampler.sample(nsteps, attempt_lambda_swap_every=swap_every, swap_sigmas=1,
        attempt_move_state_every=attempt_move_state_every,
        attempt_move_sigma_every=attempt_move_sigma_every,
        verbose=0, progress=1, multiprocess=multiprocess, capture_stdout=0)

expanded_values = sampler.expanded_values
A = biceps.Analysis(sampler, outdir=outdir, nstates=len(energies), MBAR=True)
A.plot(plottype="step", figsize=(12,14), figname=f"BICePs.pdf", pad=0.35, plot_all_distributions=1)
A.plot_energy_trace()
BS, pops = A.f_df, A.P_dP[:,len(expanded_values[:])-1]
BS /= sampler.nreplicas
K = len(expanded_values[:])-1
pops_std = A.P_dP[:,2*K]
print(f"Predicted populatins: {pops}")








