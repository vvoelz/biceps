import numpy as np
import biceps

####### Data and Output Directories #######
print(f"Possible input data extensions: {biceps.toolbox.list_possible_extensions()}")
T=[0,1,4,9,10,12,14,16,18,19,20,21,24]
states=len(T)
datadir="apomyoglobin/"
top=datadir+'pdb/T1/state0.pdb'
dataFiles = datadir+'new_CS_PF'
input_data = biceps.toolbox.sort_data(dataFiles)
print(f"Input data: {biceps.toolbox.list_extensions(input_data)}")
energies_filename = datadir+'energy_model_1.txt'
energies = np.loadtxt(energies_filename)
energies -= energies.min() # set ground state to zero, just in case
states = len(energies)
####### Parameters #######
nsteps = 1000000
print(f"nSteps of sampling: {nsteps}")
maxtau = 1000
n_lambdas = 2
outdir = '%s_steps_%s_lam'%(nsteps, n_lambdas)
biceps.toolbox.mkdir(outdir)
lambda_values = np.linspace(0.0, 1.0, n_lambdas)
parameters = [
        {"ref": 'exp', "weight": 1/3},
        {"ref": 'exp', "weight": 1/3},
        {"ref": 'exp', "weight": 1/3},
        {"ref": 'exp', "weight": 1, "pf_prior": datadir+'b15.npy',
            "Ncs_fi": datadir+'input/Nc', "Nhs_fi": datadir+'input/Nh', "states": T}
        ]
####### MCMC Simulations #######
@biceps.multiprocess(iterable=lambda_values)
def mp_lambdas(lam):
    ensemble = biceps.Ensemble(lam, energies)
    ensemble.initialize_restraints(input_data, parameters)
    sampler = biceps.PosteriorSampler(ensemble)
    sampler.sample(nsteps, verbose=False)
    sampler.traj.process_results(f"{outdir}/traj_lambda{lam}.npz")

####### Posterior Analysis #######
A = biceps.Analysis(trajs=f"{outdir}/traj*.npz", nstates=states)
A.plot()



