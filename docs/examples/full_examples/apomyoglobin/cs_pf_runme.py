import os, sys, pickle
import numpy as np
import biceps

####### Data and Output Directories #######
print(f"Possible input data extensions: {biceps.toolbox.list_possible_extensions()}")
T=[0,1,4,9,10,12,14,16,18,19,20,21,24]
states=len(T)
datadir="apomyoglobin/"
top=datadir+'pdb/T1/state0.pdb'
#dataFiles = datadir+'CS_PF'
dataFiles = 'CS_PF'
data = biceps.toolbox.sort_data(dataFiles)
res = biceps.toolbox.list_res(data)
extensions = biceps.toolbox.list_extensions(data)
print(f"Input data: {biceps.toolbox.list_extensions(data)}")
energies_filename = datadir+'energy_model_1.txt'
energies = np.loadtxt(energies_filename)
energies -= energies.min() # set ground state to zero, just in case
outdir = "results"
biceps.toolbox.mkdir(outdir)
####### Parameters #######
nsteps = 10000000
print(f"nSteps of sampling: {nsteps}")
maxtau = 1000
ref=['exp','exp','exp','exp']
n_lambdas = 2
lambda_values = np.linspace(0.0, 1.0, n_lambdas)

####### MCMC Simulations #######
for lam in lambda_values:
    print(f"lambda: {lam}")
    ensemble = biceps.Ensemble(lam, energies, top, verbose=True)
    ensemble.initialize_restraints(exp_data=data, ref_pot=ref, pf_prior=datadir+'b15.npy',
            Ncs_fi=datadir+'input/Nc', Nhs_fi=datadir+'input/Nh', state=T, extensions=extensions, debug=True)
    sampler = biceps.PosteriorSampler(ensemble.to_list())
    sampler.sample(nsteps=nsteps)
    sampler.traj.process_results(outdir+'/traj_lambda%2.2f.npz'%(lam))
    outfilename = 'sampler_lambda%2.2f.pkl'%(lam)
    fout = open(os.path.join(outdir, outfilename), 'wb')
    pickle.dump(sampler, fout)
    fout.close()
    print('...Done.')


A = biceps.Analysis(states=states, resultdir=outdir,
  BSdir='BS.dat', popdir='populations.dat',
  picfile='BICePs.pdf')
A.plot()


