import glob, os, sys, pickle
import mdtraj as md
import numpy as np
import pandas as pd
import biceps

## Compute model_data for NOE and J coupling
### NOE
#data_dir = "../../datasets/cineromycin_B/"
#outdir = "NOE/"
#states = biceps.toolbox.get_files(data_dir+"cineromycinB_pdbs/*")
#nstates = len(states)
#ind=data_dir+'atom_indice_noe.txt'
#ind_noe = ind
#biceps.toolbox.mkdir(outdir)
#model_data_NOE = biceps.toolbox.compute_distances(states, ind, outdir)
#model_data_NOE = str(outdir+"*.txt")
#exp_data_NOE = data_dir+"noe_distance.txt"
#
### J coupling
##### TODO: create function
#ind = np.load(data_dir+'ind.npy')
#indices = data_dir+'atom_indice_J.txt'
##print(ind)
#outdir = "J/"
#biceps.toolbox.mkdir(outdir)
#karplus_key=np.loadtxt(data_dir+'Karplus.txt', dtype=str)
#print('Karplus relations', karplus_key)
#biceps.toolbox.compute_nonaa_scalar_coupling(
#        states=data_dir+'cineromycinB_pdbs/*.fixed.pdb',
#        index=ind, karplus_key=karplus_key, outdir=outdir)
#exp_data_J = data_dir+'exp_Jcoupling.txt'
#model_data_J = data_dir+"J_coupling/*.txt"
#
## Now using biceps Preparation submodule
#outdir = "J_NOE/"
#biceps.toolbox.mkdir(outdir)
#preparation = biceps.Observable.Preparation(nstates=nstates, top=states[0])
#preparation.prep_noe(exp_data_NOE, model_data_NOE, indices=ind_noe, outdir=outdir, verbose=False)
#preparation.prep_J(exp_data=exp_data_J, model_data=model_data_J, indices=indices, outdir=outdir, verbose=False)
##exit()

####### Data and Output Directories #######
energies = np.loadtxt('cineromycin_B/cineromycinB_QMenergies.dat')*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
energies -= energies.min()  # set ground state to zero, just in case
states = len(energies)
print(f"Possible input data extensions: {biceps.toolbox.list_possible_extensions()}")
input_data = biceps.toolbox.sort_data('cineromycin_B/J_NOE')
print(f"Input data: {biceps.toolbox.list_extensions(input_data)}")
####### Parameters #######
nsteps=100000
print(f"nSteps of sampling: {nsteps}")
maxtau = 1000
n_lambdas = 3
outdir = '%s_steps_%s_lam'%(nsteps, n_lambdas)
biceps.toolbox.mkdir(outdir)
lambda_values = np.linspace(0.0, 1.0, n_lambdas)
parameters = [dict(ref="uniform", sigma=(0.05, 20.0, 1.02)),
        dict(ref="exp", sigma=(0.05, 5.0, 1.02), gamma=(0.2, 5.0, 1.02)),]
print(pd.DataFrame(parameters))
###### Multiprocessing Lambda values #######
@biceps.multiprocess(iterable=lambda_values)
def mp_lambdas(lam):
    ensemble = biceps.Ensemble(lam, energies)
    ensemble.initialize_restraints(input_data, parameters)
    sampler = biceps.PosteriorSampler(ensemble.to_list())
    sampler.sample(nsteps=nsteps, print_freq=1000, verbose=False)
    sampler.traj.process_results(outdir+'/traj_lambda%2.2f.npz'%(lam))
    filename = outdir+'/sampler_lambda%2.2f.pkl'%(lam)
    biceps.toolbox.save_object(sampler, filename)

'''
####### Convergence Check #######
C = biceps.Convergence(trajfile=outdir+"/traj_lambda0.00.npz", resultdir=outdir)
C.get_autocorrelation_curves(method="auto", maxtau=maxtau)
C.plot_auto_curve(fname="auto_curve.pdf", xlim=(0, maxtau))
C.process(nblock=5, nfold=10, nround=100, savefile=True,
    plot=True, block=True, normalize=True)
'''

####### Posterior Analysis #######
A = biceps.Analysis(nstates=states, outdir=outdir)
A.plot()









