import os, sys, pickle
import numpy as np
import biceps
import multiprocessing as mp

####### Data and Output Directories #######
energies = np.loadtxt('../../datasets/cineromycin_B/cineromycinB_QMenergies.dat')*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
energies -= energies.min()  # set ground state to zero, just in case
states = len(energies)
top = '../../datasets/cineromycin_B/cineromycinB_pdbs/0.fixed.pdb'
print(f"Possible input data extensions: {biceps.toolbox.list_possible_extensions()}")
#data = biceps.toolbox.sort_data('../../datasets/cineromycin_B/noe_J')
data = biceps.toolbox.sort_data('J_NOE')
res = biceps.toolbox.list_res(data)
extensions = biceps.toolbox.list_extensions(data)
print(f"Input data: {biceps.toolbox.list_extensions(data)}")
outdir = 'results_test'
biceps.toolbox.mkdir(outdir)
####### Parameters #######
nsteps=100000
print(f"nSteps of sampling: {nsteps}")
maxtau = 1000
n_lambdas = 2
lambda_values = np.linspace(0.0, 1.0, n_lambdas)
#ref = ['exp', 'exp']
ref = ['uniform', 'exp']
uncern = [[0.05, 20.0, 1.02], [0.05, 5.0, 1.02]]
for lam in lambda_values:
    #ensemble = biceps.Ensemble(lam, energies, top, verbose=True)
    ensemble = biceps.Ensemble(lam, energies, top, verbose=False)
    ensemble.initialize_restraints(exp_data=data, ref_pot=ref,
            uncern=uncern, gamma=[0.2, 5.0, 1.02], extensions=extensions)
    #print(ensemble.to_list())
    #exit()
    sampler = biceps.PosteriorSampler(ensemble.to_list())
    sampler.sample(nsteps=nsteps)
    sampler.traj.process_results(outdir+'/traj_lambda%2.2f.npz'%(lam))
    sampler.traj.read_results(os.path.join(outdir,
        'traj_lambda%2.2f.npz'%lam))
    outfilename = 'sampler_lambda%2.2f.pkl'%(lam)
    fout = open(os.path.join(outdir, outfilename), 'wb')
    pickle.dump(sampler, fout)
    fout.close()
    print('...Done.')
    exit()

'''
####### Convergence Check #######
C = biceps.Convergence(trajfile=outdir+"/traj_lambda0.00.npz", resultdir=outdir)
C.get_autocorrelation_curves(method="normal", maxtau=maxtau)
C.plot_auto_curve(fname="auto_curve.pdf", xlim=(0, maxtau))
C.process(nblock=5, nfold=10, nround=100, savefile=True,
    plot=True, block=True, normalize=True)

'''

####### Posterior Analysis #######
A = biceps.Analysis(states=states, resultdir=outdir,
    BSdir='BS.dat', popdir='populations.dat',
    picfile='BICePs.pdf')
A.plot()






