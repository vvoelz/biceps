import os, sys, pickle
import numpy as np
import biceps

# What possible experimental data does biceps accept?
print(f"Possible data restraints: {biceps.toolbox.list_possible_restraints()}")
print(f"Possible input data extensions: {biceps.toolbox.list_possible_extensions()}")
####### Data and Output Directories #######
energies = np.loadtxt('albocycline/albocycline_QMenergies.dat')
top = 'albocycline/pdbs/0.pdb'
data = biceps.toolbox.sort_data('albocycline/noe_j')
extensions = biceps.toolbox.list_extensions(data)
print(f"Input data: {biceps.toolbox.list_extensions(data)}")
res = biceps.toolbox.list_res(data)
outdir = 'results'
biceps.toolbox.mkdir(outdir)
####### Parameters #######
nsteps=1000000
maxtau = 1000
n_lambdas = 3
lambda_values = np.linspace(0.0, 1.0, n_lambdas)
print(lambda_values)
ref = ['uniform', 'exp']
uncern = [[0.05, 20.0, 1.02], [0.05, 5.0, 1.02]]
####### MCMC Simulations #######
for lam in lambda_values:
    print(f"lambda: {lam}")
    ensemble = biceps.Ensemble(lam, energies, top, verbose=False)
    ensemble.initialize_restraints(exp_data=data, ref_pot=ref,
            uncern=uncern, gamma=[0.2, 5.0, 1.02], extensions=extensions)
    sampler = biceps.PosteriorSampler(ensemble.to_list())
    sampler.sample(nsteps=nsteps)
    sampler.traj.process_results(outdir+'/traj_lambda%2.2f.npz'%(lam))
    outfilename = 'sampler_lambda%2.2f.pkl'%(lam)
    fout = open(os.path.join(outdir, outfilename), 'wb')
    pickle.dump(sampler, fout)
    fout.close()
    print('...Done.')

####### Convergence Check #######
C = biceps.Convergence(trajfile=outdir+"/traj_lambda0.00.npz")
C.get_autocorrelation_curves(method="normal", maxtau=maxtau)
C.plot_auto_curve(fname="auto_curve.pdf", xlim=(0, maxtau))
C.process(nblock=5, nfold=10, nround=100, savefile=True,
    plot=True, block=True, normalize=True)

####### Posterior Analysis #######
A = biceps.Analysis(states=100, resultdir=outdir,
    BSdir='BS.dat', popdir='populations.dat',
    picfile='BICePs.pdf')
A.plot()




