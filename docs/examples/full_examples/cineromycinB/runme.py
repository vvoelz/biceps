import os, sys, pickle #NOTE: cPickle has issues with Python 3
import numpy as np
sys.path.append("../")
import biceps

####### Data and Output Directories #######
energies = np.loadtxt('../../datasets/cineromycin_B/cineromycinB_QMenergies.dat', dtype=float)*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
energies -= energies.min()  # set ground state to zero, just in case
data = biceps.toolbox.sort_data('../../datasets/cineromycin_B/noe_J')
res = biceps.toolbox.list_res(data)
outdir = 'results_ref_normal'
biceps.toolbox.mkdir(outdir)
####### Parameters #######
nsteps=1000000
maxtau = 1000
lambda_values = [0.0, 0.5, 1.0]
ref = ['uniform', 'exp']
uncern = [[0.05, 20.0, 1.02], [0.05, 5.0, 1.02]]

####### MCMC Simulations #######
for lam in lambda_values:
    ensemble = []
    for i in range(energies.shape[0]):
        ensemble.append([])
        for k in range(len(data[0])):
            File = data[i][k]
            R = biceps.init_res(PDB_filename='../../datasets/cineromycin_B/cineromycinB_pdbs/0.fixed.pdb', lam=lam,
                energy=energies[i], ref=ref[k], data=File,
                uncern=uncern[k], gamma=[0.2, 5.0, 1.02])
            ensemble[-1].append(R)
    sampler = biceps.PosteriorSampler(ensemble)
    sampler.sample(nsteps=nsteps)
    sampler.traj.process_results(outdir+'/traj_lambda%2.2f.npz'%(lam))
    sampler.traj.read_results(os.path.join(outdir,
        'traj_lambda%2.2f.npz'%lam))
    outfilename = 'sampler_lambda%2.2f.pkl'%(lam)
    fout = open(os.path.join(outdir, outfilename), 'wb')
    pickle.dump(sampler, fout)
    fout.close()
    print('...Done.')


####### Convergence Check #######
C = biceps.Convergence(trajfile=outdir+"/traj_lambda0.00.npz", resultdir=outdir)
C.get_autocorrelation_curves(method="normal", maxtau=maxtau)
C.plot_auto_curve(fname="auto_curve.pdf", xlim=(0, maxtau))
C.process(nblock=5, nfold=10, nround=100, savefile=True,
    plot=True, block=True, normalize=True)

####### Posterior Analysis #######
A = biceps.Analysis(states=100, resultdir=outdir,
    BSdir='BS.dat', popdir='populations.dat',
    picfile='BICePs.pdf')
A.plot()



