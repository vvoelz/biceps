import numpy as np
import pandas as pd
import biceps
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





