import os, sys, pickle
import numpy as np
import biceps
import multiprocessing as mp

####### Data and Output Directories #######
print(f"Possible input data extensions: {biceps.toolbox.list_possible_extensions()}")
top ='albocycline/pdbs/0.pdb'
energies = np.loadtxt('albocycline/albocycline_QMenergies.dat')*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
energies -= energies.min()  # set ground state to zero, just in case
dataFiles = 'albocycline/J_NOE'
data = biceps.toolbox.sort_data(dataFiles)
extensions = biceps.toolbox.list_extensions(data)
print(f"Input data: {biceps.toolbox.list_extensions(data)}")
outdir = 'results'
biceps.toolbox.mkdir(outdir)

####### Parameters #######
nsteps=10000000
print(f"nSteps of sampling: {nsteps}")
maxtau = 1000
n_lambdas = 2
lambda_values = np.linspace(0.0, 1.0, n_lambdas)
ref = ['uniform', 'exp']
uncern = [[0.05, 20.0, 1.02], [0.05, 5.0, 1.02]]

####### Multiprocessing Lambda values #######
def mp_lambdas(Lambda):
    print(f"lambda: {Lambda}")
    ensemble = biceps.Ensemble(Lambda, energies, top, verbose=False)
    ensemble.initialize_restraints(input_data=data, ref_pot=ref,
            gamma=[0.2, 5.0, 1.01],extensions=extensions, debug=False)
    sampler = biceps.PosteriorSampler(ensemble.to_list())
    sampler.sample(nsteps=nsteps, verbose=False)
    sampler.traj.process_results(outdir+'/traj_lambda%2.2f.npz'%(Lambda))
    filename = outdir+'/sampler_lambda%2.2f.pkl'%(lam)
    biceps.toolbox.save_object(sampler, filename)
    print('...Done.')

# Check the number of CPU's available
print("Number of CPU's: %s"%(mp.cpu_count()))
p = mp.Pool(processes=len(lambda_values)) # knows the number of CPU's to allocate
print("Number of processes: {n_lambdas}")
#p = mp.Pool(processes=mp.cpu_count()) # knows the number of CPU's to allocate
#print("Process ID's: %s"%get_processes(p, n=lam))
jobs = []
for lam in lambda_values:
    process = p.Process(target=mp_lambdas, args=(lam,))
    jobs.append(process)
    jobs[-1].start() # Start the processes
    active_processors = [jobs[i].is_alive() for i in range(len(jobs))]
    #print("Active Processors: %s"%active_processors)
    if (len(active_processors) == mp.cpu_count()-1) and all(active_processors) == True:
        #print("Waiting until a processor becomes available...")
        while all(active_processors) == True:
            active_processors = [jobs[i].is_alive() for i in range(len(jobs))]
        #print(active_processors)
        inactive = int(np.where(np.array(active_processors) == False)[0])
        jobs[inactive].terminate()
        jobs.remove(jobs[inactive])
for job in jobs:
    job.join() # will wait until the execution is over...
p.close()

'''
####### Convergence Check #######
C = biceps.Convergence(trajfile=outdir+"/traj_lambda0.00.npz")
C.get_autocorrelation_curves(method="normal", maxtau=maxtau)
C.plot_auto_curve(fname="auto_curve.pdf", xlim=(0, maxtau))
C.process(nblock=5, nfold=10, nround=100, savefile=True,
    plot=True, block=True, normalize=True)
'''


####### Posterior Analysis #######
A = biceps.Analysis(states=100, resultdir=outdir+"/",
    BSdir='BS.dat', popdir='populations.dat',
    picfile='BICePs.pdf')
A.plot()






