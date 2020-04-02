import os, sys, pickle
import numpy as np
import biceps

####### Data and Output Directories #######
print(f"Possible input data extensions: {biceps.toolbox.list_possible_extensions()}")
T=[0,1,4,9,10,12,14,16,18,19,20,21,24]
states=len(T)
datadir="apomyoglobin/"
top=datadir+'pdb/T1/state0.pdb'
dataFiles = datadir+'new_CS_PF'
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
nsteps = 1000000
print(f"nSteps of sampling: {nsteps}")
maxtau = 1000
ref=['exp','exp','exp','exp']
weights=[1/3, 1/3, 1/3, 1]
n_lambdas = 2
lambda_values = np.linspace(0.0, 1.0, n_lambdas)

####### MCMC Simulations #######
def mp_lambdas(Lambda):
    print(f"lambda: {Lambda}")
    ensemble = biceps.Ensemble(Lambda, energies, top, verbose=False)
    ensemble.initialize_restraints(input_data=data, ref_pot=ref, pf_prior=datadir+'b15.npy',
            Ncs_fi=datadir+'input/Nc', Nhs_fi=datadir+'input/Nh', state=T,
            extensions=extensions, weights=weights, debug=False)
    sampler = biceps.PosteriorSampler(ensemble.to_list())
    sampler.sample(nsteps=nsteps, verbose=False)
    sampler.traj.process_results(outdir+'/traj_lambda%2.2f.npz'%(Lambda))
    outfilename = 'sampler_lambda%2.2f.pkl'%(Lambda)
    fout = open(os.path.join(outdir, outfilename), 'wb')
    pickle.dump(sampler, fout)
    fout.close()
    print('...Done.')

# Check the number of CPU's available
print("Number of CPU's: %s"%(mp.cpu_count()))
p = mp.Pool(processes=len(lambda_values)) # knows the number of CPU's to allocate
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

A = biceps.Analysis(states=states, resultdir=outdir,
  BSdir='BS.dat', popdir='populations.dat',
  picfile='BICePs.pdf')
A.plot()



