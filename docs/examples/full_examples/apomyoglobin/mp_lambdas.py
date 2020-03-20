import os, sys, pickle
import numpy as np
import biceps
import multiprocessing as mp

T=[0,1,4,9,10,12,14,16,18,19,20,21,24]
states=len(T)
top='pdb/T1/state0.pdb'
dataFiles = 'new_PF_CS'
out_dir=dataFiles
data = biceps.toolbox.sort_data(dataFiles)
#T = [0,1,4,9,10,12,14,16,18,19,20,21,24]
energies_filename = 'energy_model_1.txt'
energies = np.loadtxt(energies_filename)
energies -= energies.min() # set ground state to zero, just in case
outdir = "results"
biceps.toolbox.mkdir(outdir)
nsteps = 10000000
ref=['exp','exp','exp','exp']
lambda_values = [0.0,1.0]
####### Multiprocessing Lambda values #######
def mp_lambdas(Lambda):
    print(f"lambda: {Lambda}")
    ensemble = []
    for i in range(energies.shape[0]):
        ensemble.append([])
        for k in range(len(data[0])):
            File = data[i][k]
            R=biceps.init_res(PDB_filename=top,lam=Lambda,energy=energies[i],data=File,ref=ref[k],Ncs_fi='input/Nc', Nhs_fi='input/Nh',state=T[i])
            ensemble[-1].append(R)
    sampler = biceps.PosteriorSampler(ensemble,pf_prior = 'b15.npy')
    sampler.sample(nsteps=nsteps)
    sampler.traj.process_results(outdir+'/traj_lambda%2.2f.npz'%(Lambda))
    sampler.traj.read_results(os.path.join(outdir,
      'traj_lambda%2.2f.npz'%Lambda))
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



