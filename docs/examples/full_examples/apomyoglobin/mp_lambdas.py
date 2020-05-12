import numpy as np
import biceps
import multiprocessing as mp

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
outdir = "_results"
biceps.toolbox.mkdir(outdir)
####### Parameters #######
nsteps = 1000000
print(f"nSteps of sampling: {nsteps}")
maxtau = 1000
n_lambdas = 2
lambda_values = np.linspace(0.0, 1.0, n_lambdas)
parameters = [
        {"ref": 'exp', "weight": 1/3},
        {"ref": 'exp', "weight": 1/3},
        {"ref": 'exp', "weight": 1/3},
        {"ref": 'exp', "weight": 1, "pf_prior": datadir+'b15.npy',
            "Ncs_fi": datadir+'input/Nc', "Nhs_fi": datadir+'input/Nh', "states": T}
        ]
####### MCMC Simulations #######
def mp_lambdas(Lambda):
    print(f"lambda: {Lambda}")
    ensemble = biceps.Ensemble(Lambda, energies)
    ensemble.initialize_restraints(input_data, parameters)
    sampler = biceps.PosteriorSampler(ensemble.to_list())
    sampler.sample(nsteps, verbose=False)
    sampler.traj.process_results(outdir+'/traj_lambda%2.2f.npz'%(Lambda))
    filename = outdir+'/sampler_lambda%2.2f.pkl'%(lam)
    biceps.toolbox.save_object(sampler, filename)
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


A = biceps.Analysis(nstates=len(energies), outdir=outdir)
A.plot()



