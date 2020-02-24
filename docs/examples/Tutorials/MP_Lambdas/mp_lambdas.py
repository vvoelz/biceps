import os, sys, pickle #NOTE: cPickle has issues with Python 3
import numpy as np
sys.path.append("../")
import biceps
import multiprocessing as mp

####### Data and Output Directories #######
energies = np.loadtxt('energy.dat')
data = biceps.sort_data('noe')
res = biceps.list_res(data)
outdir = 'results_ref_normal'
if not os.path.exists(outdir):
    os.mkdir(outdir)
####### Parameters #######
nsteps=1000000
maxtau = 1000
lambda_values = [0.0, 0.5, 1.0]
ref = ['uniform', 'exp']
uncern = [[0.05, 20.0, 1.02], [0.05, 5.0, 1.02]]
####### Multiprocessing Lambda values #######
def mp_lambdas(Lambda):
    ####### MCMC Simulations #######
    ensemble = []
    for i in range(energies.shape[0]):
        ensemble.append([])
        for k in range(len(data[0])):
            File = data[i][k]
            R = biceps.init_res(PDB_filename='cineromycinB_pdbs/0.fixed.pdb', lam=lam,
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
# Check the number of CPU's available
print("Number of CPU's: %s"%(mp.cpu_count()))
#p = mp.Pool(processes=mp.cpu_count()-1) # knows the number of CPU's to allocate
p = mp.Pool(processes=mp.cpu_count()) # knows the number of CPU's to allocate
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

####### Convergence Check #######
C = biceps.Convergence(trajfile=outdir+"/traj_lambda0.00.npz")
C.plot_traces(fname="traces.png", xlim=(0, nsteps))
C.get_autocorrelation_curves(method="normal", maxtau=maxtau)
C.plot_auto_curve(fname="auto_curve.pdf", xlim=(0, maxtau))
C.process(nblock=5, nfold=10, nround=100, savefile=True,
    plot=True, block=True, normalize=True)


####### Posterior Analysis #######
A = biceps.Analysis(states=100, resultdir=outdir,
    BSdir='BS.dat', popdir='populations.dat',
    picfile='BICePs.pdf')
A.plot()






