import os, sys, pickle
import numpy as np
import biceps

####### Data and Output Directories #######
energies = np.loadtxt('energy_test.txt', dtype=float)*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
energies -= energies.min()  # set ground state to zero, just in case
top = 'pdb/state0.pdb'
print(f"Possible input data extensions: {biceps.toolbox.list_possible_extensions()}")
data = biceps.toolbox.sort_data('CS')
res = biceps.toolbox.list_res(data)
extensions = biceps.toolbox.list_extensions(data)
print(f"Input data: {biceps.toolbox.list_extensions(data)}")
outdir = 'results_ref_normal'
biceps.toolbox.mkdir(outdir)
####### Parameters #######
nsteps=10000
maxtau = 1000
lambda_values = [0.0, 0.5, 1.0]
ref = ['uniform', 'exp', 'exp']
uncern = [[0.05, 20.0, 1.02], [0.05, 5.0, 1.02], [0.05, 5.0, 1.02]]

####### MCMC Simulations #######
for lam in lambda_values:
    ensemble = biceps.Ensemble(lam, energies, top)
    ensemble.initialize_restraints(exp_data=data, ref_pot=ref,
            uncern=uncern, gamma=[0.2, 5.0, 1.02], extensions=extensions)
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

'''
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
'''


'''
import mdtraj as md
import numpy as np
data_dir = "../../datasets/cineromycin_B/"
ind=np.loadtxt(data_dir+'atom_indice_noe.txt')
print("indices", ind)
os.system(data_dir+'mkdir NOE')
for i in range(100):    # 100 clustered states
    t = md.load(data_dir+'cineromycinB_pdbs/%d.fixed.pdb'%i)
    d=md.compute_distances(t,ind)*10.     # convert nm to Å
    np.savetxt(data_dir+'NOE/%d.txt'%i,d)
print("Done!")
path = data_dir+'NOE/*txt'
states = 100
indices = data_dir+'atom_indice_noe.txt'
exp_data = data_dir+'noe_distance.txt'
top = data_dir+'cineromycinB_pdbs/0.fixed.pdb'
out_dir = data_dir+'noe_J'
p = biceps.Preparation('noe',states=states,indices=indices,exp_data=exp_data,top=top,data_dir=path)   # 'noe' scheme is selected
p.write(out_dir=out_dir)
fin = open(data_dir+'noe_J/0.noe','r')
text = fin.read()
fin.close()
print(text)

ind=np.load(data_dir+'ind.npy')
print('index', ind)
karplus_key=np.loadtxt(data_dir+'Karplus.txt', dtype=str)
print('Karplus relations', karplus_key)
for i in range(100):    # 100 clustered states
    J = biceps.toolbox.compute_nonaa_Jcoupling(data_dir+'cineromycinB_pdbs/%d.fixed.pdb'%i, index=ind, karplus_key=karplus_key)
    np.savetxt(data_dir+'J_coupling/%d.txt'%i,J)
path = data_dir+'J_coupling/*txt'
states = 100
indices = data_dir+'atom_indice_J.txt'
exp_data = data_dir+'exp_Jcoupling.txt'
top = data_dir+'cineromycinB_pdbs/0.fixed.pdb'
out_dir = data_dir+'noe_J'
p = biceps.Preparation('J',states=states,indices=indices,exp_data=exp_data,top=top,data_dir=path)   # 'J' scheme is selected
p.write(out_dir=out_dir)
fin = open(data_dir+'noe_J/0.J','r')
text = fin.read()
fin.close()
print(text)
'''




