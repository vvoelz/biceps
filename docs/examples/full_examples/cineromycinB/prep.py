import glob
import mdtraj as md
import numpy as np
import biceps


# Compute model_data for NOE and J coupling
## NOE
data_dir = "../../datasets/cineromycin_B/"
outdir = "NOE/"
states = biceps.toolbox.get_files(data_dir+"cineromycinB_pdbs/*")
nstates = len(states)
ind=data_dir+'atom_indice_noe.txt'
ind_noe = ind
biceps.toolbox.mkdir(outdir)
model_data_NOE = biceps.toolbox.compute_distances(states, ind, outdir)
model_data_NOE = str(outdir+"*.txt")
exp_data_NOE = data_dir+"noe_distance.txt"

## J coupling
#### TODO: create function
ind = np.load(data_dir+'ind.npy')
indices = data_dir+'atom_indice_J.txt'
#print(ind)
outdir = "J/"
biceps.toolbox.mkdir(outdir)
karplus_key=np.loadtxt(data_dir+'Karplus.txt', dtype=str)
print('Karplus relations', karplus_key)
for i in range(nstates):
    J = biceps.toolbox.compute_nonaa_Jcoupling(
            data_dir+'cineromycinB_pdbs/%d.fixed.pdb'%i,
            index=ind,
            karplus_key=karplus_key)
    np.savetxt(outdir+'%d.txt'%i,J)

exp_data_J = data_dir+'exp_Jcoupling.txt'
model_data_J = data_dir+"J_coupling/*.txt"

# Now using biceps Preparation submodule
outdir = "J_NOE/"
biceps.toolbox.mkdir(outdir)
preparation = biceps.Observable.Preparation(nstates=nstates, top=states[0])
preparation.prep_noe(exp_data_NOE, model_data_NOE, indices=ind_noe, outdir=outdir, verbose=False)
preparation.prep_J(exp_data=exp_data_J, model_data=model_data_J, indices=indices, outdir=outdir, verbose=False)





