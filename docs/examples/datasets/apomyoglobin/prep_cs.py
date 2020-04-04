import glob
import mdtraj as md
import numpy as np
import biceps

# Compute model_data for CS
data_dir = "../../datasets/apomyoglobin/"
#outdir = "CS_test/"
outdir = "CS/"
biceps.toolbox.mkdir(outdir)
states = "../../datasets/apomyoglobin/pdb/*.pdb"
# NOTE check the pH and temp with simulation parameters
#biceps.toolbox.compute_chemicalshifts(states, temp=298, pH=6, outdir=outdir)
#model_data = str(outdir+"*.txt")
states = biceps.toolbox.get_files(states)
nstates = len(states)
preparation = biceps.Restraint.Preparation(nstates=nstates, top=states[0])
# NOTE may need to be fixed
exp_data = data_dir+"new_exp_H.txt"
model_data = str(data_dir+"model_data/H/*.txt")
indices = data_dir+"cs_indices_H.txt"
#biceps.toolbox.get_indices(states[0], top=states[0],
#        selection_expression="name H",out=indices,debug=True)
preparation.prep_cs(exp_data, model_data, indices, extension="H", outdir=outdir, verbose=False)
# NOTE may need to be fixed
exp_data = data_dir+"new_exp_N.txt"
model_data = str(data_dir+"model_data/N/*.txt")
indices = data_dir+"cs_indices_N.txt"
preparation.prep_cs(exp_data, model_data, indices, extension="N", outdir=outdir, verbose=False)
# NOTE may need to be fixed
exp_data = data_dir+"new_exp_Ca.txt"
model_data = str(data_dir+"model_data/Ca/*.txt")
indices = data_dir+"cs_indices_Ca.txt"
preparation.prep_cs(exp_data, model_data, indices, extension="Ca", outdir=outdir, verbose=False)


exit()
# load in a structure and find the indices that correspond to H, Ca, N


ind=data_dir+'atom_indice_noe.txt'
biceps.toolbox.mkdir(outdir)
model_data = biceps.toolbox.compute_distances(states, ind, outdir)
model_data = str(outdir+"*.txt")




