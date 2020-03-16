import glob
import mdtraj as md
import numpy as np
import biceps


data_dir = "../../datasets/cineromycin_B/"
outdir = "NOE/"
states = biceps.toolbox.get_files(data_dir+"cineromycinB_pdbs/*")
nstates = len(states)
ind=data_dir+'atom_indice_noe.txt'
biceps.toolbox.mkdir(outdir)
model_data = biceps.toolbox.compute_distances(states, ind, outdir)
model_data = str(outdir+"*.txt")
exp_data = data_dir+"noe_distance.txt"
preparation = biceps.Observable.Preparation(nstates=nstates, indices=ind, top=states[0])
preparation.prep_noe(exp_data, model_data, outdir=outdir, verbose=False)
#preparation.prep_cs(exp_data=exp_data, model_data=model_data, extension="H", outdir=outdir)
exit()




outdir = "J/"



'''
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



