import sys, os
import mdtraj as md
import numpy as np
ind=np.loadtxt('atom_indice_noe.txt')
os.system('mkdir NOE')
#karplus_key=np.load('Karplus.npy')
for i in range(100):
    print i
    t = md.load('cineromycinB_pdbs/%d.fixed.pdb'%i)
    d=md.compute_distances(t,ind)*10.

#J = compute_nonaa_Jcoupling('cineromycinB_pdbs/%d.fixed.pdb'%i, index=index, karplus_key=karplus_key)
    np.savetxt('NOE/%d.txt'%i,d)
