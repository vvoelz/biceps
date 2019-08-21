import sys, os
sys.path.append('new_src')
from toolbox import *
index=np.load('ind.npy')
karplus_key=np.load('Karplus.npy')
for i in range(100):
    print i
    J = compute_nonaa_Jcoupling('cineromycinB_pdbs/%d.fixed.pdb'%i, index=index, karplus_key=karplus_key)
    np.savetxt('J_coupling/%d.txt'%i,J)
