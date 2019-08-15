import numpy as np
import mdtraj as md
import os, sys

#t=md.load('traj0.xtc',top='conf.gro')
#a=np.load('Assignments.npy')
Ind = np.load('Ind_noe.npy')
for i in range(100):
        print i
	p=[]
	t=md.load('%d.pdb'%i)	
        d=md.compute_distances(t,Ind)
	
	p=np.reshape(d,(30,1))
	np.savetxt("../NOE/average_state%d.txt"%i,p)

sys.exit()




