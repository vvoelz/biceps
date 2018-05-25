import numpy as np
import mdtraj as md

Ind=np.array([[0,9],[0,11],[2,9],[2,11],[4,9],[4,11],[6,9],[6,11]])

for i in range(15037):
	t=md.load('../Gens/micro/%d.pdb'%i)
	d=md.compute_distances(t,Ind)
	token=0
	for j in d[0]:
		if j - 0.1 == 0.0:
			token += 1
	print token
#	print d
#	np.savetxt('dis_state%d.txt'%i,d)
