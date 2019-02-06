import numpy as np
import mdtraj as md
import os, sys
if not os.path.exists('NOE'):
	os.mkdir('NOE')

t=md.load('traj.xtc',top='xtc.gro')	# traj and top files
Ind=np.array([[43,15],[93,65],[143,115],[43,16],[93,66],[143,116],[47,63],[97,113],[147,13],[47,64],[97,114],[147,14],[48,43],[48,45],[98,93],[98,95],[148,143],[148,145]])	# Indices for distances (0 based!)

for i in range(100):	# number of states
        b=np.load('state/state%d.npy'%i) 
	if len(b) != 0:
		print i
	        p=[]
	        for z in b:
        	        d=md.compute_distances(t[z],Ind)
                	p.append(d)
	        u=[]
	        v=[]
        	w=[]
        	for l in range(len(Ind)):
			r=[]
                	for j in range(len(p)):
                        	r.append(p[j][0][l])
                	f=np.mean(r)
                	u.append(f)
                	e=[]
                	for n in r:
                        	s=n**-6.0
                        	e.append(s)
                	k=np.mean(e)**(-1./6.)
                	w.append(k)

        	np.savetxt("NOE/average_whole_state%d.txt"%i,u)
        	np.savetxt("NOE/rminus6_whole_state%d.txt"%i,w)	# rminus6 file will be used in BICePs as model distances


