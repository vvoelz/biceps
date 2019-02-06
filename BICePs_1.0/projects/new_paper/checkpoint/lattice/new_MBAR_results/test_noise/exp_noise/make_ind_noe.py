import os, sys
import numpy as np

exp_d = [3.101003,2.388998,2.681691,2.002356,1.985492,1.848842,1.448103,2.146103]

for l in range(19):
	sim_d=[]
	filename='lattice_model_%d.noe'%l
	with open(filename) as f:
		lines=f.readlines()
	line=''.join(lines)
	fields = line.strip().split('\n')
#print fields
	field=[]
	for i in range((len(fields))):
		field.append(fields[i].strip().split())
#print field
#sys.exit()

	for j in range(1,9):
		sim_d.append(float(field[j][7]))
	error=[]
	for k in range(len(exp_d)):
		error.append(exp_d[k] - sim_d[k])
	np.savetxt('error_%d.dat'%l,error)

#sys.exit()
