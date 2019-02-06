import sys, os
import numpy as np
filename='apoMb.pf'

with open(filename) as f:
	lines=f.readlines()
line=''.join(lines)
fields = line.strip().split('\n')
#print len(fields)
#sys.exit()
exp=[]
field=[]
for i in range(1,(len(fields))):
	field.append(fields[i].strip().split())
for j in range(len(field)):
	exp.append(float(field[j][3]))
np.save('exp_data.npy',exp)
