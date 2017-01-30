import os, sys
import numpy as np
#filename='albo_37.pdb'
filename=raw_input("Please tell me your pdb filename:  ")
print 'Working...'
f=open(filename,'r')
lines=f.readlines()
f.close()
pdb=[]
for i in range(len(lines)):
	if lines[i][:5] == 'ATOM ' or lines[i][:6] == 'HETATM' or lines[i][:3] == 'TER':
		pdb.append(lines[i].strip())
outfile='new.pdb'
fid=open(outfile,'w')
for i in pdb:
	fid.writelines(i+'\n')
fid.close()
print 'Done!'	

