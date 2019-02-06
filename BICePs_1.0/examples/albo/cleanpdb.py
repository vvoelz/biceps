import os, sys
import numpy as np
#filename='albo_37.pdb'
filename=raw_input("Please tell me your pdb filename:  ")
f=open(filename,'r')
lines=f.readlines()
f.close()
pdb=[]
for i in range(len(lines)):
	if lines[i][:5] == 'ATOM ':
		pdb.append(lines[i].strip())
outfile='clean.pdb'
fid=open(outfile,'w')
for i in pdb:
	fid.writelines(i+'\n')
fid.close()
	

