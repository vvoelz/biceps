import os, sys
import numpy as np
import argparse
"""
This script could keep "ATOM" lines in any pdb file and store it as a new pdb file. -Yunhui Ge
"""
parser=argparse.ArgumentParser()
parser.add_argument("inputpdb",help='input file name (.pdb file)')
args=parser.parse_args()
filename=args.inputpdb
f=open(filename,'r')
lines=f.readlines()
f.close()
pdb=[]
for i in range(len(lines)):
	if lines[i][:5] == 'ATOM ':
		pdb.append(lines[i].strip())
outfile=filename[:-4]+'.new.pdb'
fid=open(outfile,'w')
for i in pdb:
	fid.writelines(i+'\n')
fid.close()
	

