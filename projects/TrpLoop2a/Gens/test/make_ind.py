import os, sys
import numpy as np
"""
This script works for indices generation by residue number and atom name input. The result will be a numpy array which is 0-based index that could be directly used in MdTraj.  -Yunhui Ge
"""

filename='Gens1.new.pdb'
#filename=raw_input("Please tell me your pdb filename:  ")
#resid=raw_input("Residue number (1-based index): ")
#atom_name=raw_input("Atom name (capital form): ")

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
#print field[0][5]
ind=[]
for i in range(len(field)):
	resid=raw_input("Residue number (1-based index): ")
	atom_name=raw_input("Atom name (capital form): ")
	if resid != "none":
		for i in range(len(field)):
			if (field[i][5] == resid) and (field[i][2]== atom_name):
				ind.append(int(field[i][1])-1)
	else:
		print ind		
		break

atompairs=raw_input("How many atom pairs of your input?  ")
if len(ind) % 2 != 0:
	print "Something is wrong about your input, please double check!"

elif (len(ind)/2) != int(atompairs):
	print "Something is wrong about your input, please double check"

else:	
	print str(len(ind)/2) + " atom pairs for your input file and they are (0-based index): "
	IND=zip(ind[::2],ind[1::2])
	print IND 
	np.save('Ind.npy',IND)
sys.exit()



print line
