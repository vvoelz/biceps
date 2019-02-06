import os, sys
import numpy as np

filename='clean.pdb'
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
#sys.exit()
ind=[]
for i in range(3*len(field)):
#	resid=raw_input("Residue number (1-based index): ")
	atom_name=raw_input("Atom name (capital form): ")
	if atom_name != "none":
		for i in range(len(field)):
			if (field[i][2]== atom_name):
				ind.append(int(field[i][1])-1)
	else:
		print ind		
		break
IND=zip(ind[::2],ind[1::2])
print IND 
np.save('Ind_noe.npy',IND)
sys.exit()



print line
