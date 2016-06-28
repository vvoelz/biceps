import os, sys, string
import numpy as np
import mdtraj as md

sys.path.append('../../../src')
from RestraintFile_cs import *

# read in atom pair indices
Ind = np.loadtxt('cs_indice.txt')

# read in restraint indices and distances
"""
# restraint index	<r^-6>^-1/6  from exp (nm)
0	0.334490497
1	0.39990427
2	0.2795903
"""
restraint_data = np.loadtxt('chemical_shift.txt')

# Check to see if these files have the same number of lines
if Ind.shape[0] != restraint_data.shape[0]:
    print 'The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.'%(Ind.shape[0],restraint_data.shape[0])
    sys.exit(1)


pdbfile = '../Gens/Gens0.pdb' 

# load in the pdbfile template 
topology = md.load_pdb(pdbfile).topology


r = RestraintFile()


all_atom_indices = [atom.index for atom in topology.atoms]
all_atom_residues = [atom.residue for atom in topology.atoms]
all_atom_names = [atom.name for atom in topology.atoms]


for i in range(Ind.shape[0]):
    a1 = int(Ind[i])
    restraint_index = restraint_data[i,0]
    chemical_shift        = restraint_data[i,1]
    r.add_line(restraint_index, a1, topology, chemical_shift)

print r

r.write('trploop2a.chemicalshift')


"[atom.index for atom in topology.atoms if (atom.residue.is_water and (atom.name == 'O'))]"

"""
#restraint_index atom_index1 res1 atom_name1 atom_index2 res2 atom_name2 distance(A)
0            38       SER3     H            21       LEU2     HA              3.000
1            38       SER3     H            40       SER3     HA              3.000
2            38       SER3     H            43       SER3     HB2             4.000
3            49       GLU4     H            40       SER3     HA              3.000
4            49       GLU4     H            43       SER3     HB2             4.000
5            49       GLU4     H            42       SER3     HB3             4.000
6            49       GLU4     H            51       GLU4     HA              4.000
7            49       GLU4     H            64       GLY5     H               3.000
8            64       GLY5     H            43       SER3     HB2             4.000
9            64       GLY5     H            51       GLU4     HA              4.000
10           64       GLY5     H            53       GLU4     HB3             3.000
11           64       GLY5     H            67       GLY5     HA2             3.000
"""



