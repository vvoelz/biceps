import sys, os
import mdtraj as md
from J_coupling import *


a=md.load('ligand1/traj76.xtc',top='ligand1/8690.gro')
d=compute_J3_HN_HA(a,model='Pardi')
print d[1]
