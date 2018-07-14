import sys, os
from toolbox import *

a=get_J3_HN_HA('ligand1/traj76.xtc','ligand1/8690.gro',frame=[0,1],outname='test')
#a=get_J3_HN_HA('ligand1/traj76.xtc','ligand1/8690.gro',outname='test')
print a
