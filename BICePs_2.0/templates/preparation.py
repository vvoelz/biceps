# import source code
import sys, os, glob
sys.path.append('biceps')
from Preparation import *
from toolbox import *

# REQUIRED: raw data of pre-comuted chemical shifts
#path = 'NOE/*txt'
path = 'PATH OF YOUR PRECOMPUTED OBSERVABLES'

# REQUIRED: number of states
#states = 100
states = YOUR NUMBER OF STATES

# REQUIRED: atom indices of pairwise distances
#indices = 'atom_indice_noe.txt'
indices = 'YOUR ATOM INDEX FILE'


# REQUIRED: experimental data
#exp_data = 'noe_distance.txt'
exp_data = 'YOUR EXP DATA FILE'

# REQUIRED: topology file (as it only supports topology information, so it doesn't matter which state is used)
#top = 'cineromycinB_pdbs/0.fixed.pdb'
top = 'YOUR *.PDB FILE'

# OPTIONAL: output directory of generated files
#out_dir = 'noe_J'
out_dir = 'YOUR OUTPUT DIRECTORY'

p = Preparation('YOUR EXPERIMENTAL OBSERVABLES',states=states,indices=indices,exp_data=exp_data,top=top,data_dir=path)   # 'noe' scheme is selected
p.write(out_dir=out_dir)
# This will convert pairwise distances files for each state to a BICePs readable format and saved the new files in "noe_J" folder.
