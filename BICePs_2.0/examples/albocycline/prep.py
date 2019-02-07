import sys, os, glob
from numpy import *
sys.path.append('new_src')
from Preparation import *
from PosteriorSampler import *
from Analysis_new import *
from Restraint import *

#########################################
# Lets' create input files for BICePs
############ Preparation ################
# Specify necessary argument values
path='J_coupling/*txt'
states=100
indices='atom_indice_J.txt'
exp_data='exp_Jcoupling.txt'
top='pdbs_guangfeng/0.pdb'
data_dir=path
#dataFiles = 'test_cs_H'
dataFiles = 'noe_J'
out_dir=dataFiles

p=Preparation('J',states=states,indices=indices,exp_data=exp_data,top=top,data_dir=data_dir)
p.write(out_dir=out_dir)

