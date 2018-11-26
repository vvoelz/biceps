### In BICePs 2.0, this script will play as a center role of BICePs calucltion.
### The users should specify all input files, type of reference potential
### they want to use (if other than default). --Yunhui 05/2018###

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
#path='J_coupling/*txt'
states=100
#indices='atom_indice_J.txt'
#exp_data='cs_H/chemical_shift_NH.txt'
top='cineromycinB_pdbs/0.fixed.pdb'
#data_dir=path
#dataFiles = 'test_cs_H'
dataFiles = 'noe_J'
out_dir=dataFiles
outdir = 'results_ref_normal'
#p=Preparation('cs_H',states=states,indices=indices,exp_data=exp_data,top=top,data_dir=data_dir)
#p.write(out_dir=out_dir)

#p=Preparation('noe',states=15037,
#        exp_data='noe/noe.txt',
#        top='noe/Gens/Gens0.pdb',
#        data_dir='noe/micro_distances/dis_*.txt',
#        indices='noe/noe_indices.txt')
#p.write(out_dir=out_dir)



#########################################
# Let's do analysis using MBAR and plot figures
############ MBAR and Figures ###########
# Specify necessary argument values

A = Analysis(100,dataFiles,outdir)
A.plot()

