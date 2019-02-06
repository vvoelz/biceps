import sys, os, glob
from numpy import *
sys.path.append('new_src')
from Preparation import *
from PosteriorSampler import *
from Analysis_new import *
from Restraint import *


dataFiles = 'noe_dis'
outdir = 'results_ref_normal_-0.5_exp'
A = Analysis(15037,dataFiles,outdir)
A.plot()
