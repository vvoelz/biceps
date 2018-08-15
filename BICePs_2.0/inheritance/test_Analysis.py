import sys, os, glob
from numpy import *
sys.path.append('src')
from Preparation import *
from PosteriorSampler_test import *
from Analysis_new import *
from Restraint import *



dataFiles = 'test_cs_mixed'
outdir = 'results_ref_normal_cs_mixed'
A = Analysis(50,dataFiles,outdir)
A.plot()
