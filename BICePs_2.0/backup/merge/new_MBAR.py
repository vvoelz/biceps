import sys, os, glob
sys.path.append('src') # source code path --Yunhui 04/2018
from Restraint import *
from PosteriorSampler import *
import cPickle  # to read/write serialized sampler classes
import argparse
import re
from Analysis import *



dataFiles = 'test_cs_H'
A = Analysis(50,dataFiles,'results_ref_normal')
A.MBAR_analysis()
