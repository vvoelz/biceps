#!/usr/bin/env python

__version__= 2.0
HEADER="""BICePs - Bayesian Inference of Conformational Populations, Version %s"""%(__version__)
print(HEADER)
name = "biceps"

#from Analysis import *
from J_coupling import *
from KarplusRelation import *
from Observable import *
from PosteriorSampler import *
from Preparation import *
from Restraint import *
from init_res import *
from prep_J import *
from prep_cs import *
from prep_noe import *
from prep_pf import *
from toolbox import *
#from Sampler import *
#from setup import *    # is this needed?
from Analysis import *
from convergence import *  # Should this be a submodule or just another class

