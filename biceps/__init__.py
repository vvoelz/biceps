#!/usr/bin/env python

__version__= "2.0"
HEADER="""BICePs - Bayesian Inference of Conformational Populations, Version %s"""%(__version__)
print(HEADER)
name = "biceps"

import biceps.J_coupling
import biceps.KarplusRelation
import biceps.Observable
from biceps.PosteriorSampler import PosteriorSampler
import biceps.Preparation
import biceps.Restraint
from biceps.init_res import *
import biceps.prep_J
import biceps.prep_cs
import biceps.prep_noe
import biceps.prep_pf
import biceps.toolbox
from biceps.Analysis import Analysis
from biceps.convergence import Convergence



