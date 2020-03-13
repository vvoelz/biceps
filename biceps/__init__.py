#!/usr/bin/env python

__version__= "2.0"
HEADER="""BICePs - Bayesian Inference of Conformational Populations, Version %s"""%(__version__)
print(HEADER)
name = "biceps"

import biceps.Observable
from biceps.Restraint import Ensemble
#import biceps.Restraint
from biceps.PosteriorSampler import PosteriorSampler
from biceps.Analysis import Analysis
from biceps.convergence import Convergence
import biceps.toolbox

import biceps.J_coupling
import biceps.KarplusRelation



