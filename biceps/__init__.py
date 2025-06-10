__version__= "3.0"

greeting=f"""Bayesian Inference of Conformational Populations, Version {__version__}"""
#print(greeting)
name = "biceps"

from biceps.Restraint import Preparation
from biceps.Restraint import Ensemble
from biceps.Restraint import ExpandedEnsemble
from biceps.Restraint import get_restraint_options
from biceps.PosteriorSampler import PosteriorSampler
from biceps.Analysis import Analysis
from biceps.convergence import Convergence
import biceps.toolbox
from biceps.decorators import multiprocess
import multiprocessing as mp
from multiprocessing import Manager
import biceps.J_coupling
import biceps.KarplusRelation
from biceps.rdc import RDC_predictor
from biceps.parse_star import star_to_df
import biceps.XiOpt


#from biceps.PosteriorSampler import get_scalar_couplings_with_derivatives
#from biceps.PosteriorSampler import SimpleNeuralNetwork

