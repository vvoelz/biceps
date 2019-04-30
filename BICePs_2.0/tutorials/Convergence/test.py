import sys, os
sys.path.append('biceps/')
from toolbox import *

traj = 'traj_lambda1.00.npz'

rest_type = get_rest_type(traj)  # get restraint used in sampling
print "restraint_type", rest_type

sampled_parameters = get_sampled_parameters(traj,rest_type=rest_type)
print "sampled parameters", sampled_parameters
