import sys, os, glob

sys.path.append('../../src')

from Structure import *
from PosteriorSampler import *

import numpy as np
import yaml

# Load in results from the exp-only sampling
exp_files = glob.glob('results/traj_exp_1e7steps_trial*.yaml')
t = [yaml.load( file(filename, 'r') ) for filename in exp_files]

# Load in results from the QM+exp sampling
QMexp_files = glob.glob('results/traj_QMexp_1e7steps_trial*.yaml')
t2 = [yaml.load( file(filename, 'r') ) for filename in QMexp_files]


# Calculate average free energies
if (1):
        f, f2 = [], []
        for k in range(len(t)):
            f.append([])
            f2.append([])
            # the data needs some massaging cause there might be nans or infs in there, which screws up plotting
            Ind = []  # state indices that are good
            for i in range(len(t[k]["state_f"])):
                g  = t[k]["state_f"][i]
                g2 = t2[k]["state_f"][i]
                if (~np.isnan(g) and ~np.isnan(g2) and (g != np.inf) and (g2 != np.inf)):
                    f[k].append( g )
                    f2[k].append( g2 )
                Ind.append(i)

        # reduced free energies
        f_std  = np.array(f).std(axis=0)
        f2_std = np.array(f2).std(axis=0)
        f  = np.array(f).mean(axis=0)
        f2 = np.array(f2).mean(axis=0)

        # populations
        p = np.exp(-f) 
        p = p/p.sum()

        p2 = np.exp(-f2) 
        p2 = p2/p2.sum()

        print p
        print p2

        # calculate group populations
        Ind_A = [38, 39, 65, 90]
        print 'Group A (exp):', p[Ind_A].sum()
        print 'Group A (QMexp):', p2[Ind_A].sum()

        Ind_B = [45, 59]
        Ind_C = [80, 85]
        Ind_D = [92]
        print 'Group B (exp):', p[Ind_B].sum()
        print 'Group B (QMexp):', p2[Ind_B].sum()

        print 'Group C (exp):', p[Ind_C].sum()
        print 'Group C (QMexp):', p2[Ind_C].sum()

        print 'Group D (exp):', p[Ind_D].sum()
        print 'Group D (QMexp):', p2[Ind_D].sum()



 
