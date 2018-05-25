#!/usr/bin/env python

# Import Modules:{{{
import os, sys, glob
import numpy as np
#from msmbuilder import Conformation
import mdtraj
# Can we get rid of yaml and substitute for another multicolumn layout?
# Ideas:{{{

# }}}

from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from RestraintFile_cs import *    # Class - creates Chemical shift restraint file

# }}}

# Class Restraint:{{{
class restraint_cs_N(object):
    #Notes:# {{{
    '''

    '''
    # }}}
    def __init__(self,dlogsigma_cs_N=np.log(1.02),
            sigma_cs_N_min=0.05,sigma_cs_N_max=20.0):

        # Store chemical shift restraint info   #GYN
        self.chemicalshift_N_restraints = []
        self.nchemicalshift_N = 0

        # pick initial values for sigma_cs_N (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_cs_N = dlogsigma_cs_N  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_cs_N_min = sigma_cs_N_min
        self.sigma_cs_N_max = sigma_cs_N_max
        self.allowed_sigma_cs_N = np.exp(np.arange(np.log(self.sigma_cs_N_min), np.log(self.sigma_cs_N_max), self.dlogsigma_cs_N))
#        print 'self.allowed_sigma_cs_N', self.allowed_sigma_cs_N
#        print 'len(self.allowed_sigma_cs_N) =', len(self.allowed_sigma_cs_N)
        self.sigma_cs_N_index = len(self.allowed_sigma_cs_N)/2   # pick an intermediate value to start with
        self.sigma_cs_N = self.allowed_sigma_cs_N[self.sigma_cs_N_index]



    def load_data_cs_N(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .chemicalshift file format.
        """

        # Read in the lines of the chemicalshift data file
        b = RestraintFile_cs(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line_cs(line) )  # [restraint_index, atom_index1, res1, atom_name1, chemicalshift]

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry

        ### distances ###

        # the equivalency indices for distances are in the first column of the *.biceps f
#       equivalency_indices = [entry[0] for entry in data]

        # add the chemical shift restraints
        for entry in data:
            restraint_index, i, exp_chemicalshift_N, model_chemicalshift_N  = entry[0], entry[1], entry[4], entry[5]
            self.add_chemicalshift_N_restraint(i, exp_chemicalshift_N, model_chemicalshift_N)

        # build groups of equivalency group indices, ambiguous group etc.

        self.compute_sse_chemicalshift_N()



    def add_chemicalshift_N_restraint(self, i, exp_chemicalshift_N, model_chemicalshift_N=None):
        """Add a chemicalshift NMR_Chemicalshift() object to the list."""
         # if the modeled distance is not specified, compute the distance from the conformation
#       Ind = self.conf.topology.select("index == j")
#       t=self.conf.atom_slice(Ind)
#        if model_chemicalshift_N == None:
 #              r=md.nmr.chemicalshifts_shiftx2(r,pN=2.5, temperature = 280.0)

 #              model_chemicalshift = r.mean(axis=1)
#                model_chemicalshift_N = 1  # will be replaced by pre-computed cs

        self.chemicalshift_N_restraints.append( NMR_Chemicalshift_N(i, model_chemicalshift_N, exp_chemicalshift_N))

        self.nchemicalshift_N += 1

    def compute_sse_chemicalshift_N(self, debug=True):    #GYN
        """Returns the (weighted) sum of squared errors for chemical shift values"""
#       for g in range(len(self.allowed_gamma)):

        sse_N = 0.0
        N_N = 0.0
        for i in range(self.nchemicalshift_N):
		if debug:
               		print '---->', i, '%d'%self.chemicalshift_N_restraints[i].i,
               		print '      exp', self.chemicalshift_N_restraints[i].exp_chemicalshift_N, 'model', self.chemicalshift_N_restraints[i].model_chemicalshift_N

                err_N=self.chemicalshift_N_restraints[i].model_chemicalshift_N - self.chemicalshift_N_restraints[i].exp_chemicalshift_N
                sse_N += (self.chemicalshift_N_restraints[i].weight*err_N**2.0)
                N_N += self.chemicalshift_N_restraints[i].weight
        self.sse_chemicalshift_N = sse_N
        self.Ndof_chemicalshift_N = N_N
        if debug:
            print 'self.sse_chemicalshift_N', self.sse_chemicalshift_N



class NMR_Chemicalshift_N(object):        #GYN
    """A class to store NMR chemical shift information."""

    # __init__:{{{
    def __init__(self, i, model_chemicalshift_N, exp_chemicalshift_N):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model_chemicalshift_N = model_chemicalshift_N

        # the experimental chemical shift
        self.exp_chemicalshift_N = exp_chemicalshift_N

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1




