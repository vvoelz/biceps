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
class restraint_cs_Ca(object):
    #Notes:# {{{
    '''

    '''
    # }}}
    def __init__(self,dlogsigma_cs_Ca=np.log(1.02),
            sigma_cs_Ca_min=0.05,sigma_cs_Ca_max=20.0):

        # Store chemical shift restraint info   #GYCa
        self.chemicalshift_Ca_restraints = []
        self.nchemicalshift_Ca = 0

        # pick initial values for sigma_cs_Ca (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_cs_Ca = dlogsigma_cs_Ca  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_cs_Ca_min = sigma_cs_Ca_min
        self.sigma_cs_Ca_max = sigma_cs_Ca_max
        self.allowed_sigma_cs_Ca = np.exp(np.arange(np.log(self.sigma_cs_Ca_min), np.log(self.sigma_cs_Ca_max), self.dlogsigma_cs_Ca))
#        print 'self.allowed_sigma_cs_Ca', self.allowed_sigma_cs_Ca
#        print 'len(self.allowed_sigma_cs_Ca) =', len(self.allowed_sigma_cs_Ca)
        self.sigma_cs_Ca_index = len(self.allowed_sigma_cs_Ca)/2   # pick an intermediate value to start with
        self.sigma_cs_Ca = self.allowed_sigma_cs_Ca[self.sigma_cs_Ca_index]





    def load_data_cs_Ca(self, filename, verbose=False):
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
            restraint_index, i, exp_chemicalshift_Ca, model_chemicalshift_Ca  = entry[0], entry[1], entry[4], entry[5]
            self.add_chemicalshift_Ca_restraint(i, exp_chemicalshift_Ca, model_chemicalshift_Ca)

        # build groups of equivalency group indices, ambiguous group etc.

        self.compute_sse_chemicalshift_Ca()



    def add_chemicalshift_Ca_restraint(self, i, exp_chemicalshift_Ca, model_chemicalshift_Ca=None):
        """Add a chemicalshift NMR_Chemicalshift() object to the list."""
         # if the modeled distance is not specified, compute the distance from the conformation
#       Ind = self.conf.topology.select("index == j")
#       t=self.conf.atom_slice(Ind)
#        if model_chemicalshift_Ca == None:
 #              r=md.nmr.chemicalshifts_shiftx2(r,pCa=2.5, temperature = 280.0)

 #              model_chemicalshift = r.mean(axis=1)
#                model_chemicalshift_Ca = 1  # will be replaced by pre-computed cs

        self.chemicalshift_Ca_restraints.append( NMR_Chemicalshift_Ca(i, model_chemicalshift_Ca, exp_chemicalshift_Ca))

        self.nchemicalshift_Ca += 1

    def compute_sse_chemicalshift_Ca(self, debug=True):    #GYCa
        """Returns the (weighted) sum of squared errors for chemical shift values"""
#       for g in range(len(self.allowed_gamma)):

        sse_Ca = 0.0
        N_Ca = 0.0
        for i in range(self.nchemicalshift_Ca):
		if debug:
               		print '---->', i, '%d'%self.chemicalshift_Ca_restraints[i].i,
               		print '      exp', self.chemicalshift_Ca_restraints[i].exp_chemicalshift_Ca, 'model', self.chemicalshift_Ca_restraints[i].model_chemicalshift_Ca

                err_Ca=self.chemicalshift_Ca_restraints[i].model_chemicalshift_Ca - self.chemicalshift_Ca_restraints[i].exp_chemicalshift_Ca
                sse_Ca += (self.chemicalshift_Ca_restraints[i].weight*err_Ca**2.0)
                N_Ca += self.chemicalshift_Ca_restraints[i].weight
        self.sse_chemicalshift_Ca = sse_Ca
        self.Ndof_chemicalshift_Ca = N_Ca
        if debug:
            print 'self.sse_chemicalshift_Ca', self.sse_chemicalshift_Ca



class NMR_Chemicalshift_Ca(object):        #GYCa
    """A class to store NMR chemical shift information."""

    # __init__:{{{
    def __init__(self, i, model_chemicalshift_Ca, exp_chemicalshift_Ca):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model_chemicalshift_Ca = model_chemicalshift_Ca

        # the experimental chemical shift
        self.exp_chemicalshift_Ca = exp_chemicalshift_Ca

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1




