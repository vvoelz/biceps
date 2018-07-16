##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# This file is used to initialize variables for protection factors in BICePs and prepare fuctions for compute necessary quantities for posterior sampling.
##############################################################################

##############################################################################
# Imports
##############################################################################
import os, sys, glob
import numpy as np
import mdtraj
from prep_pf import *    # Class - prepare functions for protection factors restraint file
from Restraint import *

##############################################################################
# Code
##############################################################################

class Restraint_pf(Restraint):
    """A derived class of Restraint() for protection factor restraints."""

    def load_data_pf(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .chemicalshift file format."""

        # Read in the lines of the protection factors data file
        b = prep_pf(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line_pf(line) )
        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry
        # add the protection factors restraints
        for entry in data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            self.add_restraint(NMR_Protectionfactor(i, model, exp))

        self.compute_sse()
        self.n += 1


    def compute_sse(self, debug=True):    #GYH
        """Returns the (weighted) sum of squared errors for protection factor values"""

        sse = 0.0
        N = 0.0
        for i in range(self.npf):
		if debug:
               		print '---->', i, '%d'%self.pf_restraints[i].i,
               		print '      exp', self.pf_restraints[i].exp, 'model', self.pf_restraints[i].model

                err=self.pf_restraints[i].model - self.pf_restraints[i].exp
                sse += (self.pf_restraints[i].weight*err**2.0)
                N += self.pf_restraints[i].weight
        self.sse = sse
        self.Ndof = N
        if debug:
            print 'self.sse', self.sse



