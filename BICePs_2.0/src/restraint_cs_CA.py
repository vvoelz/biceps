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
class restraint_cs_CA(object):
    #Notes:# {{{
    '''

    '''
    # }}}
    def __init__(self):

        # Store chemical shift restraint info   #GYCA
        self.chemicalshift_CA_restraints = []
        self.nchemicalshift_CA = 0


    def load_data_cs_CA(self, filename, verbose=False):
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
            restraint_index, i, exp_chemicalshift_CA, model_chemicalshift_CA  = entry[0], entry[1], entry[4], entry[5]
            self.add_chemicalshift_CA_restraint(i, exp_chemicalshift_CA, model_chemicalshift_CA)

        # build groups of equivalency group indices, ambiguous group etc.

        self.compute_sse_chemicalshift_CA()



    def add_chemicalshift_CA_restraint(self, i, exp_chemicalshift_CA, model_chemicalshift_CA=None):
        """Add a chemicalshift NMR_Chemicalshift() object to the list."""
         # if the modeled distance is not specified, compute the distance from the conformation
#       Ind = self.conf.topology.select("index == j")
#       t=self.conf.atom_slice(Ind)
#        if model_chemicalshift_CA == None:
 #              r=md.nmr.chemicalshifts_shiftx2(r,pCA=2.5, temperature = 280.0)

 #              model_chemicalshift = r.mean(axis=1)
#                model_chemicalshift_CA = 1  # will be replaced by pre-computed cs

        self.chemicalshift_CA_restraints.append( NMR_Chemicalshift_CA(i, model_chemicalshift_CA, exp_chemicalshift_CA))

        self.nchemicalshift_CA += 1

    def compute_sse_chemicalshift_CA(self, debug=True):    #GYCA
        """Returns the (weighted) sum of squared errors for chemical shift values"""
#       for g in range(len(self.allowed_gamma)):

        sse_CA = 0.0
        N_CA = 0.0
        for i in range(self.nchemicalshift_CA):
		if debug:
               		print '---->', i, '%d'%self.chemicalshift_CA_restraints[i].i,
               		print '      exp', self.chemicalshift_CA_restraints[i].exp_chemicalshift_CA, 'model', self.chemicalshift_CA_restraints[i].model_chemicalshift_CA

                err_CA=self.chemicalshift_CA_restraints[i].model_chemicalshift_CA - self.chemicalshift_CA_restraints[i].exp_chemicalshift_CA
                sse_CA += (self.chemicalshift_CA_restraints[i].weight*err_CA**2.0)
                N_CA += self.chemicalshift_CA_restraints[i].weight
        self.sse_chemicalshift_CA = sse_CA
        self.Ndof_chemicalshift_CA = N_CA
        if debug:
            print 'self.sse_chemicalshift_CA', self.sse_chemicalshift_CA



class NMR_Chemicalshift_CA(object):        #GYCA
    """A class to store NMR chemical shift information."""

    # __init__:{{{
    def __init__(self, i, model_chemicalshift_CA, exp_chemicalshift_CA):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model_chemicalshift_CA = model_chemicalshift_CA

        # the experimental chemical shift
        self.exp_chemicalshift_CA = exp_chemicalshift_CA

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1




