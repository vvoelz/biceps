##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# This file is used to initialize variables for chemical shifts (CA) in BICePs and prepare fuctions for compute necessary quantities for posterior sampling.
##############################################################################


##############################################################################
# Imports
##############################################################################


import os, sys, glob
import numpy as np
#from msmbuilder import Conformation
import mdtraj

from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from prep_cs import *    # Class - creates Chemical shift restraint file

##############################################################################
# Code
##############################################################################


class restraint_cs_Ca(object):
    def __init__(self):

        # Store chemical shift restraint info   
        self.cs_Ca_restraints = []
        self.ncs_Ca = 0
        self.sse_cs_Ca = 0 
        self.Ndof_cs_Ca = None 


    def load_data_cs_Ca(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .cs file format.
        """

        # Read in the lines of the cs data file
        b = prep_cs(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line(line) )  # [restraint_index, atom_index1, res1, atom_name1, cs]

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry


        # add the chemical shift restraints
        for entry in data:
            restraint_index, i, exp_cs_Ca, model_cs_Ca  = entry[0], entry[1], entry[4], entry[5]
            self.add_cs_Ca_restraint(i, exp_cs_Ca, model_cs_Ca)


        self.compute_sse_cs_Ca()



    def add_cs_Ca_restraint(self, i, exp_cs_Ca, model_cs_Ca=None):
        """Add a cs NMR_Chemicalshift() object to the list."""

        self.cs_Ca_restraints.append( NMR_Chemicalshift_Ca(i, model_cs_Ca, exp_cs_Ca))

        self.ncs_Ca += 1

    def compute_sse_cs_Ca(self, debug=False):    #GYCa
        """Returns the (weighted) sum of squared errors for chemical shift values"""

        sse_Ca = 0.0
        N_Ca = 0.0
        for i in range(self.ncs_Ca):
		if debug:
               		print '---->', i, '%d'%self.cs_Ca_restraints[i].i,
               		print '      exp', self.cs_Ca_restraints[i].exp_cs_Ca, 'model', self.cs_Ca_restraints[i].model_cs_Ca

                err_Ca=self.cs_Ca_restraints[i].model_cs_Ca - self.cs_Ca_restraints[i].exp_cs_Ca
                sse_Ca += (self.cs_Ca_restraints[i].weight*err_Ca**2.0)
                N_Ca += self.cs_Ca_restraints[i].weight
        self.sse_cs_Ca = sse_Ca
        self.Ndof_cs_Ca = N_Ca
        if debug:
            print 'self.sse_cs_Ca', self.sse_cs_Ca



class NMR_Chemicalshift_Ca(object):        #GYH
    """A class to store NMR chemical shift information."""

    # __init__:{{{
    def __init__(self, i, model_cs_Ca, exp_cs_Ca):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model_cs_Ca = model_cs_Ca

        # the experimental chemical shift
        self.exp_cs_Ca = exp_cs_Ca

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1




