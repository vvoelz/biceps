##############################################################################
# Authors: Yunhui Ge, Vincent Voelz, Rob Raddi
# This file is used to initialize variables for chemical shifts (HA) in BICePs and prepare fuctions for compute necessary quantities for posterior sampling.
##############################################################################

##############################################################################
# Imports
##############################################################################
import os, sys, glob
import numpy as np
from prep_cs import *   # Class - creates Chemical shift restraint file
from Restraint import *

##############################################################################
# Code
##############################################################################

class Restraint_cs_Ha(Restraint):
    """ """

    def load_data(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .cs file format."""

        #self.n = 0
        # Read in the lines of the cs data file
        read = prep_cs(filename=filename)
        if verbose:
                print read.lines
        data = []
        for line in read.lines:
                data.append( read.parse_line(line) )  # [restraint_index, atom_index1, res1, atom_name1, cs]
        # add the chemical shift restraints
        if verbose:
            print 'Loaded from', filename, ':'
        for entry in data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            self.add_restraint(NMR_Chemicalshift(i, exp, model))
            if verbose:
                print entry
        self.compute_sse()
        #self.n += 1


