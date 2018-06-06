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
#from msmbuilder import Conformation
import mdtraj

from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from prep_noe import *   # Class - creates NOE (Nuclear Overhauser effect) restraint file

##############################################################################
# Code
##############################################################################

# Class Restraint:{{{
class restraint_noe(object):
    def __init__(self,gamma_min=0.1,gamma_max=1,dloggamma=0.1, use_log_normal_noe=False):

        # Store noe restraint info
        self.noe_restraints = []
        self.noe_equivalency_groups = {}
        self.nnoe = 0
        self.dloggamma = np.log(1.01)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max	
	self.allowed_gamma = np.exp(np.arange(np.log(self.gamma_min), np.log(self.gamma_max), self.dloggamma))
	print "self.allowed_gamma", self.allowed_gamma
        #self.sse_noe = np.array([0.0 for gamma in self.allowed_gamma])
	#print "len(self.sse_noe)", len(self.sse_noe)
        self.Ndof_noe = 0.0
	self.use_log_normal_noe = use_log_normal_noe


    """A class for all restraints"""

    # Load Experimental Data (ALL Restraints):{{{
    def load_data_noe(self, filename, verbose=False):
        """Load in the experimental NOE noe restraints from a .noe file format.
        """

        # Read in the lines of the biceps data file
        b = prep_noe(filename=filename)
        data = []
        for line in b.lines:
                data.append( b.parse_line(line) )  # [restraint_index, atom_index1, res1, atom_name1, atom_index2, res2, atom_name2, noe]

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry

        ### noe ###

        # the equivalency indices for noe are in the first column of the *.biceps file
        equivalency_indices = [entry[0] for entry in data]
        if verbose:
            print 'noe equivalency_indices', equivalency_indices


        # add the noe restraints
        for entry in data:
            restraint_index, i, j, exp_noe, model_noe = entry[0], entry[1], entry[4], entry[7], entry[8]
            self.add_noe_restraint(i, j, exp_noe, model_noe, equivalency_index=restraint_index)

        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups_noe()




    def add_noe_restraint(self, i, j, exp_noe, model_noe=None,
                               equivalency_index=None):
        """Add an NOE NMR_Distance() object to the set"""

        # if the modeled noe is not specified, compute the noe from the conformation
        if model_noe == None:
            ri = self.conf.xyz[0,i,:]
            rj = self.conf.xyz[0,j,:]
            dr = rj-ri
            model_noe = np.dot(dr,dr)**0.5

        self.noe_restraints.append( NMR_Distance(i, j, model_noe, exp_noe,
                                                      equivalency_index=equivalency_index) )
        self.nnoe += 1


    def build_groups_noe(self, verbose=False):
        """Build equivalency and ambiguity groups for noe and dihedrals,
        and store pre-computed SSE and d.o.f for noe and dihedrals"""

        # compile noe_equivalency_groups from the list of NMR_Distance() objects
        for i in range(len(self.noe_restraints)):
            d = self.noe_restraints[i]
            if d.equivalency_index != None:
                if not self.noe_equivalency_groups.has_key(d.equivalency_index):
                    self.noe_equivalency_groups[d.equivalency_index] = []
                self.noe_equivalency_groups[d.equivalency_index].append(i)
        if verbose:
            print 'self.noe_equivalency_groups', self.noe_equivalency_groups


        # adjust the weights of noe and dihedrals to account for equivalencies
        self.adjust_weights()
        # precompute SSE and Ndof for noe
        self.compute_sse_noe()


    def adjust_weights(self):
        """Adjust the weights of noe and dihedral restraints based on their equivalency group."""

        for group in self.noe_equivalency_groups.values():
            n = float(len(group))
            for i in group:
                self.noe_restraints[i].weight = 1.0/n


    def compute_sse_noe(self, debug=False):
        """Returns the (weighted) sum of squared errors for noe,
        and the *effective* number of noe (i.e. the sum of the weights)"""
	print "len(self.allowed_gamma)",len(self.allowed_gamma)
	self.sse_noe = np.array([0.0 for gamma in self.allowed_gamma])
        print "len(self.sse_noe)",len(self.sse_noe)
	if debug:
	    print 'self.allowed_gamma', self.allowed_gamma
        for g in range(len(self.allowed_gamma)):

            sse = 0.
            N = 0.
            for i in range(self.nnoe):

                gamma = self.allowed_gamma[g]
#                if g == 0:
#                    print '---->', i, '(%d,%d)'%(self.noe_restraints[i].i, self.noe_restraints[i].j),
#                    print '      exp',  self.noe_restraints[i].exp_noe, 'model', self.noe_restraints[i].model_noe
                if self.use_log_normal_noe:
                    err = np.log(self.noe_restraints[i].model_noe/(gamma*self.noe_restraints[i].exp_noe))
                    #print 'log-normal err', err
                else:
                    err = gamma*self.noe_restraints[i].exp_noe - self.noe_restraints[i].model_noe
                    #print 'err', err
                sse += (self.noe_restraints[i].weight * err**2.0)
                N += self.noe_restraints[i].weight
            #print 'total sse =', sse
            self.sse_noe[g] = sse
            self.Ndof_noe = N
        if debug:
            print 'self.sse_noe', self.sse_noe

class NMR_Distance(object):
    """A class to store NMR noe information."""

    # __init__:{{{
    def __init__(self, i, j, model_noe, exp_noe, equivalency_index=None):

        # Atom indices from the Conformation() defining this noe
        self.i = i
        self.j = j

        # the model noe in this structure (in Angstroms)
        self.model_noe = model_noe

        # the experimental NOE noe (in Angstroms)
        self.exp_noe = exp_noe

        # the index of the equivalency group (i.e. a tag for equivalent H's)
        self.equivalency_index = equivalency_index

        # N equivalent noe should only get 1/N of the weight when computing chi^2
        self.weight = 1.0  # default is N=1




