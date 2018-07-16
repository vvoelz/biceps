##############################################################################
# Authors: Yunhui Ge, Vincent Voelz, Rob Raddi
# This file is used to initialize variables for protection factors in BICePs and prepare fuctions for compute necessary quantities for posterior sampling.
##############################################################################

##############################################################################
# Imports
##############################################################################
import os, sys, glob
import numpy as np
import mdtraj
from prep_noe import *   # Class - creates NOE (Nuclear Overhauser effect) restraint file
from Restraint import *

##############################################################################
# Code
##############################################################################

class Restraint_noe(Restraint):
    """A derived class of Restraint() for noe distance restraints."""

    def load_data_noe(self, filename, verbose=False):
        """Load in the experimental NOE noe restraints from a .noe file format."""

        # Read in the lines of the biceps data file
        b = prep_noe(filename=filename)
        data = []
        for line in b.lines:
                data.append( b.parse_line(line) )  # [restraint_index, atom_index1, res1, atom_name1, atom_index2, res2, atom_name2, noe]
        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry
        # the equivalency indices for noe are in the first column of the *.biceps file
        equivalency_indices = [entry[0] for entry in data]
        if verbose:
            print 'noe equivalency_indices', equivalency_indices
        # add the noe restraints
        for entry in data:
            restraint_index, i, j, exp, model = entry[0], entry[1], entry[4], entry[7], entry[8]
            ri = self.conf.xyz[0,i,:]
            rj = self.conf.xyz[0,j,:]
            dr = rj-ri
            model = np.dot(dr,dr)**0.5
            self.add_restraint(NMR_Distance(i, j, model, exp,
                equivalency_index=equivalency_index))
        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups()
        self.n += 1


    def build_groups(self, verbose=False):
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
        self.compute_sse()


    def adjust_weights(self):
        """Adjust the weights of noe and dihedral restraints based on their equivalency group."""

        for group in self.noe_equivalency_groups.values():
            n = float(len(group))
            for i in group:
                self.noe_restraints[i].weight = 1.0/n


    def compute_sse(self, debug=False):
        """Returns the (weighted) sum of squared errors for noe,
        and the *effective* number of noe (i.e. the sum of the weights)"""
#	print "len(self.allowed_gamma)",len(self.allowed_gamma)
	self.sse = np.array([0.0 for gamma in self.allowed_gamma])
#        print "len(self.sse)",len(self.sse)
	if debug:
	    print 'self.allowed_gamma', self.allowed_gamma
        for g in range(len(self.allowed_gamma)):

            sse = 0.
            N = 0.
            for i in range(self.nnoe):

                gamma = self.allowed_gamma[g]
#                if g == 0:
#                    print '---->', i, '(%d,%d)'%(self.noe_restraints[i].i, self.noe_restraints[i].j),
#                    print '      exp',  self.noe_restraints[i].exp, 'model', self.noe_restraints[i].model
                if self.use_log_normal:
                    err = np.log(self.noe_restraints[i].model/(gamma*self.noe_restraints[i].exp))
                    #print 'log-normal err', err
                else:
                    err = gamma*self.noe_restraints[i].exp - self.noe_restraints[i].model
                    #print 'err', err
                sse += (self.noe_restraints[i].weight * err**2.0)
                N += self.noe_restraints[i].weight
            #print 'total sse =', sse
            self.sse[g] = sse
            self.Ndof = N
        if debug:
            print 'self.sse', self.sse



