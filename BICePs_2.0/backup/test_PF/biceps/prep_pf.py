##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# This file is used to prepare input files of J coupling constants in BICePs.
##############################################################################


##############################################################################
# Imports
##############################################################################

import os, sys, glob, string
import numpy as np
# FORMAT (Chemical shift)
# column        description
#
# 0             restraint index
# 1		atom index 1
# 2             residue 1
#
# 3             protection factor

# 
# EQUIVALENT PROTONS
# Multiple restraints can share the same restraint index -- this means they are equivalent protons
# 
# AMBIGUOUS ASSIGNMENTS
# There may be two (or more) sets of protons assigned different distances, but we don't know which is which.
# BICePs has limited capabilities to deal with this situation, with the ambiguous restraint info read in separately.
#
# UPPER and LOWER BOUNDS
# BICePs restraints do not have upper/lower bounds, only a mean distance value.  Any values specified in  
# XPLOR/CNS files are ignored.

##############################################################################
# Code
##############################################################################

def biceps_restraint_line_pf_precomputed(restraint_index, i, topology,exp_pf, protection_factor):
    """Returns a formatted string for a line in protectionfactor restraint file.

    0             restraint_index
    1             atom index 1
    2             residue 1
    3             exp_data
    4             protection factor
    """

    resname1  = [atom.residue for atom in topology.atoms if atom.index == i][0]
    atomname1 = [atom.name for atom in topology.atoms if atom.index == i][0]

    #resname1, atomname1 = topology.atoms[i].residue, topology.atoms[i].name
    #resname2, atomname2 = topology.atoms[j].residue, topology.atoms[j].name

    return '%-8d     %-8d %-8s     %8.4f    %8.4f'%(restraint_index, i, resname1, exp_pf,protection_factor)


def biceps_restraint_line_pf_precomputed_header():
    """Returns a header string the the protectionfactor restraint file."""

    return "#" + string.joinfields(['restraint_index', 'atom_index1', 'res1', 'exp_pf','protection_factor'], ' ')




def biceps_restraint_line_pf(restraint_index, i, exp_pf, topology):
    """Returns a formatted string for a line in protectionfactor restraint file.

    0             restraint_index
    1             atom index 1
    2             residue 1
    3             exp_pf
    """

    resname1  = [atom.residue for atom in topology.atoms if atom.index == i][0]
    atomname1 = [atom.name for atom in topology.atoms if atom.index == i][0]

    #resname1, atomname1 = topology.atoms[i].residue, topology.atoms[i].name
    #resname2, atomname2 = topology.atoms[j].residue, topology.atoms[j].name

    return '%-8d     %-8d %-8s    %8.4f'%(restraint_index, i, resname1, exp_pf)


def biceps_restraint_line_pf_header():
    """Returns a header string the the protectionfactor restraint file."""

    return "#" + string.joinfields(['restraint_index', 'atom_index1', 'res1','exp_pf'], ' ')


class prep_pf(object):
    """A class containing input/output methods for writing protectionfactor Restaint Files."""

    def __init__(self, filename=None, precomputed_pf=False):
        """Initialize the RestraintFile_cs class."""

        self.header = biceps_restraint_line_pf_header()
        self.comments = []
        self.lines  = []

        if filename != None:
            self.read(filename)


    def read(self, filename):
        """Read a protectionfactor restraint file."""

        # read in the lines of from the input file
        fin = open(filename, 'r')
        lines = fin.readlines()
        fin.close()

        # parse the header
        if lines[0][0] == '#':
            self.header = lines.pop(0).strip()

        # store other '#' lines as comments
        while lines[0][0] == '#':
            self.comments.append( lines.pop(0).strip() )

        # read the other lines 
        while len(lines) > 0:
            self.lines.append( lines.pop(0).strip() )


    def write(self, filename, verbose=True):
        """Write stored protectionfactor restraint information to file."""

        fout = open(filename, 'w')
        fout.write(self.header+'\n')
        for line in self.comments:
            fout.write(line+'\n')
        for line in self.lines:
            fout.write(line+'\n')
        fout.close()

        print 'Wrote', filename


    def add_line(self, restraint_index, i,  topology, exp_pf, protection_factor=None):
        """Add a line to the protection_factor file."""
        if precomputed_pf:
            self.lines.append(biceps_restraint_line_pf_precomputed(restraint_index, i,  topology, exp_pf,protection_factor))
        else:
            self.lines.append(biceps_restraint_line_pf(restraint_index, i,  topology, exp_pf))

    def parse_line(self, line):
        """Parse a protectionfactor data line and return the values

        RETURNS
        restraint_index, atom_index1, res1, protectionfactor 
        """

        fields = line.strip().split()
        if precomputed_pf:
            if len(fields) != 4:
                raise Exception, "Incorrect number of fields in parsed protectionfactor line!"

            restraint_index = int(fields[0])
            atom_index1     = int(fields[1])
            res1            = fields[2]
            exp_pf          = float(fields[3])
            protection_factor      = float(fields[4])
            return restraint_index, atom_index1, res1, exp_pf, protection_factor

        else:
            if len(fields) != 3:
                raise Exception, "Incorrect number of fields in parsed protectionfactor line!"

            restraint_index = int(fields[0])
            atom_index1     = int(fields[1])
            res1            = fields[2]
            exp_pf          = float(fields[3])
            return restraint_index, atom_index1, res1, exp_pf

