##############################################################################
# Authors: Vincent Voelz
# Contributors: Yunhui Ge,  Rob Raddi
# This file is used to prepare input files of J coupling constants in BICePs.
##############################################################################


##############################################################################
# Imports
##############################################################################
import os, sys, glob, string
import numpy as np


# NOTES on BICePs restraint format:
#
# FORMAT (NOE)
# column        description
#
# 0             restraint index
#
# 1             atom index 1
# 2             residue 1
# 3             atom name 1
#
# 4             atom index 2
# 5             residue 2
# 6             atom name 2
#
# 7             noe (in Angstroms)
#


# 
# EQUIVALENT PROTONS
# Multiple restraints can share the same restraint index -- this means they are equivalent protons
# 
# AMBIGUOUS ASSIGNMENTS
# There may be two (or more) sets of protons assigned different noes, but we don't know which is which.
# BICePs has limited capabilities to deal with this situation, with the ambiguous restraint info read in separately.
#
# UPPER and LOWER BOUNDS
# BICePs restraints do not have upper/lower bounds, only a mean noe value.  Any values specified in  
# XPLOR/CNS files are ignored.

##############################################################################
# Code
##############################################################################

def biceps_restraint_line_noe(restraint_index, i, j, topology, exp_noe, model_noe):
    """Returns a formatted string for a line in NOE restraint file.

    0             restraint_index
    1             atom index 1
    2             residue 1
    3             atom name 1
    4             atom index 2
    5             residue 2
    6             atom name 2
    7             exp_noe (in Angstroms)
    8		  model_noe (in Angstroms)
    """

    resname1  = [atom.residue for atom in topology.atoms if atom.index == i][0]
    atomname1 = [atom.name for atom in topology.atoms if atom.index == i][0]

    resname2  = [atom.residue for atom in topology.atoms if atom.index == j][0]
    atomname2 = [atom.name for atom in topology.atoms if atom.index == j][0]

    #resname1, atomname1 = topology.atoms[i].residue, topology.atoms[i].name
    #resname2, atomname2 = topology.atoms[j].residue, topology.atoms[j].name

    return '%-8d     %-8d %-8s %-8s     %-8d %-8s %-8s     %8.4f    %8.4f'%(restraint_index, i, resname1, atomname1, j, resname2, atomname2, exp_noe, model_noe) 


def biceps_restraint_line_noe_header():
    """Returns a header string the the NOE restraint file."""

    return "#" + string.joinfields(['restraint_index', 'atom_index1', 'res1', 'atom_name1', 'atom_index2', 'res2', 'atom_name2', 'exp_noe(A)', 'model_noe(A)'], ' ')


class prep_noe(object):
    """A class containing input/output methods for writing NOE Restaint Files."""

    def __init__(self, filename=None):
        """Initialize the RestraintFile_noe class."""
      
        self.header = biceps_restraint_line_noe_header()
        self.comments = []
        self.lines  = []

        if filename != None:
            self.read(filename)
      
  
    def read(self, filename):
        """Read a NOE restraint file."""

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
        """Write stored NOE restraint information to file."""

        fout = open(filename, 'w')
        fout.write(self.header+'\n')
        for line in self.comments:
            fout.write(line+'\n')
        for line in self.lines:
            fout.write(line+'\n')
        fout.close()

        print 'Wrote', filename


    def add_line(self, restraint_index, i, j, topology, exp_noe, model_noe):
        """Add a line to the NOE file."""

        self.lines.append(biceps_restraint_line_noe(restraint_index, i, j, topology, exp_noe, model_noe))

    def parse_line(self, line):
        """Parse a NOE data line and return the values

        RETURNS
        restraint_index, atom_index1, res1, atom_name1, atom_index2, res2, atom_name2, exp_noe(A), model_noe(A) 
        """

        fields = line.strip().split()
        if len(fields) != 9:
            raise Exception, "Incorrect number of fields in parsed noe line!"

        restraint_index = int(fields[0])
        atom_index1     = int(fields[1])
        res1            = fields[2]
        atom_name1      = fields[3]
        atom_index2     = int(fields[4])
        res2            = fields[5]
        atom_name2      = fields[6]
        exp_noe        = float(fields[7])
	model_noe	= float(fields[8])

        return restraint_index, atom_index1, res1, atom_name1, atom_index2, res2, atom_name2, exp_noe, model_noe

