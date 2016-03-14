import os, sys, glob, string
import numpy as np


# NOTES on BICePs restraint format:
#
# FORMAT
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
# 7             distance (in Angstroms)
#
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



def biceps_restraint_line(restraint_index, i, j, topology, distance):
    """Returns a formatted string for a line in BICePs restraint file.

    0             restraint_index
    1             atom index 1
    2             residue 1
    3             atom name 1
    4             atom index 2
    5             residue 2
    6             atom name 2
    7             distance (in Angstroms)
    """

    resname1  = [atom.residue for atom in topology.atoms if atom.index == i][0]
    atomname1 = [atom.name for atom in topology.atoms if atom.index == i][0]

    resname2  = [atom.residue for atom in topology.atoms if atom.index == j][0]
    atomname2 = [atom.name for atom in topology.atoms if atom.index == j][0]

    #resname1, atomname1 = topology.atoms[i].residue, topology.atoms[i].name
    #resname2, atomname2 = topology.atoms[j].residue, topology.atoms[j].name

    return '%-8d     %-8d %-8s %-8s     %-8d %-8s %-8s     %8.4f'%(restraint_index, i, resname1, atomname1, j, resname2, atomname2, distance) 


def biceps_restraint_line_header():
    """Returns a header string the the biceps restraint file."""

    return "#" + string.joinfields(['restraint_index', 'atom_index1', 'res1', 'atom_name1', 'atom_index2', 'res2', 'atom_name2', 'distance(A)'], ' ')


class RestraintFile(object):
    """A class containing input/output methods for writing BICePs Restaint Files."""

    def __init__(self, filename=None):
        """Initialize the RestraintFile class."""
      
        self.header = biceps_restraint_line_header()
        self.comments = []
        self.lines  = []

        if filename != None:
            self.read(filename)
      
  
    def read(self, filename):
        """Read a BICePs restraint file."""

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
        """Write stored BICePs restraint information to file."""

        fout = open(filename, 'w')
        fout.write(self.header+'\n')
        for line in self.comments:
            fout.write(line+'\n')
        for line in self.lines:
            fout.write(line+'\n')
        fout.close()

        print 'Wrote', filename


    def add_line(self, restraint_index, i, j, topology, distance):
        """Add a line to the BICePs file."""

        self.lines.append(biceps_restraint_line(restraint_index, i, j, topology, distance))

    def parse_line(line):
        """Parse a BICePs data line and return the values

        RETURNS
        restraint_index, atom_index1, res1, atom_name1, atom_index2, res2, atom_name2, distance(A) 
        """

        fields = line.strip().split()
        if len(fields) != 8:
            raise Exception, "Incorrect number of fields in parsed BICePs line!"

        restraint_index = int(fields[0])
        atom_index1     = int(fields[1])
        res1            = fields[2]
        atom_name1      = fields[3]
        atom_index2     = int(fields[4])
        res2            = fields[5]
        atom_name2      = fields[6]
        distance        = float(fields[7])

        return restraint_index, atom_index1, res1, atom_name1, atom_index2, res2, atom_name2, distance
