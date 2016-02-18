#!/usr/bin/env python
#
# This script is designed to help convert XPLOR/CNS distance restraints to BICePs restraint format of columns
#
# REQUIREMENTS
# mdtraj python library must be installed (http://mdtraj.org)
#
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


import os, sys, string
import mdtraj as md

usage = """Usage: python convert_xplor_to_biceps.py xplorfile pdbfile

    REQUIRED INPUTS
    xplorfile - a XPLOR/CNS distance restraint file
    pdbfile   - PDB containing the names and residues of atoms references

    OUTPUT 
    BICePs-style distance restraint file, printed to standard output

    Try:
    >>> python convert_xplor_to_biceps.py ApoMb_Lecomte/apomb_distances_reformat.tbl ApoMb_Lecomte/deposit_reformat.pdb
"""


if len(sys.argv) < 3:
    print usage
    sys.exit(1)

xplorfile = sys.argv[1]
pdbfile = sys.argv[2]


# load in the pdbfile template 
topology = md.load_pdb(pdbfile).topology



""" Example XPLOR format:
!Gly-5
assign (resid  5    and  name  hn   )(resid  3   and  name  hb2  ) 4.0 2.2 1.5
assign (resid  5    and  name  hn   )(resid  4   and  name  ha   ) 4.0 2.2 1.5
assign (resid  5    and  name  hn   )(resid  4   and  name  hb1  ) 3.0 1.2 0.3
assign (resid  5    and  name  hn   )(resid  5   and  name  ha#  ) 3.0 1.2 0.3
assign (resid  5    and  name  hn   )(resid  6   and  name  hn   ) 4.0 2.2 1.5
"""


def xplor2pdb_hydrogen_name(name):
    """Convert an xplor-type H atom name (with wildcards) to a PDB-style name (or list of names)."""

    if name == 'hn':
        return 'H' 

    name = string.upper(name)

    if name.count('*#'):
        return [name.replace('*#',s) for s in ['11','21','31','21','22','23','31','32','33']]
    elif name.count('#'):
        return [name.replace('#',s) for s in ['1','2','3']]
    elif name.count('*'):
        return [name.replace('*',s) for s in ['1','2','3']]

    else:
        return name

def indices_from_xplor_selection(sel, topology):
    """Given an xplor-style selection string, and an mdtraj Topology object, returns the atom indices
    referred to. WARNING: this is not robust; it's tuned to the selection style in apomb_distances_reformat.tbl
    """

    # find atom indices for selection 1
    sel = sel.replace('(', '').replace(')', '')  # remove parentheses
    prefix = sel.split('name')[0]   # Ex:  "resid  5    and " 
    ### if the prefix has a "resid" in it, replace it with the mdtraj-friendly resSeq 
    prefix = prefix.replace('resid ', 'resSeq ')

    name = xplor2pdb_hydrogen_name( sel.split('name')[1].strip() )
    Ind = []
    if type(name) == list:
        for s in name:
            selection_language = prefix + 'name %s'%s
            #print 'selection_language', selection_language
            Ind += list(topology.select(selection_language.strip()))
    else:
        selection_language = prefix + 'name %s'%name
        #print 'selection_language', selection_language
        Ind += list(topology.select(selection_language.strip()))

    if len(Ind) == 0:
        raise Exception, "Can't find atom indices for the given selection."

    return Ind


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

    return '%-8d     %-8d %-8s %-8s     %-8d %-8s %-8s     %8.3f'%(restraint_index, i, resname1, atomname1, j, resname2, atomname2, distance) 

def biceps_restraint_line_header():
    """Returns a header string the the biceps restraint file."""

    return "#" + string.joinfields(['restraint_index', 'atom_index1', 'res1', 'atom_name1', 'atom_index2', 'res2', 'atom_name2', 'distance(A)'], ' ')


### PARSE the XPLOR distance restraint file

# load in the lines of the file
fin = open(xplorfile, 'r')
lines = fin.readlines()

# strip out all comments
lines = [line for line in lines if line[0] != '!']  

restraint_index = 0
print biceps_restraint_line_header() 

# find atom selections
for line in lines:

    # find atom selections
    if line[0:6] == 'assign':

        line = line.replace('assign','')  # remove the 'assign' label
        assert line.count('(') >= 2 and line.count(')') >=2, "Can't find two parenthesis-enclosed atom selections!"
        sel1 = line[line.index('('):(line.index(')')+1)]
        line = line.replace(sel1, '')
        sel2 = line[line.index('('):(line.index(')')+1)]
        line = line.replace(sel2, '')

        distance, dupper, dlower = [float(s) for s in line.strip().split()]

        # find atom indices for selection 1
        Ind1 = indices_from_xplor_selection(sel1, topology)

        # find atom indices for selection 2
        Ind2 = indices_from_xplor_selection(sel2, topology)

        for i in Ind1:
            for j in Ind2:

                print biceps_restraint_line(restraint_index, i, j, topology, distance)

        # increment the restraint index
        restraint_index += 1

