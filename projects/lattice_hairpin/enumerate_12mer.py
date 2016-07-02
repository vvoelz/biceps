#! /usr/bin/python

usage = """
enumerate.py <configfile>

Try:  enumerate.py enumerate.conf

This program will read in an HP chain specified in the configure file,
and perform a full enumeration of conformational space.

The problem tablulates:

    1) the density of states (in energies/contacts)
    
    2) the number density of unique contact states, i.e. disjoint collections
       of microscopic conformations all sharing a unique set of interresidue contacts. 

These values are printed as output.

"""


import sys
sys.path.append('/Users/vv/git/HPSandbox')

from Config import *
from Chain import *
from Monty import *
from Replica import *
from Trajectory import *

import random
import string
import math
import os

g = random.Random(randseed)


if len(sys.argv) < 2:
    print 'Usage:  enumerate.py <configfile>'
    sys.exit(1)
 
 
VERBOSE = 1
    

if __name__ == '__main__':

    if len(sys.argv) < 2:
	print usage
	sys.exit(1)
	
    configfile = sys.argv[1]
    config = Config( filename=configfile)
    if VERBOSE: config.print_config()
    
    # create a single Replica
    replicas = [ Replica(config,0) ]
    
    traj = Trajectory(replicas,config)	# a trajectory object to write out trajectories

    nconfs = 0
    contact_states = {}		# dictionary of {repr{contact state}: number of conformations}
    contacts = {}               # dictionary of {number of contacts: number of conformations}


    #################
    #
    # This is a useful subroutine for enumerating all conformations of an HP chain
    #
    # NOTE: in order for this to work correctly, the initial starting vector must be [0,0,0,....,0]
    # 
    done = 0
    while not(done):
	    	    
	if len(replicas[0].chain.vec) == replicas[0].chain.n-1:    
	    if replicas[0].chain.viable:		
		if replicas[0].chain.nonsym():
		    
		    # tally the number of contacts
		    state = replicas[0].chain.contactstate()
		    ncontacts = len(state)
		    if contacts.has_key(ncontacts) == False:
			contacts[ncontacts] = 1
		    else:
			contacts[ncontacts] = contacts[ncontacts] + 1

		    # tally the contact state
		    this_state_repr = repr(state)
		    if contact_states.has_key(this_state_repr) == False:
			contact_states[this_state_repr] = 1
		    else:
			contact_states[this_state_repr] = contact_states[this_state_repr] + 1

		    # tally the number of conformations
		    nconfs = nconfs + 1

		    # write to trajectory
		    if (nconfs % config.TRJEVERY) == 0:
			traj.queue_trj(replicas[0])
		    # print progress
		    if (nconfs % config.PRINTEVERY) == 0:
			print '%-4d confs  %s'%(nconfs,replicas[0].chain.vec)
    
		done = replicas[0].chain.shift()
		    
	    else:
		done = replicas[0].chain.shift()

	else:
	    if replicas[0].chain.viable:
		replicas[0].chain.grow()
	    else:
		done = replicas[0].chain.shift()

	if replicas[0].chain.vec[0] == 1:    # skip the other symmetries
	    break	
    #
    #
    #################
        
    
    # write the last of the trj and ene buffers
    # and close all the open trajectory file handles
    traj.cleanup(replicas)
    
    # print out the density of contact states
    print
    print 'DENSITY of CONTACT STATES:'
    print '%s %-40s %s'%('index', 'contact state','number of conformations')
    for istate in range(len(contact_states.keys())):
        state = contact_states.keys()[istate]
	print '%d %-40s %d'%(istate, state, contact_states[state])
    
    # print out the density of states (energies)
    print 
    print 'DENSITY of STATES (in energies/contacts):'
    print '%-20s %-20s %s'%('number of contacts','energy (kT)','number of conformations')
    for c in contacts.keys():
	print '%-20d %-20d %d'%(c,config.eps*c,contacts[c])
    print
    print 'at T = %4.1f K'%config.T
	

    
