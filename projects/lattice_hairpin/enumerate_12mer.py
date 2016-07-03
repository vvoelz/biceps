#! /usr/bin/env python

import sys
sys.path.append('/Users/vv/git/HPSandbox')

from Config import *
from Chain import *
from Monty import *
from Replica import *
from Trajectory import *

import random, string, os

import numpy as np
# Required: http://mdtraj.org/
import mdtraj as md

g = random.Random(randseed)


if len(sys.argv) < 2:
    print """Usage:  enumerate_12mer.py <configfile>

        Try: >>> python enumerate_12mer.py 12mer.conf """
    sys.exit(1)
 
 
VERBOSE = 1

# make an MDTraj trajectory to store the coordinate frames as PDBs 
t = md.load('template.pdb')  # note the first frame is junk

if __name__ == '__main__':

    configfile = sys.argv[1]
    config = Config( filename=configfile)
    if VERBOSE: config.print_config()
    
    # create a single Replica
    replicas = [ Replica(config,0) ]
    
    traj = Trajectory(replicas,config)	# a trajectory object to write out trajectories

    nconfs = 0
    contact_states = {}		# dictionary of {repr{contact state}: number of conformations}
    contacts = {}               # dictionary of {number of contacts: number of conformations}

    contact_state_keys = []     # a list of unique contact states, e.g. elements "[(0,9),(0,11)]", whose order defines the
                                # contact state indices
         
    assignments = []  # a list of contact state indices for every frame in the trajectory

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
                    if contact_state_keys.count(this_state_repr) == 0:
                        contact_state_keys.append(this_state_repr)
			contact_states[this_state_repr] = 1
		    else:
			contact_states[this_state_repr] = contact_states[this_state_repr] + 1

		    # tally the number of conformations
		    nconfs = nconfs + 1

                    # VAV testing
                    # print nconfs, replicas[0].chain.coords 
                    # add a new frame to the MDTraj trajectory, using  our template.pdb

                    ## if this is the first frame of the MDTraj Trajectory, just replace the coordinates
                    if nconfs == 1:
                        t.xyz[0,:,0] = np.array(replicas[0].chain.coords)[:,0]*0.1 # convert to nm
                        t.xyz[0,:,1] = np.array(replicas[0].chain.coords)[:,1]*0.1
                    else:
                        tnew = md.load('template.pdb') 
                        tnew.xyz[0,:,0] = np.array(replicas[0].chain.coords)[:,0]*0.1
                        tnew.xyz[0,:,1] = np.array(replicas[0].chain.coords)[:,1]*0.1
                        t = t.join(tnew)
                    print t 

                    # add the contact state index to the assignments
                    assignments.append( contact_state_keys.index(this_state_repr) )

		    # write to HPSandbox trajectory
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
    
    # Save the MDTraj traj as a PDB
    print 'Saving the MDTraj microstate trajectory as microstates.pdb ...'
    t.save_pdb('microstates.pdb')
    print '...Done.'

    # compute pairwise distances for all frames
    atom_pairs = np.loadtxt('distance_indices.dat')
    distances = md.compute_distances(t, atom_pairs)*10.0  # convert back to angstroms
    np.savetxt('distances.dat', distances, fmt='%8.6f') 

    # write contact state assignments to file
    assignments = np.array(assignments)
    np.savetxt('assignments.dat', assignments, fmt='%d')

    # write mean distances for each contact state
    for istate in range(len(contact_state_keys)):
        # slice out only frames from this state
        mean_distances = distances[(assignments == istate),:].mean(axis=0)
        np.savetxt('mean_distances/mean_distances_state%d.dat'%istate, mean_distances, fmt='%8.6f')

    # write the contact state indices to file
    fout = open('contact_state_indices.dat','w')
    fout.write('#index\tcontacts\tncontacts\tnconfs\n')
    for istate in range(len(contact_state_keys)):
       ncontacts = len(eval(contact_state_keys[istate]))
       nconfs = contact_states[contact_state_keys[istate]]
       fout.write('%d\t%s\t%d\t%d\n'%(istate,contact_state_keys[istate], ncontacts, nconfs))
    fout.close()

    # calculate perfect experimental distances
    ncontacts_by_microstate = np.array([len(eval(contact_state_keys[istate])) for istate in assignments])
    pops = np.exp(-config.eps*ncontacts_by_microstate)
    pops = pops/pops.sum()
    exp_distances = np.dot(pops,distances)
    np.savetxt('exp_distances.dat', exp_distances, fmt='%8.6f')

    print '#contact state\tpopulation'
    for istate in range(len(contact_state_keys)):
       print istate, pops[istate] 



    ###########
    
    # write the last of the trj and ene buffers
    # and close all the open trajectory file handles
    traj.cleanup(replicas)
    
    # print out the density of contact states
    print
    print 'DENSITY of CONTACT STATES:'
    print '%s %-40s %s'%('index', 'contact state','number of conformations')
    for istate in range(len(contact_state_keys)):
        state = contact_state_keys[istate]
	print '%d %-40s %d'%(istate, state, contact_states[state])
    
    # print out the density of states (energies)
    print 
    print 'DENSITY of STATES (in energies/contacts):'
    print '%-20s %-20s %s'%('number of contacts','energy (kT)','number of conformations')
    for c in contacts.keys():
	print '%-20d %-20d %d'%(c,config.eps*c,contacts[c])
    print
    print 'at T = %4.1f K'%config.T
	
