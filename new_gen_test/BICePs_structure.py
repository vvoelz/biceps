#!/usr/bin/env python

# Load Modules:{{{
import numpy as np
import os,sys

# }}}

class Sampler(object):
    """A class to perform posterior sampling of conformational populations"""
    def __init__(self,ensemble):
        print 'initialize class'
        self.ensembles = [ ensemble ]
        self.nstates = len(ensemble)
        self.nensembles = len(self.ensembles)
        print self.ensembles,self.nstates

    def build_ensembles(self):
        print 'building ensembles...'
        # lets say nstates is actually ambiguous states...
        for i in range(self.nstates):
            print self.ensembles[0][i]

    def neglogP(self, new_ensemble_index, new_state):
        s = self.ensembles[0][new_ensemble_index][new_state]
        print s
        sigma = 1. # random sigma value
        gamma = 2. # random gamma value
        mu = float(s.split('_')[1])
        self.neglogP = -np.log(np.exp(-(gamma-mu)**2./(2*sigma**(2.))))
        return self.neglogP

    def sample(self, steps):
        print 'I am sampling with %s steps....'%steps

# Structure:{{{
#class Structure():
#    def __init__(self,PDB,free_energy=None):
#        '''Initialize the class'''
#        self.free_energy = free_energy
#        self.observable_data = []
#        print('Initialize the class...')
#        print('Storing observable data in lists and arrays')
#
#    def load_data(self, filename):
#        print('--> Load data')
#       return "myClass method1"
#
#    def method3(self):
#        return myClass.method2(self,test_array)
#        print('--> Build groups')
# }}}

class RestraintFn(Sampler):
    def likelihood(self, Sampler, expData, new_ensemble_index, new_state):
        # experimental data goes somewhere...
        print expData
        Sampler.neglogP(self,new_ensemble_index, new_state)

    def restraint_0(self, ensemble, expData):
        print 'restraint_0 = Distances'

    def restraint_1(self, ensemble):
        print 'restraint_1 = NMR data'

    def restraint_2(self, ensemble):
        print 'restraint_2 = NOE'

    def restraint_3(self, ensemble):
        print 'restraint_3 = Whatever'

    def restraint_4(self, ensemble):
        print 'restraint_4'


