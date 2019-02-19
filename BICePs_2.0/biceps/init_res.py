import os, sys, glob
import numpy as np
import mdtraj
from toolbox import *
from Observable import * # Containers for experimental observables
from Restraint import *



def init_res(PDB_filename, lam, energy, ref=None, data, uncern=None, gamma=None):
    """
    Initialize corresponding restraint class based on experimental observables in input files for each conformational state.

    :param str PDB_filename: topology file name ('*.pdb')

    :param float lam: lambdas

    :param float energy: potential energy for each conformational state

    :param str default=None ref: reference potential (if default, will use our suggested reference potential for each experimental observables)

    :param str data: BICePs input files directory

    :param list default=None uncern: nuisance parameters range (if default, will use our suggested broad range (may increase sampling requirement for convergence))

    :param list default=None gamma: only for NOE, range of gamma (if default, will use our suggested broad range (may increase sampling requirement for convergence)) 
    """
#        Restraint.__init__(self, PDB_filename, ref, use_global_ref_sigma=True)
        if not isinstance(ref, basestring):
            raise ValueError("reference potential type must be a 'str'")
        if not isinstance(lam,float):
            raise ValueError("lambda should be a single number with type of 'float'")
        if not isinstance(energy,float):                
            raise ValueError("energy should be a single number with type of 'float'")        
        if uncern ==  None:
            sigma_min, sigma_max, dsigma=0.05, 20.0, np.log(1.02)
        else:
            if len(uncern) != 3:
                raise ValueError("uncertainty should be a list of three items: sigma_min, sigma_max, dsigma")
            else:
                sigma_min, sigma_max, dsigma = uncern[0], uncern[1], np.log(uncern[2])
        if gamma ==  None:
            gamma_min, gamma_max, dgamma = 0.05, 20.0, np.log(1.02)
        else:
            if len(gamma) != 3:
                raise ValueError("gamma should be a list of three items: gamma_min, gamma_max, dgamma")
            else:
                gamma_min, gamma_max, dgamma = gamma[0], gamma[1], np.log(gamma[2])

        if data!= None:
            if data.endswith('cs_H'):
                if ref ==  None:
                    R = Restraint_cs_H(PDB_filename,ref='exp',dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                    R.prep_observable(lam=lam, free_energy=energy, filename=data)
                else:
                    R = Restraint_cs_H(PDB_filename,ref=ref,dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                    R.prep_observable(lam=lam, free_energy=energy, filename=data)


            elif data.endswith('cs_CA'):
                if ref == None:
                    R = Restraint_cs_Ca(PDB_filename,ref='exp',dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                    R.prep_observable(lam=lam, free_energy=energy, filename=data)
                else:
                    R = Restraint_cs_Ca(PDB_filename,ref=ref,dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                    R.prep_observable(lam=lam, free_energy=energy, filename=data)

            elif data.endswith('cs_Ha'):
                if ref == None:
                    R = Restraint_cs_Ha(PDB_filename,ref='exp',dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                    R.prep_observable(lam=lam, free_energy=energy, filename=data)

                else:
                    R = Restraint_cs_Ha(PDB_filename,ref=ref,dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                    R.prep_observable(lam=lam, free_energy=energy, filename=data)

            elif data.endswith('cs_N'):
                if ref == None:
                    R = Restraint_cs_N(PDB_filename,ref='exp',dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                    R.prep_observable(lam=lam, free_energy=energy, filename=data)
                else:
                    R = Restraint_cs_N(PDB_filename,ref=ref,dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                    R.prep_observable(lam=lam, free_energy=energy, filename=data)


            elif data.endswith('J'):
                if ref == None:
                    R = Restraint_J(PDB_filename,ref='uniform',dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)   
                    R.prep_observable(lam=lam, free_energy=energy, filename=data)
                else:
                    R = Restraint_J(PDB_filename,ref=ref,dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                    R.prep_observable(lam=lam, free_energy=energy, filename=data)

            elif data.endswith('noe'):
                if ref == None:
                    R = Restraint_noe(PDB_filename,ref='gau',dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                    R.prep_observable(lam=lam, free_energy=energy, filename=data, dloggamma = dgamma, gamma_min = gamma_min, gamma_max = gamma_max)
                else:
                    R = Restraint_noe(PDB_filename,ref=ref,dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                    R.prep_observable(lam=lam, free_energy=energy, filename=data, dloggamma = dgamma, gamma_min = gamma_min, gamma_max = gamma_max)
                    
            elif data.endswith('pf'):
                if ref == None:
                    R = Restraint_pf(PDB_filename,ref='exp',dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                    R.prep_observable(lam=lam, free_energy=energy, filename=data)
                else:
                    R = Restraint_pf(PDB_filename,ref=ref,dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                    R.prep_observable(lam=lam, free_energy=energy, filename=data)

        else:
            raise ValueError("Incompatible File extension. Use:{.noe,.J,.cs_H,.cs_Ha, .cs_Ca, .cs_N,.pf}")
        return R

__all__ = [
    'init_res',
]

