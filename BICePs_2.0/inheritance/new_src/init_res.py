import os, sys, glob
import numpy as np
import mdtraj
from toolbox import *
from Observable import * # Containers for experimental observables
from Restraint import *



def init_res(PDB_filename, lam, energy, ref, data, uncern):
#        Restraint.__init__(self, PDB_filename, ref, use_global_ref_sigma=True)
        if not isinstance(ref, basestring):
            raise ValueError("reference potential type must be a 'str'")
        if not isinstance(lam,float):
            raise ValueError("lambda should be a single number with type of 'float'")
        if not isinstance(energy,float):                
            raise ValueError("energy should be a single number with type of 'float'")        
        if not uncern: # if it is an empty list
            sigma_min, sigma_max, dsigma=0.05, 20.0, 1.02
        else:
            if len(uncern) != 3:
                raise ValueError("uncertainty should be a list of three items: sigma_min, sigma_max, dsigma")
            else:
                sigma_min, sigma_max, dsigma = uncern[0], uncern[1], uncern[2]
        if data!= None:
            if data.endswith('cs_H'):
                R = Restraint_cs_H(PDB_filename,ref=ref)
                R.prep_observable(lam=lam, free_energy=energy,
                        filename=data)
                # Change the experimental Uncertainty after prepping observable
                R.exp_uncertainty(dlogsigma=dsigma, sigma_min=sigma_min,
                        sigma_max=sigma_max)
                

            elif data.endswith('cs_CA'):
                R = Restraint_cs_Ca(PDB_filename,ref=ref)
                R.prep_observable(lam=lam, free_energy=energy,
                        filename=data)
                R.exp_uncertainty(dlogsigma=dsigma, sigma_min=sigma_min,
                        sigma_max=sigma_max)

            elif data.endswith('cs_Ha'):
                R = Restraint_cs_Ha(PDB_filename,ref=ref)
                R.prep_observable(lam=lam, free_energy=energy,
                        filename=data)
                R.exp_uncertainty(dlogsigma=dsigma, sigma_min=sigma_min,
                        sigma_max=sigma_max)

            elif data.endswith('cs_N'):
                R = Restraint_cs_N(PDB_filename,ref=ref)
                R.prep_observable(lam=lam, free_energy=energy,
                        filename=data)
                R.exp_uncertainty(dlogsigma=dsigma, sigma_min=sigma_min,
                        sigma_max=sigma_max)

            elif data.endswith('J'):
                R = Restraint_J(PDB_filename,ref=ref)  # good ref
                R.prep_observable(lam=lam, free_energy=energy,
                        filename=data)
                R.exp_uncertainty(dlogsigma=dsigma, sigma_min=sigma_min,
                        sigma_max=sigma_max)

            elif data.endswith('noe'):
                R = Restraint_noe(PDB_filename,ref=ref)   # good ref
                R.prep_observable(lam=lam, free_energy=energy,
                        filename=data)
                R.exp_uncertainty(dlogsigma=dsigma, sigma_min=sigma_min,
                        sigma_max=sigma_max)

            elif data.endswith('pf'):
                R = Restraint_pf(PDB_filename,ref=ref)
                R.prep_observable(lam=lam, free_energy=energy,
                        filename=data)
                R.exp_uncertainty(dlogsigma=dsigma, sigma_min=sigma_min,
                        sigma_max=sigma_max)
        else:
            raise ValueError("Incompatible File extension. Use:{.noe,.J,.cs_H,.cs_Ha, .cs_Ca, .cs_N,.pf}")
        return R
