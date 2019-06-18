import os, sys, glob
import numpy as np
import mdtraj
from toolbox import *
from Observable import * # Containers for experimental observables
from Restraint import *



def init_res(PDB_filename, lam, energy, data, ref=None, uncern=None, gamma=None, precomputed_pf = False, Ncs=None, Nhs=None):
    """Initialize corresponding restraint class based on experimental observables in input files for each conformational state.

    :param str PDB_filename: topology file name ('*.pdb')

    :param float lam: lambdas

    :param float energy: potential energy for each conformational state

    :param str default=None ref: reference potential (if default, will use our suggested reference potential for each experimental observables)

    :param str data: BICePs input files directory

    :param list default=None uncern: nuisance parameters range (if default, will use our suggested broad range (may increase sampling requirement for convergence))

    :param list default=None gamma: only for NOE, range of gamma (if default, will use our suggested broad range (may increase sampling requirement for convergence))"""

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
    if not precomuted_pf:
        if Ncs or Nhs == None:
            raise ValueError("Ncs and Nhs are needed!")
        # add uncern option here later
        # don't trust these numbers, need to be confirmed!!! Yunhui 06/2019
        beta_c_min, beta_c_max, dbeta_c = 0.05, 0.25, 0.01
        beta_h_min, beta_h_max, dbeta_h = 0.0, 5.2, 0.2
        beta_0_min, beta_0_max, dbeta_0 = -10.0, 0.0, 0.2
        xcs_min, xcs_max, dxcs = 5.0, 8.5, 0.5
        xhs_min, xhs_max, dxhs = 2.0, 2.8, 0.1
        bs_min, bs_max, dbs = 3.0, 21.0, 1.0

        allowed_xcs=np.arange(xcs_min,xcs_max,dxcs)
        allowed_xhs=np.arange(xhs_min,xhs_max,dxhs)
        allowed_bs=np.arange(bs_min,bs_max,dbs)


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
            if not precomuted_pf:
                if Ncs or Nhs == None:
                    raise ValueError("Ncs and Nhs are needed!")
            # add uncern option here later
            # don't trust these numbers, need to be confirmed!!! Yunhui 06/2019
            beta_c_min, beta_c_max, dbeta_c = 0.05, 0.25, 0.01
            beta_h_min, beta_h_max, dbeta_h = 0.0, 5.2, 0.2
            beta_0_min, beta_0_max, dbeta_0 = -10.0, 0.0, 0.2
            xcs_min, xcs_max, dxcs = 5.0, 8.5, 0.5
            xhs_min, xhs_max, dxhs = 2.0, 2.8, 0.1
            bs_min, bs_max, dbs = 3.0, 21.0, 1.0

            allowed_xcs=np.arange(xcs_min,xcs_max,dxcs)
            allowed_xhs=np.arange(xhs_min,xhs_max,dxhs)
            allowed_bs=np.arange(bs_min,bs_max,dbs)
            # 107=residue numbers, Nc/Nh file names are hard coded for now. Yunhui 06/19
            Ncs=np.zeros((len(allowed_xcs),len(allowed_bs),107))
            Nhs=np.zeros((len(allowed_xhs),len(allowed_bs),107))
            for o in range(len(allowed_xcs)):
                for p in range(len(allowed_xhs)):
                    for q in range(len(allowed_bs)):
                        infile_Nc='input/Nc/Nc_x%0.1f_b%d_state%03d.npy'%(allowed_xcs[o], allowed_bs[q],i)
                        infile_Nh='input/Nh/Nh_x%0.1f_b%d_state%03d.npy'%(allowed_xhs[p], allowed_bs[q],i)
                        Ncs[o,q,:] = (np.load(infile_Nc))
                        Nhs[p,q,:] = (np.load(infile_Nh))


            if ref == None:
                R = Restraint_pf(PDB_filename,ref='exp',dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                R.prep_observable(lam=lam, free_energy=energy, filename=data,precomputed_pf,Ncs, Nhs)
            else:
                R = Restraint_pf(PDB_filename,ref=ref,dlogsigma=dsigma, sigma_min=sigma_min,sigma_max=sigma_max)
                R.prep_observable(lam=lam, free_energy=energy, filename=data,precomputed_pf,Ncs, Nhs)

    else:
        raise ValueError("Incompatible File extension. Use:{.noe,.J,.cs_H,.cs_Ha, .cs_Ca, .cs_N,.pf}")
    return R

#__all__ = [
#    'init_res',
#]

