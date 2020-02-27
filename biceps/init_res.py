import os, sys, glob
import numpy as np
import mdtraj
import biceps.Restraint as Restraint

def init_res(PDB_filename, lam, energy, data, ref=None, uncern=None,
        gamma=None, precomputed_pf = False, Ncs_fi=None, Nhs_fi=None,
        state = None):
    """Initialize corresponding restraint class based on experimental observables in input files for each conformational state.

    :param str PDB_filename: topology file name ('*.pdb')
    :param float lam: lambdas
    :param float energy: potential energy for each conformational state
    :param str default=None ref: reference potential (if default, will use our suggested reference potential for each experimental observables)
    :param str data: BICePs input files directory
    :param list default=None uncern: nuisance parameters range (if default, will use our suggested broad range (may increase sampling requirement for convergence))
    :param list default=None gamma: only for NOE, range of gamma (if default, will use our suggested broad range (may increase sampling requirement for convergence))"""

    if ref is not None:
        if not isinstance(ref, str):
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
        gamma_min, gamma_max, dloggamma = 0.05, 20.0, np.log(1.02)
    else:
        if len(gamma) != 3:
            raise ValueError("gamma should be a list of three items: gamma_min, gamma_max, dgamma")
        else:
            gamma_min, gamma_max, dloggamma = gamma[0], gamma[1], np.log(gamma[2])
    #if precomputed_pf == False:
    #    if Ncs == None or Nhs == None:
    #        raise ValueError("Ncs and Nhs are needed!")
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
        if data.endswith('pf'):
            if not precomputed_pf:
                if Ncs_fi == None or Nhs_fi == None or state == None:
                    raise ValueError("Ncs and Nhs and state numebr are needed!")
            # add uncern option here later
            # don't trust these numbers, need to be confirmed!!! Yunhui 06/2019
            beta_c_min, beta_c_max, dbeta_c = 0.05, 0.25, 0.01
            beta_h_min, beta_h_max, dbeta_h = 0.0, 5.2, 0.2
            beta_0_min, beta_0_max, dbeta_0 = -10.0, 0.0, 0.2
            xcs_min, xcs_max, dxcs = 5.0, 8.5, 0.5
            xhs_min, xhs_max, dxhs = 2.0, 2.7, 0.1
            bs_min, bs_max, dbs = 15.0, 16.0, 1.0

            allowed_xcs=np.arange(xcs_min,xcs_max,dxcs)
            allowed_xhs=np.arange(xhs_min,xhs_max,dxhs)
            allowed_bs=np.arange(bs_min,bs_max,dbs)
            # 107=residue numbers, Nc/Nh file names are hard coded for now. Yunhui 06/19
            Ncs=np.zeros((len(allowed_xcs),len(allowed_bs),107))
            Nhs=np.zeros((len(allowed_xhs),len(allowed_bs),107))
            for o in range(len(allowed_xcs)):
                for q in range(len(allowed_bs)):
                    infile_Nc='%s/Nc_x%0.1f_b%d_state%03d.npy'%(Ncs_fi, allowed_xcs[o], allowed_bs[q],state)
                    Ncs[o,q,:] = (np.load(infile_Nc))
            for p in range(len(allowed_xhs)):
                for q in range(len(allowed_bs)):
                    infile_Nh='%s/Nh_x%0.1f_b%d_state%03d.npy'%(Nhs_fi, allowed_xhs[p], allowed_bs[q],state)
                    Nhs[p,q,:] = (np.load(infile_Nh))

        ###########################################
        # Generalizing (input data --> restraint)
        ###########################################
        if ref ==  None:
            # TODO: place the default ref inside the class
            ref = 'exp' # 'uniform'
        extension = data.split(".")[-1]
        R = getattr(Restraint, "Restraint_%s"%(extension))
        R = R(PDB_filename, ref, dsigma, sigma_min, sigma_max)
        args = {"%s"%key: val for key,val in locals().items()
                if key in R.prep_observable.__code__.co_varnames}
        R.prep_observable(**args)
    else:
        raise ValueError("Incompatible File extension. Use:{.noe,.J,.cs_H,.cs_Ha, .cs_Ca, .cs_N,.pf}")
    return R



