import sys, os, glob

sys.path.append('../../')
from Structure import *
from PosteriorSampler import *

import cPickle  # to read/write serialized sampler classes

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("lam", help="a lambda value between 0.0 and 1.0  denoting the Hamiltonian weight (E_data + lambda*E_QM)", type=float)
parser.add_argument("outdir", help="the name of the output directory")
parser.add_argument("nsteps", help="Number of sampling steps", type=int)
parser.add_argument("--noref", help="Do not use reference potentials (default is to use them)",
                    action="store_true")
parser.add_argument("--lognormal", help="Use log-normal distance restraints (default is normal)",
                    action="store_true")
parser.add_argument("--verbose", help="use verbose output",
                    action="store_true")
args = parser.parse_args()


print '=== Settings ==='
print 'lam', args.lam
print 'outdir', args.outdir
print 'nsteps', args.nsteps
print '--noref', args.noref
print '--lognormal', args.lognormal
print '--verbose', args.verbose



"""OUTPUT 

    Files written:
        <outdir>/traj_lambda_<lambda>.yaml  - YAML Trajectory file 
        <outdit>/sampler_<lambda>.pkl       - a cPickle'd sampler object
"""
#T1=[0]
T1=[0,7,12,14,16,18,21]
T2=[0,6,8,14,15,16,18,19,20,21,23,24]
T3=[0,5,7,8,11,12,14,15,16,17,18,19,20,21,22,23,24]
T4=[0,3,8,11,12,14,15,16,17,18,19,20,21,22,23,24]
T5=[0,1,4,9,10,12,14,16,18,19,20,21,24]
T6=[0,2,4,5,8,9,11,12,13,14,15,16,17,18,19,20,21,22,24]



# Make a new directory if we have to
if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)


#########################################
# Let's create our ensemble of structures

# experimental restraints 
expdata_filename_PF = '../../apoMb.pf' 
#expdata_filename_J = ''
#expdata_filename_cs_H = 'apoMb.chemicalshift'
#expdata_filename_cs_Ha =
#expdata_filename_cs_N = 'apoMb_N.chemicalshift'
#expdata_filename_cs_Ca = 'apoMb_Ca.chemicalshift'
# model energies
#dist_prior = np.load('sampled_states_array.npy')
if (1):
    nclusters = len(T1)
    energies_filename = '../../new_traj/tram/final_300K/energy_model_0.txt' #'energy.txt'
    energies = loadtxt(energies_filename)
 #   print 'energies.shape', energies.shape
    energies -= energies.min()  # set ground state to zero, just in case

#sys.exit()
#if (1):
    # model distances 
#    model_distances = loadtxt('NOE/rminus6_whole_state0.txt')  #GYH:We have rminus6 data already
#    print 'model_distances.shape', model_distances.shape
#    print 'model_distances', model_distances

############

# We will instantiate a number of Structure() objects to construct the ensemble
ensemble = []

QuickTest = False  # Test
if QuickTest:
    allowed_xcs=np.arange(5.0,6.0,0.5)	# change this after test 05/2017
    allowed_xhs=np.arange(2.0,2.1,0.1)	# change this after test 05/2017
    allowed_bs=np.arange(3.0,5.0,1.0)	# change this after test 05/2017
else:
    allowed_xcs = np.arange(5.0,  8.5, 0.5)  # change this after test 05/2017
    allowed_xhs = np.arange(2.0,  2.7, 0.1)  # change this after test 05/2017
    allowed_bs  = np.arange(18.0, 19.0, 1.0)   # change this after test 05/2017



for i in range(nclusters):
#for i in range(2):
    print
    print '#### STRUCTURE %d ####'%T1[i]

    # no information from QM --> lam = 0.0
    # QM + exp               --> lam = 1.0
    ## s = Structure('gens-pdb-kcenters-dih-1.8/Gen%d.pdb'%i, args.lam*energies[i], expdata_filename, use_log_normal_distances=False)
#    model_protectionfactor = loadtxt('FH/pf_state%d.txt'%i)
#    model_protectionfactor = loadtxt('../HDX/PF/test/pf_state%d.txt'%i)
#    model_protectionfactor = loadtxt('new_traj/PF/no_F/T6/pf_state%d.txt'%i) 
#    model_protectionfactor = loadtxt('new_traj/PF/T3/pf_state%d.txt'%i) 

#    model_protectionfactor = loadtxt('new_traj/PF/new_pf/for_yunhui/txt/T4/pf_state%d.txt'%i) 
#    model_chemicalshift_H = loadtxt('new_traj/H/T6/cs_%d.txt'%i)
#    model_chemicalshift_N = loadtxt('new_traj/N/T6/cs_%d.txt'%i)
#    model_chemicalshift_Ca = loadtxt('new_traj/Ca/T6/cs_%d.txt'%i)
#    model_chemicalshift_H = loadtxt('EGH/H/cs_%d.txt'%i)
#    model_chemicalshift_N = loadtxt('EGH/N/cs_%d.txt'%i)
#    model_chemicalshift_Ca = loadtxt('EGH/Ca/cs_%d.txt'%i)

#    s = Structure('Gens/Gens%d.pdb'%i, args.lam*energies[i], expdata_filename_noe, expdata_filename_J, expdata_filename_cs, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0)
#    s = Structure('new_traj/pdb/state%d.pdb'%i, args.lam*energies[i],  expdata_filename_cs_H=expdata_filename_cs_H, expdata_filename_cs_N=expdata_filename_cs_N,expdata_filename_cs_Ca=expdata_filename_cs_Ca, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0, dalpha=0.1, alpha_min=0.1, alpha_max=5.0)	#GYH
#    s = Structure('../HDX/states/test/state%d.pdb'%i, args.lam*energies[i],  expdata_filename_PF=expdata_filename_PF, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0, dalpha=0.001, alpha_min=3.831, alpha_max=3.832)        #GYH

#    s = Structure('new_traj/pdb/T6/state%d.pdb'%i, args.lam*energies[i],  expdata_filename_cs_H=expdata_filename_cs_H, expdata_filename_cs_N=expdata_filename_cs_N,expdata_filename_cs_Ca=expdata_filename_cs_Ca, expdata_filename_PF=expdata_filename_PF, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0, dalpha=0.1, alpha_min=-10.0, alpha_max=10.0) #GYH

#    s = Structure('new_traj/pdb/T3/state%d.pdb'%i, args.lam*energies[i],  expdata_filename_cs_H=expdata_filename_cs_H, expdata_filename_cs_N=expdata_filename_cs_N,expdata_filename_cs_Ca=expdata_filename_cs_Ca, expdata_filename_PF=expdata_filename_PF, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0) #GYH


    # Read in the values of Nc and Nh for all residues, and all nuisance parm values
    Ncs=np.zeros((len(allowed_xcs),len(allowed_bs),107))
    for o in range(len(allowed_xcs)):
        for q in range(len(allowed_bs)):
                infile_Nc='../../input/Nc/Nc_x%0.1f_b%d_state%03d.npy'%(allowed_xcs[o], allowed_bs[q],T1[i])
                Ncs[o,q,:] = (np.load(infile_Nc))

    Nhs=np.zeros((len(allowed_xhs),len(allowed_bs),107))
    for p in range(len(allowed_xhs)):
	for q in range(len(allowed_bs)):
                infile_Nh='../../input/Nh/Nh_x%0.1f_b%d_state%03d.npy'%(allowed_xhs[p], allowed_bs[q],T1[i])
		Nhs[p,q,:] = (np.load(infile_Nh))
    #sys.exit()

#    print Ncs[0,0,0]
#    print Nhs.shape
#    sys.exit()    

    if QuickTest:
        s = Structure('new_traj/pdb/state%d.pdb'%T1[i], args.lam*energies[i],
			expdata_filename_PF=expdata_filename_PF,
			use_log_normal_distances=False,
			dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0, 
			dbeta_c=0.005, beta_c_min=0.02, beta_c_max=0.03,
			dbeta_h=0.05, beta_h_min=0.00, beta_h_max=0.10,
			dbeta_0=0.2, beta_0_min=0.0, beta_0_max=0.4,
			dxcs=0.5, xcs_min=5.0, xcs_max=6.0,
			dxhs=0.1, xhs_min=2.0, xhs_max=2.1,
			dbs=1.0, bs_min=3.0, bs_max=5.0,
			Ncs=Ncs, Nhs=Nhs)        #GYH
    else:
        s = Structure('../../new_traj/pdb/state%d.pdb'%T1[i], args.lam*energies[i],
                        expdata_filename_PF=expdata_filename_PF,
                        use_log_normal_distances=False,
                        dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0,
                        dbeta_c=0.01, beta_c_min=0.05, beta_c_max=0.25,
                        dbeta_h=0.2, beta_h_min=0.0, beta_h_max=5.2,
                        dbeta_0=0.2, beta_0_min=-10.0, beta_0_max=0.0,
                        dxcs=0.5, xcs_min=5.0, xcs_max=8.5,
                        dxhs=0.1,xhs_min=2.0, xhs_max=2.7,
                        dbs=1.0, bs_min=18.0, bs_max=19.0,
                        Ncs=Ncs, Nhs=Nhs)        #GYH


    

#    sys.exit()

#    s = Structure('Gens/Gens%d.pdb'%i, args.lam*energies[i], expdata_filename_noe=expdata_filename_noe, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0)

    # NOTE: Upon instantiation, each Structure() object computes the distances from the given PDB.
    #       However, our clusters represent averaged conformation states, and so we   
    #       need to replace these values with our own r^-6-averaged, precomputed ones

    # replace PDB distances with r^-6-averaged distances
#    print 'len(s.distance_restraints)', len(s.distance_restraints)
#    for j in range(len(s.distance_restraints)):
#        print s.distance_restraints[j].i, s.distance_restraints[j].j, model_distances[j]
#        s.distance_restraints[j].model_distance = model_distances[j]

    # replace chemicalshift  with  precomputed ones
#    print 'len(s.chemicalshift_H_restraints)', len(s.chemicalshift_H_restraints)
#    for j in range(len(s.chemicalshift_H_restraints)):
#        print s.chemicalshift_H_restraints[j].i, model_chemicalshift_H[j]
#        s.chemicalshift_H_restraints[j].model_chemicalshift_H = model_chemicalshift_H[j]

    # replace chemicalshift  with  precomputed ones
#    print 'len(s.chemicalshift_N_restraints)', len(s.chemicalshift_N_restraints)
#    for j in range(len(s.chemicalshift_N_restraints)):
#        print s.chemicalshift_N_restraints[j].i, model_chemicalshift_N[j]
#        s.chemicalshift_N_restraints[j].model_chemicalshift_N = model_chemicalshift_N[j]

    # replace chemicalshift  with  precomputed ones
#    print 'len(s.chemicalshift_Ca_restraints)', len(s.chemicalshift_Ca_restraints)
#    for j in range(len(s.chemicalshift_Ca_restraints)):
#        print s.chemicalshift_Ca_restraints[j].i, model_chemicalshift_Ca[j]
#        s.chemicalshift_Ca_restraints[j].model_chemicalshift_Ca = model_chemicalshift_Ca[j]


    # replace protection_factor with precomputed ones				#GYH
#    print 'len(s.protectionfactor_restraints)', len(s.protectionfactor_restraints)
#    for j in range(len(s.protectionfactor_restraints)):
#        print s.protectionfactor_restraints[j].i, s.protectionfactor_restraints[j].model_protectionfactor[1,1,1,1,1,1]
#        s.protectionfactor_restraints[j].model_protectionfactor = model_protectionfactor[j]


    # update the chemicalshift sse's!
#    s.compute_sse_chemicalshift_H()
#    s.compute_sse_chemicalshift_N()
#    s.compute_sse_chemicalshift_Ca()

    # update the distance sse's!			
#    s.compute_sse_distances()

    # update the protectionfactor sse's!		#GYH
    s.compute_sse_protectionfactor()
#    print 'state:', i, 's.sse_protectionfactor', s.sse_protectionfactor[0,0,0,1,0,0]
    # add the structure to the ensemble
    ensemble.append( s )
#    sys.exit()



# Print out the agreement for model 53 (highest-pop)
#for drest in ensemble[53].distance_restraints:
#    print 'state 53 d[ %d - %d ] ='%(drest.i, drest.j), drest.model_distance, 'd_exp =', drest.exp_distance

#model_distances = [drest.model_distance for drest in ensemble[53].distance_restraints]
#exp_distances = [drest.exp_distance for drest in ensemble[53].distance_restraints]

if (0):
  plt.figure()
  lookat = [51, 53, 55, 0]
  for i in range(len(lookat)):
    model_distances = [drest.model_distance for drest in ensemble[lookat[i]].distance_restraints]
    exp_distances = [drest.exp_distance for drest in ensemble[lookat[i]].distance_restraints]
    print 'model %d'%lookat[i], exp_distances, 'model_distances', model_distances
    plt.subplot(2,2,i+1) 
    plt.plot(exp_distances, model_distances,'.')
    plt.plot([0,5],[0,5],'k-')
    plt.xlabel('exp distance ($\\AA$)')
    plt.ylabel('model distance ($\\AA$)')
    plt.title('model %d'%lookat[i])
  plt.show()


  ##########################################
  # Next, let's do some posterior sampling


else:
  #sampler = PosteriorSampler(ensemble, use_reference_prior=True, sample_ambiguous_distances=False)
  sampler = PosteriorSampler(ensemble, dlogsigma_noe=np.log(1.01), sigma_noe_min=0.2, sigma_noe_max=10.0,
                                dlogsigma_J=np.log(1.02), sigma_J_min=0.05, sigma_J_max=20.0,
                                dlogsigma_cs_H=np.log(1.01), sigma_cs_H_min=0.01, sigma_cs_H_max=10.0,
				dlogsigma_cs_Ha=np.log(1.01), sigma_cs_Ha_min=0.01, sigma_cs_Ha_max=5.0,
				dlogsigma_cs_N=np.log(1.01), sigma_cs_N_min=0.01, sigma_cs_N_max=10.0,
				dlogsigma_cs_Ca=np.log(1.01), sigma_cs_Ca_min=0.01, sigma_cs_Ca_max=10.0,
				dlogsigma_PF=np.log(1.01), sigma_PF_min=0.01, sigma_PF_max=10.0,		#GYH
				dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0,
#				dbeta_c=0.005, beta_c_min=0.02, beta_c_max=0.03,
#                                dbeta_h=0.05, beta_h_min=0.00, beta_h_max=0.10,
#                                dbeta_0=0.2, beta_0_min=0.0, beta_0_max=0.4,
#                                dxcs=0.5, xcs_min=5.0, xcs_max=6.0,
#                                dxhs=0.1, xhs_min=2.0, xhs_max=2.1,
#                                dbs=1.0, bs_min=3.0, bs_max=5.0,
                                dbeta_c=0.01, beta_c_min=0.05, beta_c_max=0.25,
                                dbeta_h=0.2, beta_h_min=0.0, beta_h_max=5.2,
                                dbeta_0=0.2, beta_0_min=-10.0, beta_0_max=0.0,
                                dxcs=0.5, xcs_min=5.0, xcs_max=8.5,
                                dxhs=0.1, xhs_min=2.0, xhs_max=2.7,
                                dbs=1.0, bs_min=18.0, bs_max=19.0,
                                use_reference_prior_noe=False,
                                use_reference_prior_H=False,
				use_reference_prior_Ha=False,
				use_reference_prior_N=False,
				use_reference_prior_Ca=False,
 				use_reference_prior_PF=True,         # <-------
				sample_ambiguous_distances=False,
				use_gaussian_reference_prior_noe=False,
				use_gaussian_reference_prior_H=False,
				use_gaussian_reference_prior_Ha=False,
				use_gaussian_reference_prior_N=False,
				use_gaussian_reference_prior_Ca=False,
				use_gaussian_reference_prior_PF=False)
#sys.exit()
#if(0):

  
  #sampler = PosteriorSampler(ensemble, use_reference_prior=True)
  sampler.sample(args.nsteps)  # number of steps
  print 'Processing trajectory...',
  sampler.traj.process()  # compute averages, etc.
  print '...Done.'

  print 'Writing results...',
  sampler.traj.write_results(os.path.join(args.outdir,'traj_lambda%2.2f.yaml'%args.lam))
  print '...Done.'

  # pickle the sampler object
  print 'Pickling the sampler object ...', 
  outfilename = 'sampler_lambda%2.2f.pkl'%args.lam
  print outfilename,
  fout = open(os.path.join(args.outdir, outfilename), 'wb')
  # Pickle dictionary using protocol 0.
  cPickle.dump(sampler, fout)
  print '...Done.' 


