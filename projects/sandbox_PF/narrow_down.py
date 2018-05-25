import sys, os
import numpy as np
from datetime import datetime
startTime = datetime.now()
def compute_PF(beta_c, beta_h, beta_0, Nc, Nh):
	return beta_c * Nc + beta_h * Nh + beta_0  #GYH: new equation for PF calculation 03/2017 

allowed_xcs=np.arange(5.0,8.5,0.5)
allowed_xhs=np.arange(2.0,2.8,0.1)
allowed_bs=np.arange(3.0,21.0,1.0)
allowed_beta_c=np.arange(0.02,0.095,0.005)
allowed_beta_h=np.arange(0.00,2.00,0.05)
allowed_beta_0=np.arange(-3.0,1.0,0.2)
bad_bc=[]
bad_bh=[]
bad_b0=[]
bad_xc=[]
bad_xh=[]
bad_b=[]
#if (1):
for i in range(1):
	Ncs=np.zeros((len(allowed_xcs),len(allowed_bs),107))
	Nhs=np.zeros((len(allowed_xhs),len(allowed_bs),107))
    	for o in range(len(allowed_xcs)):
        	for p in range(len(allowed_xhs)):
                	for q in range(len(allowed_bs)):
                        	infile_Nc='input/Nc/Nc_x%0.1f_b%d_state%03d.npy'%(allowed_xcs[o], allowed_bs[q],i)
                        	infile_Nh='input/Nh/Nh_x%0.1f_b%d_state%03d.npy'%(allowed_xhs[p], allowed_bs[q],i)
                        	Ncs[o,q,:] = (np.load(infile_Nc))
                        	Nhs[p,q,:] = (np.load(infile_Nh))
#print Ncs[0,0,0]
#print Nhs[0,0,0]
	model_protectionfactor = np.zeros((len(allowed_beta_c), len(allowed_beta_h), len(allowed_beta_0), len(allowed_xcs), len(allowed_xhs), len(allowed_bs)))
#print model_protectionfactor.shape
#sys.exit()
#	for n in range(107):
	for x in range(len(allowed_xcs)):
        	for y in range(len(allowed_xhs)):
			for z in range(len(allowed_bs)):
                        	for m in range(len(allowed_beta_c)):
                                	for j in range(len(allowed_beta_h)):
                                        	for k in range(len(allowed_beta_0)):
							for n in range(107):
#							print 'allowed_beta_c[m]', allowed_beta_c[m], 'allowed_beta_h[j]', allowed_beta_h[j], 'allowed_beta_0[k]', allowed_beta_0[k], 'Ncs[x,z,i]', Ncs[x,z,i], 'Nhs[y,z,i]', Nhs[y,z,i]
                                                		model_protectionfactor[m,j,k,x,y,z] = compute_PF(allowed_beta_c[m], allowed_beta_h[j], allowed_beta_0[k], Ncs[x,z,n], Nhs[y,z,n])
#							print 'model_protectionfactor[',m,j,k,x,y,z,']', model_protectionfactor[m,j,k,x,y,z]
								if model_protectionfactor[m,j,k,x,y,z] < 0.0:
									bad_bc.append(allowed_beta_c[m])
									bad_bh.append(allowed_beta_h[j])
									bad_b0.append(allowed_beta_0[k])
									bad_xc.append(allowed_xcs[x])
									bad_xh.append(allowed_xhs[y])
									bad_b.append(allowed_bs[z])
									print 'state', i, 'model_protectionfactor[',m,j,k,x,y,z,']', model_protectionfactor[m,j,k,x,y,z]	
									print 'beta_c', allowed_beta_c[m], 'beta_h', allowed_beta_h[j], 'beta_0', allowed_beta_0[k], 'xcs', allowed_xcs[x], 'xhs', allowed_xhs[y], 'bs', allowed_bs[z]	
									break
print 'bad_bc', set(bad_bc)
print 'bad_bh', set(bad_bh)
print 'bad_b0', set(bad_b0)
print 'bad_xc', set(bad_xc)
print 'bad_xh', set(bad_xh)
print 'bad_b',  set(bad_b)
print datetime.now() - startTime
