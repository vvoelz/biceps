import sys, os
import numpy as np
from datetime import datetime
startTime = datetime.now()
def compute_PF(beta_c, beta_h, beta_0, Nc, Nh):
	return beta_c * Nc + beta_h * Nh + beta_0  #GYH: new equation for PF calculation 03/2017 

allowed_xcs=np.arange(5.0,6.0,0.5)
allowed_xhs=np.arange(2.0,2.1,0.1)
allowed_bs=np.arange(3.0,5.0,1.0)
allowed_beta_c=np.arange(0.02,0.03,0.005)
allowed_beta_h=np.arange(0.00,0.10,0.05)
allowed_beta_0=np.arange(0.0,0.4,0.2)
if (1):
    Ncs=np.zeros((len(allowed_xcs),len(allowed_bs),107))
    Nhs=np.zeros((len(allowed_xhs),len(allowed_bs),107))
    for o in range(len(allowed_xcs)):
        for p in range(len(allowed_xhs)):
                for q in range(len(allowed_bs)):
                        infile_Nc='input/Nc/Nc_x%0.1f_b%d_state016.npy'%(allowed_xcs[o], allowed_bs[q])
                        infile_Nh='input/Nh/Nh_x%0.1f_b%d_state016.npy'%(allowed_xhs[p], allowed_bs[q])
                        Ncs[o,q,:] = (np.load(infile_Nc))
                        Nhs[p,q,:] = (np.load(infile_Nh))

model_protectionfactor = np.zeros((len(allowed_beta_c), len(allowed_beta_h), len(allowed_beta_0), len(allowed_xcs), len(allowed_xhs), len(allowed_bs)))
print model_protectionfactor.shape
#sys.exit()
sse=0.
exp=np.load('exp_data.npy')
for i in range(107):
	for x in range(len(allowed_xcs)):
        	for y in range(len(allowed_xhs)):
			for z in range(len(allowed_bs)):
                        	for m in range(len(allowed_beta_c)):
                                	for j in range(len(allowed_beta_h)):
                                        	for k in range(len(allowed_beta_0)):
#							print 'allowed_beta_c[m]', allowed_beta_c[m], 'allowed_beta_h[j]', allowed_beta_h[j], 'allowed_beta_0[k]', allowed_beta_0[k], 'Ncs[x,z,i]', Ncs[x,z,i], 'Nhs[y,z,i]', Nhs[y,z,i]
                                                	model_protectionfactor[m,j,k,x,y,z] = compute_PF(allowed_beta_c[m], allowed_beta_h[j], allowed_beta_0[k], Ncs[x,z,i], Nhs[y,z,i]) # GYH: will be modified with final file format 03/2017
#							print 'model_protectionfactor[',m,j,k,x,y,z,']', model_protectionfactor[m,j,k,x,y,z]
#							sys.exit()
#	print i, model_protectionfactor[1,1,1,1,1,1]
	err = model_protectionfactor[1,1,1,1,1,1] - exp[i]
	sse += err**2.0
print sse	
print datetime.now() - startTime
