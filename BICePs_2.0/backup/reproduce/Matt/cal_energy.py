import sys, os
import numpy as np

allowed_sigma=np.load('allowed_sigma.npy')
allowed_gamma=np.load('allowed_gamma.npy')
total_sse = np.load('sse.npy')
sigma=allowed_sigma[140]
gamma=allowed_gamma[164]
Ndof=18.0
f=0.0
#f=4.42972835777
ref=39.4180894778
#ref=38.443241448
sse = total_sse[0][164]
#sse=13.3215863605
logZ=0.0118472697254

E = Ndof*np.log(sigma) + sse/(2.*sigma**2.0) + (Ndof/2.)*np.log(2.*np.pi) - ref + f+ logZ

print E
