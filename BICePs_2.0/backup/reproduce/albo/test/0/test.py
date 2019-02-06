import sys, os
import numpy as np


allowed_gamma=np.arange(0.1,0.4,0.1)
allowed_sigma_noe=np.arange(1.05,1.08,0.01)
allowed_sigma_J=np.arange(1.00,1.03,0.01)
d0=[0.425,0.4263,0.3115]
d1=[0.5252,0.3939,0.3889]
j0=[11.5993,11.5908,3.7798,3.0653]
j1=[11.5936,11.5880,11.5994,10.8603]
exp_j=[15.5,16.1,6.1,6.3]
exp_d=[0.38,0.38,0.38]
w_d=[1,0.5,0.5]
w_j=[1,1/3.,1/3.,1/3.]
print allowed_gamma
print allowed_sigma_noe
print allowed_sigma_J
sse_j0=[]
sse_j1=[]
sse_d0=[[] for i in range(len(allowed_gamma))]
sse_d1=[[] for i in range(len(allowed_gamma))]
for i in range(len(j0)):
    sse_j0.append((j0[i]-exp_j[i])**2.0*w_j[i])
    sse_j1.append((j1[i]-exp_j[i])**2.0*w_j[i])

for i in range(len(allowed_gamma)):
    for j in range(len(d0)):
        sse_d0[i].append((d0[j]-exp_d[j]*allowed_gamma[i])**2.0*w_d[j])
        sse_d1[i].append((d1[j]-exp_d[j]*allowed_gamma[i])**2.0*w_d[j])
print 'sse_j0',sum(sse_j0)
print 'sse_j1',sum(sse_j1)
print 'sse_d0[0]',sum(sse_d0[0])
print 'sse_d0[1]',sum(sse_d0[1])
print 'sse_d0[2]',sum(sse_d0[2])
print 'sse_d0[3]',sum(sse_d0[3])
print 'sse_d1[0]',sum(sse_d1[0])
print 'sse_d1[1]',sum(sse_d1[1])
print 'sse_d1[2]',sum(sse_d1[2])
print 'sse_d1[3]',sum(sse_d1[3])



