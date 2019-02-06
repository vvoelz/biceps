import sys, os 
import numpy as np
from matplotlib import pyplot as plt

a=np.load('traj_lambda1.00.npz')
b=a['arr_0'].item()['trajectory']
allowed_gamma=np.load('allowed_gamma.npy')
allowed_sigma=np.load('allowed_sigma.npy')
sampled_gamma=np.zeros(len(allowed_gamma))
sampled_sigma=np.zeros(len(allowed_sigma))

gamma_list=[]
sigma_list=[]

for i in range(len(b)):
    gamma_list.append(b[i][4][0][1])
    sigma_list.append(b[i][4][0][0])

for i in range(len(gamma_list)):
    sampled_gamma[gamma_list[i]]+=1
    sampled_sigma[sigma_list[i]]+=1


a0=np.load('traj_lambda0.00.npz')
b0=a0['arr_0'].item()['trajectory']
allowed_gamma=np.load('allowed_gamma.npy')
allowed_sigma=np.load('allowed_sigma.npy')
sampled_gamma0=np.zeros(len(allowed_gamma))
sampled_sigma0=np.zeros(len(allowed_sigma))

gamma_list0=[]
sigma_list0=[]

for i in range(len(b)):
    gamma_list0.append(b[i][4][0][1])
    sigma_list0.append(b[i][4][0][0])
    
for i in range(len(gamma_list)):
    sampled_gamma0[gamma_list[i]]+=1
    sampled_sigma0[sigma_list[i]]+=1

x=np.arange(0,len(gamma_list),1)
plt.figure(figsize=(10,20))
plt.subplot(2,1,1)
plt.plot(x,gamma_list0)
plt.xlabel('steps')
plt.ylabel('gamma')
plt.title('lambda_0.0')
plt.subplot(2,1,2)
plt.plot(x,gamma_list)
plt.xlabel('steps')
plt.ylabel('gamma')
plt.title('lambda_1.0')
plt.tight_layout()
plt.savefig('gamma_trace.pdf')


plt.figure(figsize=(10,20))
plt.subplot(2,1,1)
plt.plot(x,sigma_list0)
plt.xlabel('steps')
plt.ylabel('sigma')
plt.title('lambda_0.0')
plt.subplot(2,1,2)
plt.plot(x,sigma_list)
plt.xlabel('steps')
plt.ylabel('sigma')
plt.title('lambda_1.0')
plt.tight_layout()
plt.savefig('sigma_trace.pdf')

plt.figure(figsize=(10,20))
plt.subplot(2,1,1)
plt.step(allowed_gamma,sampled_gamma0)
plt.hold(True)
plt.step(allowed_gamma,sampled_gamma)
plt.legend(['exp', 'sim+exp'])
plt.xlabel('gamma')
plt.subplot(2,1,2)
plt.step(allowed_sigma,sampled_sigma0)
plt.hold(True)
plt.step(allowed_sigma,sampled_sigma)
plt.legend(['exp', 'sim+exp'])
plt.xlabel('sigma')
plt.tight_layout()
plt.savefig('distribution.pdf')



