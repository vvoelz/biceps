import sys, os
sys.path.append('new_src')
from toolbox import *

a=np.load('traj_lambda1.00.npz')['arr_0'].item()
rest = a['rest_type']
rest_type=[]
for i in rest:
    if i.split('_')[1] != 'noe':
        rest_type.append(i.split('_')[1])
    elif i.split('_')[1] == 'noe':
        rest_type.append(i.split('_')[1])
        rest_type.append('gamma')

T_new = a['trajectory'][::4000]
fold = 10
nsnaps = len(T_new)
dx = int(nsnaps/fold)
rounds = 100
all_JSD=[]
all_JSDs=[[] for i in range(fold)]
traj = 'traj_lambda1.00.npz'
for subset in range(fold):
    half = dx * (subset+1)/2
    T1 = T_new[:half]
    T2 = T_new[half:dx*(subset+1)]
    T_total = T_new[:dx*(subset+1)]
    all_JSD.append(compute_JSD(T1,T2,T_total,traj))
    for r in range(rounds):
        mT1 = np.random.choice(len(T_total),len(T_total)/2,replace=False)
        mT2 = np.delete(np.arange(0,len(T_total),1),mT1)
	temp_T1, temp_T2 = [],[]
	for snapshot in mT1:
		temp_T1.append(T_total[snapshot])
	for snapshot in mT2:
		temp_T2.append(T_total[sanpshot])
        all_JSDs[subset].append(compute_JSD(temp_T1,temp_T2,T_total,traj))

np.save('all_JSD.npy',all_JSD)
np.save('all_JSDs.npy',all_JSDs)
