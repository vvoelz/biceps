import sys, os
from toolbox_temp import *

traj = 'traj_lambda1.00.npz' 
#print traj.endswith('npz')
#sys.exit()
rest_type = get_rest_type(traj)
#print rest_type
allowed_parameters = get_allowed_parameters(traj,rest_type=rest_type)
#print allowed_parameters[0]
ac = compute_ac(traj,1,rest_type=rest_type,allowed_parameters=allowed_parameters)
#print ac
parameters_counts = []
for i in range(len(allowed_parameters)):
	parameters_counts.append(np.zeros(len(allowed_parameters[i])))
#print parameters_counts
#sys.exit()
a=np.load(traj)['arr_0'].item()
T_new = a['trajectory'][::2]
fold = 10
nsnaps = len(T_new)
dx = int(nsnaps/fold)
rounds = 2
all_JSD=[]
all_JSDs=[[] for i in range(fold)]
traj = 'traj_lambda1.00.npz'
for subset in range(fold):
    half = dx * (subset+1)/2
    T1 = T_new[:half]
    T2 = T_new[half:dx*(subset+1)]
    T_total = T_new[:dx*(subset+1)]
    all_JSD.append(compute_JSD(T1,T2,T_total,rest_type,allowed_parameters,parameters_counts))
    for r in range(rounds):
        mT1 = np.random.choice(len(T_total),len(T_total)/2,replace=False)
        mT2 = np.delete(np.arange(0,len(T_total),1),mT1)
        temp_T1, temp_T2 = [],[]
        for snapshot in mT1:
                temp_T1.append(T_total[snapshot])
        for snapshot in mT2:
                temp_T2.append(T_total[snapshot])
        all_JSDs[subset].append(compute_JSD(temp_T1,temp_T2,T_total,rest_type,allowed_parameters,parameters_counts))
all_JSD=np.array(all_JSD)
all_JSDs=np.array(all_JSDs)
#np.save('all_JSD2.npy',all_JSD)
#np.save('all_JSDs2.npy',all_JSDs)
plot_conv(all_JSD,all_JSDs,rest_type)
