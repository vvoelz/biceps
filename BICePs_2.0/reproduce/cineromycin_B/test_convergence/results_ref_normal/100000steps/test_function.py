import sys, os
import numpy as np


def compute_JSD(T1,T2,T_total,traj):
    '''Compute JSD for a given part of trajectory.
    Parameters
    ----------
    rest_type: A list of experimental observables used in BICePs sampling
    T1, T2, T_total: part 1, part2 and total (part1 + part2)
    traj: trajectory from BICePs sampling
    '''
    a = np.load(traj)['arr_0'].item()
    rest = a['rest_type']
    rest_type=[]
    for i in rest:
        if i.split('_')[1] != 'noe':
            rest_type.append(i.split('_')[1])
        elif i.split('_')[1] == 'noe':
            rest_type.append(i.split('_')[1])
            rest_type.append('gamma')
    all_JSD = np.zeros(len(rest_type))
    if 'gamma' in rest_type:
        for i in range(len(rest_type)):
            if i == len(rest_type)-1:   # means it is gamma
                r1, r2, r_total, r_all = np.zeros(len(a['allowed_gamma'])),np.zeros(len(a['allowed_gamma'])),np.zeros(len(a['allowed_gamma'])),np.zeros(len(a['allowed_gamma']))
                for j in T1:
                    r1[j[4:][0][i-1][1]]+=1
                for j in T2:
                    r2[j[4:][0][i-1][1]]+=1
                for j in T_total:
                    r_total[j[4:][0][i-1][1]]+=1
                #for j in T_new:
                #    r_all[j[4:][0][i-1][1]]+=1
                N1=sum(r1)
                N2=sum(r2)
                N_total = sum(r_total)
                H1 = -1.*r1/N1*np.log(r1/N1)
                H1 = sum(np.nan_to_num(H1))
                H2 = -1.*r2/N2*np.log(r2/N2)
                H2 = sum(np.nan_to_num(H2))
                H = -1.*r_total/N_total*np.log(r_total/N_total)
                H = sum(np.nan_to_num(H))
                JSD = H-(N1/N_total)*H1-(N2/N_total)*H2
#                print 'JSD', JSD
                all_JSD[i] = JSD
    #            all_hist[i].append(r1)
    #            all_hist[i].append(r2)
    #            all_hist[i].append(r_total)
    #            all_hist[i].append(r_all)
    #        elif i == len(rest_type) - 2: # means it is sigma_noe
    #            r1, r2, r_total, r_all = np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i]))
    #            for j in T1:
    #                r1[j[4:][0][i][0]]+=1
    #            for j in T2:
    #                r2[j[4:][0][i][0]]+=1
    #            for j in T_total:
    #                r_total[j[4:][0][i][0]]+=1
    #            for j in T_new:
    #                r_all[j[4:][0][i][0]]+=1
    #            N1=sum(r1)
    #            N2=sum(r2)
    #            N_total = sum(r_total)
    #            H1 = -1.*r1/N1*np.log(r1/N1)
    #            H1 = sum(np.nan_to_num(H1))
    #            H2 = -1.*r2/N2*np.log(r2/N2)
    #            H2 = sum(np.nan_to_num(H2))
    #            H = -1.*r_total/N_total*np.log(r_total/N_total)
    #            H = sum(np.nan_to_num(H))
    #            JSD = H-(N1/N_total)*H1-(N2/N_total)*H2
    #            print 'JSD', JSD
    #            all_JSD[i] = JSD
    #            all_hist[i].append(r1)
    #            all_hist[i].append(r2)
    #            all_hist[i].append(r_total)
    #            all_hist[i].append(r_all)
            else:
                r1, r2, r_total, r_all = np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i]))
                for j in T1:
                    #print j
                    #sys.exit()
                    r1[j[4:][0][i][0]]+=1
                for j in T2:
                    r2[j[4:][0][i][0]]+=1
                for j in T_total:
                    r_total[j[4:][0][i][0]]+=1
    #            for j in T_new:
    #                r_all[j[4:][0][i][0]]+=1
                N1=sum(r1)
                N2=sum(r2)
                N_total = sum(r_total)
                H1 = -1.*r1/N1*np.log(r1/N1)
                H1 = sum(np.nan_to_num(H1))
                H2 = -1.*r2/N2*np.log(r2/N2)
                H2 = sum(np.nan_to_num(H2))
                H = -1.*r_total/N_total*np.log(r_total/N_total)
                H = sum(np.nan_to_num(H))
                JSD = H-(N1/N_total)*H1-(N2/N_total)*H2
    #            print 'JSD', JSD
                all_JSD[i] = JSD
    #            all_hist[i].append(r1)
    #            all_hist[i].append(r2)
    #            all_hist[i].append(r_total)
    #            all_hist[i].append(r_all)
    else:
        for i in range(len(rest_type)):
                r1, r2, r_total, r_all = np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i]))
                for j in T1:
                    r1[j[4:][0][i][0]]+=1
                for j in T2:
                    r2[j[4:][0][i][0]]+=1
                for j in T_total:
                    r_total[j[4:][0][i][0]]+=1
    #            for j in T_new:
    #                r_all[j[4:][0][i][0]]+=1
                N1=sum(r1)
                N2=sum(r2)
                N_total = sum(r_total)
                H1 = -1.*r1/N1*np.log(r1/N1)
                H1 = sum(np.nan_to_num(H1))
                H2 = -1.*r2/N2*np.log(r2/N2)
                H2 = sum(np.nan_to_num(H2))
                H = -1.*r_total/N_total*np.log(r_total/N_total)
                H = sum(np.nan_to_num(H))
                JSD = H-(N1/N_total)*H1-(N2/N_total)*H2
    #            print 'JSD', JSD
                all_JSD[i] = JSD
    #            all_hist[i].append(r1)
    #            all_hist[i].append(r2)
    #            all_hist[i].append(r_total)
    #            all_hist[i].append(r_all)
    return all_JSD

a=np.load('traj_lambda1.00.npz')['arr_0'].item()
rest = a['rest_type']
rest_type=[]
for i in rest:
    if i.split('_')[1] != 'noe':
        rest_type.append(i.split('_')[1])
    elif i.split('_')[1] == 'noe':
        rest_type.append(i.split('_')[1])
        rest_type.append('gamma')

#fold = range(4)
#T_all = a['trajectory']
#T_new = T_all[::1]
#nsnaps = len(T_new)
#dx=int(nsnaps/2)
#dx = int(nsnaps/len(fold))
#T1 = T_new[dx*fold[-2]:dx*fold[-1]]
#T2 = T_new[dx*fold[-1]:]
#T_total = T_new[dx*fold[-2]:]

fold = range(4)
nsnaps = len(a['trajectory'])
dx = int(nsnaps/len(fold))
T_all = a['trajectory']
T1 = a['trajectory'][dx*fold[-2]:dx*fold[-1]]
T2 = a['trajectory'][dx*fold[-1]:]
T_total = a['trajectory'][dx*fold[-2]:]


print compute_JSD(T1,T2,T_total,'traj_lambda1.00.npz')



