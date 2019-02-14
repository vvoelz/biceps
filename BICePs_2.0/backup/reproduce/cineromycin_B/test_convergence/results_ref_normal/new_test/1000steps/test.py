import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



colors=['red', 'blue','black','green']
labels=['50-75%','75-100%','50-100%','0-100%']
a=np.load('traj_lambda1.00.npz')['arr_0'].item()
rest = a['rest_type']
rest_type=[]
for i in rest:
    if i.split('_')[1] != 'noe':
        rest_type.append(i.split('_')[1])
    elif i.split('_')[1] == 'noe':
        rest_type.append(i.split('_')[1])
        rest_type.append('gamma')
n_rest = len(rest_type)
all_hist = [[] for i in range(n_rest)]
all_JSD = np.zeros(n_rest)
fold = range(100)
nsnaps = len(a['trajectory'])
dx = int(nsnaps/len(fold))
T_all = a['trajectory']
T1 = a['trajectory'][dx*fold[-2]:dx*fold[-1]]
T2 = a['trajectory'][dx*fold[-1]:]
T_total = a['trajectory'][dx*fold[-2]:]
mixing = True

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
            for j in T_all:
                r_all[j[4:][0][i-1][1]]+=1
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
            print 'JSD', JSD
            all_JSD[i] = JSD
            all_hist[i].append(r1)
            all_hist[i].append(r2)
            all_hist[i].append(r_total)
            all_hist[i].append(r_all)
        elif i == len(rest_type) - 2: # means it is sigma_noe
            r1, r2, r_total, r_all = np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i]))
            for j in T1:
                r1[j[4:][0][i][0]]+=1
            for j in T2:
                r2[j[4:][0][i][0]]+=1
            for j in T_total:
                r_total[j[4:][0][i][0]]+=1
            for j in T_all:
                r_all[j[4:][0][i][0]]+=1
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
            print 'JSD', JSD
            all_JSD[i] = JSD
            all_hist[i].append(r1)
            all_hist[i].append(r2)
            all_hist[i].append(r_total)
            all_hist[i].append(r_all)
        else:
            r1, r2, r_total, r_all = np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i]))
            for j in T1:
                r1[j[4:][0][i][0]]+=1
            for j in T2:
                r2[j[4:][0][i][0]]+=1
            for j in T_total: 
                r_total[j[4:][0][i][0]]+=1
            for j in T_all:
                r_all[j[4:][0][i][0]]+=1
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
            print 'JSD', JSD
            all_JSD[i] = JSD
            all_hist[i].append(r1)
            all_hist[i].append(r2)
            all_hist[i].append(r_total)
            all_hist[i].append(r_all)
else:
    for i in range(len(rest_type)):
            r1, r2, r_total, r_all = np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i]))
            for j in T1:
                r1[j[4:][0][i][0]]+=1
            for j in T2:
                r2[j[4:][0][i][0]]+=1
            for j in T_total:
                r_total[j[4:][0][i][0]]+=1
            for j in T_all:
                r_all[j[4:][0][i][0]]+=1
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
            print 'JSD', JSD
            all_JSD[i] = JSD
            all_hist[i].append(r1)
            all_hist[i].append(r2)
            all_hist[i].append(r_total)
            all_hist[i].append(r_all)

if mixing:
    all_JSD_m = [[] for i in range(n_rest)]
    all_hist_m = [[] for i in range(n_rest)]
    rounds = 1000
    for r in range(rounds):
        print r
        mT1 = np.random.choice(len(T_total),len(T_total)/2,replace=False)
        mT2 = np.delete(np.arange(0,len(T_total),1),mT1)
#        print (mT1), (mT2)
#        sys.exit()
        if 'gamma' in rest_type:
            for i in range(len(rest_type)):
                if i == len(rest_type)-1:   # means it is gamma
                    r1, r2, r_total = np.zeros(len(a['allowed_gamma'])),np.zeros(len(a['allowed_gamma'])),np.zeros(len(a['allowed_gamma']))
                    for j in mT1:
                        r1[T_total[j][4:][0][i-1][1]]+=1
                    for j in mT2:
                        r2[T_total[j][4:][0][i-1][1]]+=1
                    for j in T_total:
                        r_total[j[4:][0][i-1][1]]+=1
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
                    #print 'JSD', JSD
                    all_JSD_m[i].append(JSD)
                    all_hist_m[i].append(r1)
                    all_hist_m[i].append(r2)
                    all_hist_m[i].append(r_total)

                else:         # means it is sigma
                    r1, r2, r_total = np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i]))
                    for j in mT1:
                        r1[T_total[j][4:][0][i][0]]+=1
                    for j in mT2:
                        r2[T_total[j][4:][0][i][0]]+=1
                    for j in T_total:
                        r_total[j[4:][0][i][0]]+=1
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
                    #print 'JSD', JSD
                    all_JSD_m[i].append(JSD)
                    all_hist_m[i].append(r1)
                    all_hist_m[i].append(r2)
                    all_hist_m[i].append(r_total)
        else:         # means it is sigma
            r1, r2, r_total = np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i])),np.zeros(len(a['allowed_sigma'][i]))
            for j in mT1:
                r1[T_total[j][4:][0][i][0]]+=1
            for j in mT2:
                r2[T_total[j][4:][0][i][0]]+=1
            for j in T_total:
                r_total[j[4:][0][i][0]]+=1
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
            #print 'JSD', JSD
            all_JSD_m[i].append(JSD)
            all_hist_m[i].append(r1)
            all_hist_m[i].append(r2)
            all_hist_m[i].append(r_total)
    np.save('all_JSD_m.npy',all_JSD_m)
#    sys.exit()
    plt.figure(figsize=(10,5*n_rest))
    if 'gamma' in rest_type:
        for i in range(n_rest):
            if i == n_rest-1:  # means it is gamma
                plt.subplot(n_rest,1,i+1)
                counts,bins = np.histogram(all_JSD_m[i], bins=np.arange(0,max(all_JSD_m[i]),(max(all_JSD_m[i])-min(all_JSD_m[i]))/len(all_JSD_m[i])) )
                plt.plot(bins[0:-1],counts)
                plt.plot([all_JSD[i]],[0],'*', ms=12)
                plt.xlabel('$JSD_{\gamma}$')
            elif i == n_rest-2:  # means it is sigma_noe
                plt.subplot(n_rest,1,i+1)
                counts,bins = np.histogram(all_JSD_m[i], bins=np.arange(0,max(all_JSD_m[i]),(max(all_JSD_m[i])-min(all_JSD_m[i]))/len(all_JSD_m[i])) )
                plt.plot(bins[0:-1],counts)
                plt.plot([all_JSD[i]],[0],'*', ms=12)
                plt.xlabel('$JSD_{\sigma_{%s}}$'%rest_type[i])
            else:
                plt.subplot(n_rest,1,i+1)
                counts,bins = np.histogram(all_JSD_m[i], bins=np.arange(0,max(all_JSD_m[i]),(max(all_JSD_m[i])-min(all_JSD_m[i]))/len(all_JSD_m[i])) )
                plt.plot(bins[0:-1],counts)
                plt.plot([all_JSD[i]],[0],'*', ms=12)
                if rest_type[i].find('cs') == -1:   # means it is not cs_xxx
                    plt.xlabel('$JSD_{\sigma_{%s}}$'%rest_type[i])
                else:
                    plt.xlabel('$JSD_{\sigma_{{%s}_{%s}}}$'%(rest_type[i].split('_')[0],rest_type[i].split('_')[1]))
                
    else:
        for i in range(n_rest):
            plt.subplot(n_rest,1,i+1)
            counts,bins = np.histogram(all_JSD_m[i], bins=np.arange(0,max(all_JSD_m[i]),(max(all_JSD_m[i])-min(all_JSD_m[i]))/len(all_JSD_m[i])) )
            plt.plot(bins[0:-1],counts)
            plt.plot([all_JSD[i]],[0],'*', ms=12)
            if rest_type[i].find('cs') == -1:   # means it is not cs_xxx
                plt.xlabel('$JSD_{\sigma_{%s}}$'%rest_type[i])
            else:
                plt.xlabel('$JSD_{\sigma_{{%s}_{%s}}}$'%(rest_type[i].split('_')[0],rest_type[i].split('_')[1]))
    plt.tight_layout()
    plt.savefig('JSD_p_test.pdf')

sys.exit()

plt.figure(figsize=(10,5*n_rest))
if 'gamma' in rest_type:
    for i in range(n_rest):
        if i == n_rest-1:  # means it is gamma
            plt.subplot(n_rest,1,i+1)
            for j in range(len(all_hist[i])):
                plt.step(a['allowed_gamma'],all_hist[i][j],'-',color=colors[j],label=labels[j])
                plt.legend(loc='best')
                plt.xlabel('$\gamma$')
                plt.ylabel('$P(\gamma)$')
                xmax = [k for k, e in enumerate(all_hist[i][-1]) if e != 0.][-1]
                xmin = [k for k, e in enumerate(all_hist[i][-1]) if e != 0.][0]
                d_x = (max(a['allowed_gamma']) - min(a['allowed_gamma'])) / len(a['allowed_gamma'])
                plt.xlim(a['allowed_gamma'][xmin]-d_x,a['allowed_gamma'][xmax]+d_x)
                plt.yticks([])
        elif i == n_rest-2:  # means it is sigma_noe
            plt.subplot(n_rest,1,i+1)
            for j in range(len(all_hist[i])):
                plt.step(a['allowed_sigma'][i],all_hist[i][j],'-',color=colors[j],label=labels[j])
                plt.legend(loc='best')
                plt.xlabel('$\sigma_{%s}$'%rest_type[i])
                plt.ylabel('$P(\sigma_{%s})$'%rest_type[i])
                xmax = [k for k, e in enumerate(all_hist[i][-1]) if e != 0.][-1]
                xmin = [k for k, e in enumerate(all_hist[i][-1]) if e != 0.][0]
                d_x = (max(a['allowed_sigma'][i]) - min(a['allowed_sigma'][i])) / len(a['allowed_sigma'][i])
                plt.xlim(a['allowed_sigma'][i][xmin]-d_x, a['allowed_sigma'][i][xmax]+d_x)
                plt.yticks([])
        else:
            plt.subplot(n_rest,1,i+1)
            for j in range(len(all_hist[i])):
                plt.step(a['allowed_sigma'][i],all_hist[i][j],'-',color=colors[j],label=labels[j])
                plt.legend(loc='best')
                if rest_type[i].find('cs') == -1:   # means it is not cs_xxx
                    plt.xlabel('$\sigma_{%s}$'%rest_type[i])
                    plt.ylabel('$P(\sigma_{%s})$'%rest_type[i])
                else:
                    plt.xlabel('$\sigma_{{%s}_{%s}}$'%(rest_type[i].split('_')[0],rest_type[i].split('_')[1]))
                    plt.ylabel('$P(\sigma_{{%s}_{%s}})$'%(rest_type[i].split('_')[0],rest_type[i].split('_')[1]))
                xmax = [k for k, e in enumerate(all_hist[i][-1]) if e != 0.][-1]
                xmin = [k for k, e in enumerate(all_hist[i][-1]) if e != 0.][0]
                d_x = (max(a['allowed_sigma'][i]) - min(a['allowed_sigma'][i])) / len(a['allowed_sigma'][i])
                plt.xlim(a['allowed_sigma'][i][xmin]-d_x,a['allowed_sigma'][i][xmax]+d_x)
                plt.yticks([])
else:
    for i in range(n_rest):
            plt.subplot(n_rest,1,i+1)
            for j in range(len(all_hist[i])):
                plt.step(a['allowed_sigma'][i],all_hist[i][j],'-',color=colors[j],label=labels[j])
                plt.legend(loc='best')
                if rest_type[i].find('cs') == -1:   # means it is not cs_xxx
                    plt.xlabel('$\sigma_{%s}$'%rest_type[i])
                    plt.ylabel('$P(\sigma_{%s})$'%rest_type[i])
                else:
                    plt.xlabel('$\sigma_{{%s}_{%s}}$'%(rest_type[i].split('_')[0],rest_type[i].split('_')[1]))
                    plt.ylabel('$P(\sigma_{{%s}_{%s}})$'%(rest_type[i].split('_')[0],rest_type[i].split('_')[1]))
                xmax = [k for k, e in enumerate(all_hist[i][-1]) if e != 0.][-1]
                xmin = [k for k, e in enumerate(all_hist[i][-1]) if e != 0.][0]
                d_x = (max(a['allowed_sigma'][i]) - min(a['allowed_sigma'][i])) / len(a['allowed_sigma'][i])
                plt.xlim(a['allowed_sigma'][i][xmin]-d_x,a['allowed_sigma'][i][xmax]+d_x)
                plt.yticks([])
plt.tight_layout()
plt.savefig('convergence.pdf')

for i in range(len(rest_type)):
    print rest_type[i], 'JSD = ', all_JSD[i]
            

    
