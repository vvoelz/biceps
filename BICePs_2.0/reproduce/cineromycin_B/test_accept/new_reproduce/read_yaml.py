import sys, os
import yaml
import numpy as np

with open('cineromycinB_expdata_VAV.yaml','r') as f:
    l = yaml.load(f)
#print l
#k = []
#k=[float("{0:.1f}".format(n)) for n in l['JCoupling_Expt']]
print l
sys.exit()
#for i in l['JCoupling_Karplus']:
#    k.append(i)
k=l['NOE_PairIndex']
m=l['NOE_Expt']
n=l['NOE_Equivalent']
new=[]
a=1
#print len(k)
#print len(m)
#print len(n)
for i in range(len(n)):
    print len(n[i])
#        print i
#        break
sys.exit()
#sys.exit()
for i in range(len(k)):
    for j in range(len(n)):
        if k[i] not in n[j]:
            new.append([a,m[i]])
            a+=1
        else:
            p = j
            new.append([a,m[i]])
    
print new
#np.save('ind.npy',k)
#np.savetxt('atom_indice_noe.txt',k,fmt='%i')
#np.savetxt('noe_distance.txt',m,fmt='%.1f')
