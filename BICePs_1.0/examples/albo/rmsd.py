import os, sys
import numpy as np
from matplotlib import pyplot as plt
import mdtraj as md
T = 300.0

import os ,sys
filename='populations_ref_normal.dat'
with open(filename) as f:
        lines=f.readlines()
line=''.join(lines)
fields = line.strip().split('\n')
field=[]
for i in range((len(fields))):
        field.append(fields[i].strip().split())



t=md.load('albo-xtal-single1.fixedname.pdb')
t1=md.load('albo-xtal-single2-converted.fixedname.pdb')
Ind = t.topology.select("backbone or name == 'CB'")
#t2=t.atom_slice(Ind)
#t4=t1.atom_slice(Ind)
c=[]
d=[]
for i in range(100):
	t3=md.load('pdbs_guangfeng/%d.pdb'%i)
	r1=md.rmsd(t,t3)
	r2=md.rmsd(t1,t3)
	if r1 > r2:
		d.append(float(field[i][2]))
	else:
		c.append(float(field[i][2]))
print sum(c)
print sum(d)
sys.exit()
	
#t3=md.load('pdbs_guangfeng/38.pdb')
#t5=t3.atom_slice(Ind)
#t4=t3.atom_slice(Ind)
#t5=md.load('turn.pdb')
#len(t)

r1=md.rmsd(t,t3)
r2=md.rmsd(t1,t3)
#r1=md.rmsd(t4,t5)
print r1
print r2
#np.savetxt('rmsd_23.txt',r)
#sys.exit(1)
sys.exit()
print '#frame\tRMSD'
for i in np.arange(0,t1.xyz.shape[0],1):
    print i, r[i]
#sys.exit(1)

#time = np.arange(0,0.01*200001,0.01)
#print time
#try:
#plt.figure()
#plt.plot(time, r)
#plt.xlabel('time (ns)')
#plt.ylabel('rmsd (nm)')
#plt.show()
#except:
   # pass

#sys.exit(1)
np.histogram(r, bins=np.arange(0,1.01,0.01) )
counts,bins = np.histogram(r, bins=np.arange(0,1.01,0.01) )
np.histogram(r1,bins=np.arange(0,1.01,0.01))
counts1,bins=np.histogram(r1,bins=np.arange(0,1.01,0.01))
x=np.arange(0,1.01,0.01)
plt.figure(figsize=(8,8))
bincenters = (bins[0:-1]+bins[1:])/2.0
axes=plt.gca()
axes.axes.get_yaxis().set_ticks([])
plt.ylim([0,80000])
plt.plot(bincenters, counts,linewidth=3,label='All-residue')
plt.plot(bincenters,counts1,color='red',linewidth=3,label='Turn part residues')
#plt.title('RMSD Histogram of TrpLoop2a')
plt.xlabel('RMSD (nm)',fontsize=24,fontweight='bold')
plt.xticks(fontsize=24)
plt.legend(loc='upper right',fontsize=24)
plt.savefig("RMSD_96_4.pdf")
#print np.bincount(x)
#sys.exit()
plt.show()
sys.exit()
kT = 0.001987*T
f = -kT*np.log(counts)
f -= f.min()
plt.figure(figsize=(8,8))
plt.plot(bincenters, f)
plt.xlabel('rmsd (nm)')
plt.ylabel('free energy (kcal/mol)')
plt.savefig("FE_300K.pdf")
#plt.show()
