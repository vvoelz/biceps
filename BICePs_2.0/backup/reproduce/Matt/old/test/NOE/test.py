import sys, os
import numpy as np

for i in range(100):
    new=[]
    with open('%d.noe'%i) as f:
        lines=f.readlines()
    line=''.join(lines)
    fields = line.strip().split('\n')
    field=[]
    for j in range((len(fields))):
        field.append(fields[j].strip().split())
    for k in range(1,len(field)):
        new.append(float(field[k][-1]))
    np.savetxt('NOE/test%d.txt'%i,new)
