import os, sys
a=[9,10,11,12,13,14,15,16,17,18,19,20]
b=[8,9,10,11,12,13,14,15,16,17,18,19]
for i in range(len(a)):
        os.system('mv RUN%d RUN%d'%(a[i],b[i]))
