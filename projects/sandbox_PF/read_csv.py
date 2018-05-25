import sys, os
import numpy as np
import csv
#with open ('test.csv','rb') as csvfile:
#	lines=csv.reader(csvfile, delimiter=' ', quotechar='|')
#	for row in lines:
#		print row
filename='test.csv'
with open(filename) as f:
        lines=f.readlines()
line=''.join(lines)
fields = line.strip().split('\n')
#print fields
#sys.exit()
field=[]
for i in range((len(fields))):
        field.append(fields[i].strip().split(','))
#print len(field[1])
PF=[]
#print field[2][1]
#sys.exit()
for j in range(len(field)):
	if len(field[j]) == 3 and field[j][2] != '':
#		print j
		PF.append(float(field[j][2]))
#np.savetxt('PF.txt',PF)
#print PF
#print len(PF)
ind=np.arange(0,107,1)
exp=[]
for k in PF:
	exp.append(np.log(k))
#print exp
#print zip(ind,exp)
with open('exp_data.txt', 'w') as f:
	writer=csv.writer(f,delimiter='\t')
	writer.writerows(zip(ind,exp))
quit()

