import os ,sys
filename='populations_ref_normal.dat'
with open(filename) as f:
        lines=f.readlines()
line=''.join(lines)
fields = line.strip().split('\n')
field=[]
for i in range((len(fields))):
        field.append(fields[i].strip().split())
print field[0][2]
