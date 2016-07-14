import os ,sys
filename='Karplus.txt'
with open(filename) as f:
        lines=f.readlines()
line=''.join(lines)
fields = line.strip().split('\n')
print fields[1]
sys.exit()
field=[]
for i in range((len(fields))):
        field.append(fields[i].strip().split())
#print field
#sys.exit()
#print field[0][5]
ind=[]
for i in range(len(field)):
        resid=raw_input("Residue number (1-based index): ")
        atom_name=raw_input("Atom name (capital form): ")
        if resid != "none":
                for i in range(len(field)):
                        if (field[i][5] == resid) and (field[i][2]== atom_name):
                                ind.append(int(field[i][1])-1)
        else:
                print ind

