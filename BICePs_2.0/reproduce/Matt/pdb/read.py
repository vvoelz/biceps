

with open('reweight.txt') as file:
    lines = file.readlines()

for i in range(len(lines)):
    lines[i] = lines[i].strip()
    lines[i] = lines[i].split()
    print lines[i]
    lines[i][0] = int(lines[i][0])
    lines[i][1] = float(lines[i][1])

print lines
