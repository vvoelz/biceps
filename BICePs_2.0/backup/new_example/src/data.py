import sys, os, glob
import numpy as np
import re

def sort_data(dataFiles):
    dir_list=[]
    if not os.path.exists(dataFiles):
                raise ValueError("data directory doesn't exist")
    if ',' in dataFiles:
        print 'Sorting out the data...\n'
        raw_dir = (dataFiles).split(',')
	for dirt in raw_dir:
		if dirt[-1] == '/':
			dir_list.append(dirt+'*')
		else:
			dir_list.append(dirt+'/*')
    else:
	raw_dir = dataFiles
	if raw_dir[-1] == '/':
	        dir_list.append(dataFiles+'*')
	else:
		dir_list.append(dataFiles+'/*')
#    print 'dir_list', dir_list

    data = [[] for x in xrange(7)] # list for every extension; 7 possible experimental observables supported
    # Sorting the data by extension into lists. Various directories is not an issue...
    for i in range(0,len(dir_list)):
        convert = lambda txt: int(txt) if txt.isdigit() else txt
        # This convert / sorted glob is a bit fishy... needs many tests
        for j in sorted(glob.glob(dir_list[i]),key=lambda x: [convert(s) for s in re.split("([0-9]+)",x)]):
            if j.endswith('.noe'):
                data[0].append(j)
            elif j.endswith('.J'):
                data[1].append(j)
            elif j.endswith('.cs_H'):
                data[2].append(j)
            elif j.endswith('.cs_Ha'):
                data[3].append(j)
            elif j.endswith('.cs_N'):
                data[4].append(j)
            elif j.endswith('.cs_CA'):
                data[5].append(j)
            elif j.endswith('.pf'):
                data[6].append(j)
            else:
                raise ValueError("Incompatible File extension. Use:{.noe,.J,.cs_H,.cs_Ha}")
    data = np.array(filter(None, data)) # removing any empty lists
    Data = np.stack(data, axis=-1)
    data = Data.tolist()
    return data

