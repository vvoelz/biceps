import pynmrstar as pystr
import numpy as np
import pandas as pd
pd.options.display.max_rows = None
pd.options.display.max_columns = None


def star_to_df(ID, category="Atom_chem_shift"):
    """Converts a BMRB-star file to a pandas DataFrame.
    Notes: https://pynmrstar.readthedocs.io/en/latest/index.html

    Args:
        ID(str or int) - (BMRB-Star file) or (BMRB database ID)

    Returns:
        df(pd.DataFrame)
    """

    if type(ID) == int:   entry = pystr.Entry.from_database(ID) # parse from database
    elif type(ID) == str: entry = pystr.Entry.from_file(ID) # parse from file
    else:                 return TypeError

    #help(entry)
    #print(entry)
    #print(entry.__dict__)
    #print(entry.print_tree())
    #print(entry.get_loops_by_category("coupling_constants_set_1"))
    #print(entry.get_loops_by_category("Coupling_constant")[0])
    #exit()
    all_data = entry.get_loops_by_category(category)#[0]
    result = []
    for i,data in enumerate(all_data):
        data_dict = data.__dict__
        tags = data_dict["_tags"]
        _data = []
        for tag in tags: _data.append(np.array(data[tag]))
        array = np.array(_data)
        df = pd.DataFrame(array.transpose(), columns=tags)
        result.append(df)
    if len(result) == 1: return result[0]
    if len(result) > 1: return result



if __name__ == "__main__":

    df = star_to_df(ID=20009, category="Atom_chem_shift") # 2RVD
    df.to_csv("2RVD_cs_data.csv")
    df = star_to_df(ID=5694, category="Atom_chem_shift")  # 1UAO
    df.to_csv("1UAO_cs_data.csv")
    df = star_to_df(ID=5694, category="Coupling_constant")  # 1UAO
    df.to_csv("1UAO_J_data.csv")

    #print(df)





# NOTE: save for future use?
#IDs = np.array(cs_data['ID'])
#atomIDs = np.array(cs_data['Atom_ID'])
#resIDs = np.array(cs_data['Seq_ID'])
#exp = np.array(cs_data['val'])
#model = np.array(np.zeros(len(exp))) # TODO: this is just temp


#columns = ["index", "resID1", 'atom_index1','atom_ID1', 'exp', 'model']
#array = np.array([IDs, resIDs, atomIDs, atomIDs, exp, model])
#print(array.shape)

