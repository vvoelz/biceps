# Residual dipolar couplings: determination of the molecular alignment tensor magnitude of oriented proteins

import pandas as pd
import mdtraj as md
import numpy as np
import uncertainties as u
import scipy.optimize
#from sklearn import metrics
from scipy.constants import physical_constants, mu_0, hbar

import matplotlib.pyplot as plt

# Simple Plot:{{{

def simple_plot(x,y,xlabel='x',ylabel='y',name=None,size=111,Type='scatter',
        color=False,fig_size=(12,10),invert_x_axis=False,fit=False,order=None,
        xLine=None,yLine=None,
        annotate_text=None,text_x=0,text_y=0,
        annotate_x=0,annotate_y=0,
        arrow='->', plot_ref_line=True):
    '''
    Returns a plot and saves it to the working directory
    unless stated otherwise.
    x = numpy array
    y = numpy array
    xlabel,ylabel,name = strings
    size = axis size
    color = color of line
    fig_size = (x,y)
    '''
    marks = ['o','D','2','>','*',',',"4","8","s",
             "p","P","*","h","H","+","x","X","D","d"]
    colors = ['k','b','g','r','c','m','y',
              'k','b','g','r','c','m','y']

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(size)

    if plot_ref_line:
        _min = np.min([x.min(), y.min()])
        _max = np.max([x.max(), y.max()])
        ax.plot([_min, _max], [_min, _max], 'k-', label="_nolegend_")

    if Type=='scatter':
        if color==False:
            ax.scatter(x,y,color='k')
        else:
            ax.scatter(x,y,color=color)

    if Type=='line':
        if color==False:
            ax.plot(x,y,'k')
        else:
            ax.plot(x,y,color)
    if fit==True:
        #ax.plot(x,y,label="_nolegend_")
        z = np.polyfit(x, y, order)
        n_coeff = len(z)
        #################################################
        p = np.poly1d(z)
        ax.plot(x,p(x),"r-", label="_nolegend_")
        # the line equation:
        #print('LINEST data:')
        #if order==1:
        #    print("y=%.6f*x+(%.6f)"%(z[0],z[1]))
        #    print(scipy.stats.linregress(x,y))
        #elif order==2:
        #    print("y=%.6f*x**2.+(%.6f)*x+(%.6f)"%(z[0],z[1],z[2]))
        #elif order==3:
        #    print("y=%.6f*x**3.+(%.6f)*x**2.+(%.6f)*x+(%.6f)"%(z[0],z[1],z[2],z[3]))
        #elif order==4:
        #    print("y=%.6fx**4.+(%.6f)*x**3.+(%.6f)*x**2.\
        #            +(%.6f)*x+(%.6f)"%(z[0],z[1],z[2],z[3],z[4]))
        #else:
        #    print('You need to add a greater order of polynomials to the script')

        #print(scipy.stats.chi2(y))
        #print(scipy.stats.ttest_ind(x, y, axis=0, equal_var=True))
        #eq = "y=%.6fx+(%.6f)"%(z[0],z[1])

    ax.set_xlabel('%s'%xlabel, fontsize=16)
    ax.set_ylabel('%s'%ylabel, fontsize=16)
    # Does the x-axis need to be reverse?
    if invert_x_axis==True:
        plt.gca().invert_xaxis()
    # Setting the ticks and tick marks
    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks()]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels()]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    if annotate_text != None:
        ax.annotate(r'%s'%annotate_text,
                xy=(annotate_x,annotate_y),xytext=(text_x,text_y), color="red",
                #arrowprops=dict(facecolor='black', arrowstyle=arrow),
                fontsize=16)
    if xLine!=None:
        ax.axhline(xLine)
    if yLine!=None:
        ax.axhline(yLine)
    fig.tight_layout()
    if name==None:
        pass
    else:
        fig.savefig('%s'%name)
    return ax
    #fig.show()


# }}}

# helper functions:{{{
def is_symmetric_and_traceless(tensor):
    # Check if the matrix is symmetric
    if not np.allclose(tensor, tensor.T):
        return False, "Tensor is not symmetric"
    # Check if the matrix is traceless
    if not np.isclose(np.trace(tensor), 0):
        return False, "Tensor is not traceless"
    return True, "Tensor is symmetric and traceless"

def diagonalize_tensor(tensor):
    """
    Diagonalizes the given tensor (assumed to be 3x3 and symmetric).

    :param tensor: A 3x3 symmetric matrix representing the alignment tensor.
    :return: Diagonalized tensor, eigenvalues, and eigenvectors.
    """
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    # Diagonalize the tensor
    diagonal_tensor = np.diag(eigenvalues)
    # The eigenvectors are in the columns of 'eigenvectors' matrix
    # 'diagonal_tensor' is the tensor expressed in the principal axis system
    return diagonal_tensor, eigenvalues, eigenvectors


def get_exp_RDC_file(file, k):
    df = star_to_df(ID=file, category="_RDC_constraint") # 2MOR
    list_cols = ['ID',
           'Comp_index_ID_1', 'Seq_ID_1', 'Comp_ID_1', 'Atom_ID_1',
           'Comp_index_ID_2', 'Seq_ID_2','Comp_ID_2', 'Atom_ID_2',
           'RDC_val', 'RDC_lower_bound', 'RDC_upper_bound',
           'RDC_val_err',
           'PDB_strand_ID_1', 'PDB_residue_no_1',
           'PDB_residue_name_1', 'PDB_atom_name_1',
           'PDB_strand_ID_2','PDB_residue_no_2',
           'PDB_residue_name_2', 'PDB_atom_name_2',
           'Auth_atom_ID_1',
           'Auth_alt_ID_2']

#    for i in range(len(df)):
#        print(i, df[i][['PDB_atom_name_2','Auth_atom_ID_1']].to_numpy()[0])
#    exit()

    #k = 2
    results = []
    for i in range(k, k+1):
        #print(df[i][['PDB_atom_name_2','Auth_atom_ID_1']].to_numpy()[0])
        _df = df[i][list_cols]
        exp = _df['RDC_val'].to_numpy()
        #print(_df[['PDB_residue_name_2', 'Seq_ID_2', 'PDB_atom_name_2','Auth_atom_ID_1', 'RDC_val']])
        # residue number1, chainid1, atom name1, residue number2, chainid2, atom name2, RDC EXP, rdc pred
        columns = ["resIdx1","chain1","atom1","resIdx2","chain2","atom2","exp","pred"]
        result = []
        for row in range(len(exp)):
            _result = {}
            _result["resIdx1"] = _df['PDB_residue_no_1'].to_numpy()[row]
            #print(_result["resIdx1"])
            #_result["res1"] = _df['PDB_residue_name_1'].to_numpy()[row]
            _result["chain1"] = _df['PDB_strand_ID_1'].to_numpy()[row]
            _result["atom1"] = _df['PDB_atom_name_1'].to_numpy()[row]
            _result["resIdx2"] = _df['PDB_residue_no_2'].to_numpy()[row]
            #print(_result["atom1"])
            #_result["res2"] = _df['PDB_residue_name_2'].to_numpy()[row]
            _result["chain2"] = _df['PDB_strand_ID_2'].to_numpy()[row]
            _result["atom2"] = _df['PDB_atom_name_2'].to_numpy()[row]
            _result["exp"] = exp[row]
            _result["pred"] = 1.0
            result.append(_result)
        result_pair = pd.DataFrame(result)
    val1,val2 = df[i][['PDB_atom_name_2','Auth_atom_ID_1']].to_numpy()[0]
    #data_file = f"ubq_pati_input_{val1}_{val2}.dat"
    #result_pair.to_csv(data_file, index=False, sep=' ', header=False)
    return result_pair

# }}}

class RDC_predictor:
    # See this paper for details:
    # https://web.math.princeton.edu/~amits/publications/saupe.pdf
    # They theory is very similar to "1.3 Previous approach"

    def __init__(self, filename):
        self.filename = filename
        self.traj = md.load(filename)
        non_water_atoms = self.traj.topology.select('not water')
        self.traj = self.traj.atom_slice(non_water_atoms)

        self.gyromagnetic_ratio_constants = {
            'H': 267.52218744*10**6,  # rad s−1 T−1 for 1H
            'H_in_H2O': 267.5153151*10**6,  # rad s−1 T−1 for 1H in H2O
            'D': 41.065*10**6,  # rad s−1 T−1 for 2H
            'T': 285.3508*10**6,  # rad s−1 T−1 for 3H
            'He': -203.7894569*10**6,  # For 3He
            'Li': 103.962*10**6,  # For 7Li
            'C': 67.2828*10**6,  # For 13C
            'N': 19.331*10**6,  # For 14N
            'N15': -27.116*10**6,  # For 15N
            'O': -36.264*10**6,  # For 17O
            'F': 251.815*10**6,  # For 19F
            'Na': 70.761*10**6,  # For 23Na
            'Al': 69.763*10**6,  # For 27Al
            'Si': -53.190*10**6,  # For 29Si
            'P': 108.291*10**6,  # For 31P
            'Fe': 8.681*10**6,  # For 57Fe
            'Cu': 71.118*10**6,  # For 63Cu
            'Zn': 16.767*10**6,  # For 67Zn
            'Xe': -73.997*10**6,  # For 129Xe
            }
        self.gyromagnetic_ratios = self.calculate_gyromagnetic_ratios()


    def calculate_gyromagnetic_ratios(self):
        """Calculates the gyromagnetic ratio for each atom in the structure."""
        gyromagnetic_ratios = []
        for atom in self.traj.topology.atoms:
            # Use the element symbol to get the gyromagnetic ratio
            ratio = self.gyromagnetic_ratio_constants.get(atom.element.symbol, None)
            gyromagnetic_ratios.append(ratio)
        return gyromagnetic_ratios

    def get_gyromagnetic_ratio(self, atom_index):
        """Returns the gyromagnetic ratio for a specific atom."""
        return self.gyromagnetic_ratios[atom_index]


    def calculate_distances_in_meters(self, atom_index_1, atom_index_2):
        """Calculates the distance between two atoms in the first frame."""
        atom1_xyz = self.traj.xyz[0, atom_index_1]
        atom2_xyz = self.traj.xyz[0, atom_index_2]
        distance = np.linalg.norm(atom1_xyz - atom2_xyz)*1e-9 # in meters
        return distance

    def get_atom_indices(self, df):
        #print(df)
        indices = []
        topology = self.traj.topology
        table = topology.to_dataframe()[0]
        #print(table)
        #exit()
        #"resIdx1","atom1","resIdx2","atom2"
        for k in range(len(df["resIdx1"].to_numpy())):
            try:
                atom1,atom2 = df["atom1"].to_numpy()[k],df["atom2"].to_numpy()[k]
                #print(atom1,atom2)
                resIdx1,resIdx2 = df["resIdx1"].to_numpy()[k],df["resIdx2"].to_numpy()[k]
                #print(resIdx1,resIdx2)
                group1 = table.iloc[np.where(int(resIdx1) == table["resSeq"].to_numpy())[0]]
                group2 = table.iloc[np.where(int(resIdx2) == table["resSeq"].to_numpy())[0]]
                #print(k, group1, group2)
                row1 = group1.iloc[np.where(atom1 == group1["name"].to_numpy())[0]]
                row2 = group2.iloc[np.where(atom2 == group2["name"].to_numpy())[0]]
                #print(k, row1, row2)
                indices.append([row1["serial"].values[0]-1,row2["serial"].values[0]-1])
            except(Exception) as e:
                print(e)
        indices = np.array(indices)
        self.atom_indices = indices
        return indices

    def get_bond_vector_info(self):
        """Append bond_vector information by adding cosine components for each bond vector.
        For each bond vector $ \mathbf{v} = (v_x, v_y, v_z) $, the cosine components are calculated by normalizing the vector:
        $$ \cos(\theta_x) = \frac{v_x}{\|\mathbf{v}\|}, \quad \cos(\theta_y) = \frac{v_y}{\|\mathbf{v}\|}, \quad \cos(\theta_z) = \frac{v_z}{\|\mathbf{v}\|} $$
        where $ \|\mathbf{v}\| = \sqrt{v_x^2 + v_y^2 + v_z^2} $ is the norm of the vector.
        """

        A_indices, B_indices = np.array(self.atom_indices).T

        # Only keep the pairs where the C index directly follows the CA index
        A_B_pairs = [(A, B) for A, B in zip(A_indices, B_indices)]
        self.internuclear_distances = []
        self.gyromagnetic_ratio_pairs = []
        for i, (A, B) in enumerate(A_B_pairs):
            self.internuclear_distances.append(self.calculate_distances_in_meters(A, B))
            A_gyro_ratio = self.get_gyromagnetic_ratio(A)
            B_gyro_ratio = self.get_gyromagnetic_ratio(B)
            self.gyromagnetic_ratio_pairs.append((A_gyro_ratio, B_gyro_ratio))

        bond_vector_info = []
        for i, (A, B) in enumerate(A_B_pairs):
            bond_vector = self.traj.xyz[0, B] - self.traj.xyz[0, A]
            bond_vector /= np.linalg.norm(bond_vector) # make it a unit vector
            cos_x, cos_y, cos_z = bond_vector / np.linalg.norm(bond_vector)
            gamma_A, gamma_B = self.gyromagnetic_ratio_pairs[i]
            bond_vector_info.append({
                'bond_vector':bond_vector, 'cos_x': cos_x, 'cos_y': cos_y, 'cos_z': cos_z,
                'gyromagnetic_ratio_pairs': self.gyromagnetic_ratio_pairs[i],
                'r':self.internuclear_distances[i],
                'Dmax': -mu_0*gamma_A*gamma_B*hbar / (4. * np.pi**2 * self.internuclear_distances[i]**3)
                })
        return bond_vector_info


    def predict_RDCs_with_SVD(self, experimental_rdcs, scaling_factors=None):
        """Predicts RDCs for all assigned bond vectors, given experimental RDC data.
        Optimizes the alignment tensor to fit the experimental RDC data using Singular Value Decomposition.
        :param experimental_rdcs: List of experimental RDC values.

        """

        self.bond_vector_info = self.get_bond_vector_info()
        if scaling_factors == None: scaling_factors = np.ones(len(experimental_rdcs))

        ########################################################################
        # Construct the structure matrix, A and
        # the experimental dipolar coupling vector, D_exp
        A, D_exp = [], []
        for i, exp in enumerate(experimental_rdcs):
            bv = self.bond_vector_info[i]
            #Dmax = 1.
            Dmax = bv['Dmax']
            A_bond_vector = Dmax*np.array([
                0.5 * (3. * bv['cos_z']**2 - 1.),
                0.5 * (bv['cos_x']**2 - bv['cos_y']**2),
                2.0 * bv['cos_x'] * bv['cos_y'],
                2.0 * bv['cos_x'] * bv['cos_z'],
                2.0 * bv['cos_y'] * bv['cos_z']
                ])
            A.append(A_bond_vector)
            D_exp.append(exp * scaling_factors[i])
        ###########################################
        # $$ Dmax * A * S  = D_exp $$, solve for S
        ###########################################
        A, D_exp = np.array(A), np.array(D_exp)
        # Perform the linear algebra on this vector and matrix
        U, w, V = np.linalg.svd(A, full_matrices=False)

        # Remove inf items from the inverted 'w' vector
        w_inv = 1. / w
        for i in range(w_inv.size):
            w_inv[i] = 0. if w_inv[i] == np.inf else w_inv[i]
        w_inv_diag = np.diag(w_inv)

        # Calculate the Saupe matrix
        A_inv = np.dot(V.T, np.dot(w_inv_diag, U.T)) # pseudo-inverse
        Saupe = np.dot(A_inv, D_exp)
        ########################################################################

        # Reconstruct the 3x3 Saupe matrix and diagonalize
        S_xyz, Da, Dr, Rh = [], [], [], []
        for x in range(0, len(Saupe), 5):
            s = Saupe[x:x+5]
            s_xyz = np.array([[-0.5*(s[0]-s[1]), s[2], s[3]],
                              [s[2], -0.5*(s[0]+s[1]), s[4]],
                              [s[3], s[4], s[0]]
                              ])
            s_xyz = np.linalg.eigvals(s_xyz).real
            S_xyz.append(s_xyz)

            xx, yy, zz = [i for i in sorted(abs(s_xyz))]
            Da.append(max(s_xyz)/2. if max(s_xyz) == zz else min(s_xyz)/2.) # in Hz
            Dr.append((yy-xx)/3.)
            Rh.append(Dr[-1]/abs(Da[-1]))

        ##############################################
        # $$ Dmax * A * S  = D_AB $$, solve for D_AB
        ##############################################
        # Calculate the predicted RDCs (D_AB)
        D_AB = np.dot(A, Saupe)
        delta_D = []
        for i in range(len(experimental_rdcs)):
            delta_D.append(D_AB[i] - experimental_rdcs[i])
            D_AB[i] = D_AB[i]/scaling_factors[i]

        # Calculate the root-mean-squared error
        RMSE = np.sqrt(sum([dev**2 for dev in delta_D])/(len(delta_D))) # in Hz
        # Calculate the RMS of the experimental RDCs
        RMS = np.sqrt(np.mean([rdc**2 for rdc in experimental_rdcs]))
        # Calculate the Q-factor
        Q = RMSE / RMS

        # Calculate R^2
        R2 = np.corrcoef(D_AB, experimental_rdcs)[0,-1]**2

        # Compile dataframe of parameters
        self.parameters = pd.DataFrame([{"Sxx": xx, "Syy": yy, "Szz": zz,
                                         "Da": Da[0], "Dr": Dr[0], "eta": Rh[0]}])

        # Compile dataframe of statistics
        self.statistics = pd.DataFrame([{"Q": Q, "R2": R2, "RMSE": RMSE}])

        # Compile results in dataframe
        A_indices, B_indices = np.array(self.atom_indices).T
        A_B_pairs = [(A, B) for A, B in zip(A_indices, B_indices)]
        results = []
        for i, (A, B) in enumerate(A_B_pairs):
            res = self.traj.topology.atom(B).residue
            atom1,atom2 = self.traj.topology.atom(B).name,self.traj.topology.atom(A).name
            results.append({"residue": res, "atom1":atom1, "atom2":atom2, "exp": experimental_rdcs[i], "rdc": D_AB[i]})
        return pd.DataFrame(results)



if __name__ == "__main__":

    import time
    pd.options.display.max_rows = None
    from parse_star import star_to_df
    filename = "1ubq.pdb"
    outfile = filename.replace(".pdb", ".png")
    rdcp = RDC_predictor(filename)

    file = "../../ubq_biceps/experimental_data/1d3z_mr.str"
    df = get_exp_RDC_file(file, k=2) # 2, 5, 6, 10
    print(df)
    exit()
    atom_name_1, atom_name_2 = df[["atom1", "atom2"]].to_numpy().T
    exp = df["exp"].to_numpy(dtype=float)

    rdcp.get_atom_indices(df)

    start_time = time.time()
    rdcs = rdcp.predict_RDCs_with_SVD(exp)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print(rdcs)
    stats = rdcp.statistics
    parameters = rdcp.parameters
    print(parameters)
    print(stats)


    imgfile = "rdc_corr.png"
    x,y = rdcs["exp"],rdcs["rdc"]
    R2 = stats["R2"].to_numpy()[0]

    ax = simple_plot(x,y,
                xlabel=r'Experiment (Hz)',
                ylabel=r'Prediction (Hz)',
                name=imgfile,
                size=111,Type='scatter',
                color=False,fig_size=(8,6),invert_x_axis=False,fit=True,order=1,
                xLine=None,yLine=None,
                annotate_text=r"$R^{2}$ = %0.2f"%R2,text_x=x.min(),text_y=y.max()-1,
                annotate_x=x.min(),annotate_y=y.max(),
                arrow='->')
    #ax.set_title(f"RDCs for {val1} - {val2}", fontsize=18)
    fig = ax.get_figure()
    fig.tight_layout(pad=2.4)
    fig.savefig(imgfile, dpi=400)
    #exit()

















