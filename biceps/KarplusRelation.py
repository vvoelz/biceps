import numpy as np

"""Note that this code has been adapted from

MDTraj: A Python Library for Loading, Saving, and Manipulating
         Molecular Dynamics Trajectories.
 Copyright 2012-2014 Stanford University and the Authors

 Authors: Kyle A. Beauchamp
 Contributors: Robert McGibbon, TJ Lane, Osama El-Gabalawy

https://github.com/mdtraj/mdtraj/blob/master/mdtraj/nmr/scalar_couplings.py
"""

class KarplusRelation(object):
    """ A class containing formulae giving J-coupling values from dihedral angle."""

    def __init__(self):
        """Initialize the KarplusRelation class."""

        pass

    def J(self, angle, key):
        """Returns the predicted J-coupling constant given an angle in degrees,
        and a key e.g., 'Karplus_HH'"""

        if key == 'Karplus_HH':
            return self.Karplus_HH(angle)
        elif key == 'BothnerBy_HH':
            return self.BothnerBy_HH(angle)
        elif key == 'Allylic':
            return self.Allylic(angle)
        elif key == 'Karplus_antiperiplanar_O':
            return self.Karplus_HH_appO(angle)
        else:
            print("%s is not an appropriate Karplus key..."%key)


    def Karplus_HH(self, angle):
        """Returns predicted J-coupling constant given an angle in degrees."""

        if np.abs(angle) < 90.0:
            J0 = 10.0
        else:
            J0 = 14.0
        # Convert to radians
        theta = angle*np.pi/180.0

        return  J0*(np.cos(theta))**2


    def Karplus_HH_appO(self, angle):
        """Returns predicted J-coupling constant given an angle in degrees,
        for sp3-single bond HH where there is an antiperiplanar Oxygen"""

        if np.abs(angle) < 90.0:
            J0 = 8.0
        else:
            J0 = 11.0
        # Convert to radians
        theta = angle*np.pi/180.0

        return  J0*(np.cos(theta))**2

    def Allylic(self, angle):
        """Returns predicted J-coupling constant given an angle in degrees,
        for an allylic sp3-sp2 bond"""

        if np.abs(angle) < 90.0:
            J0 = 6.6
        else:
            J0 = 11.6
        # Convert to radians
        theta = angle*np.pi/180.0

        return  J0*(np.cos(theta))**2 + 2.6*(np.sin(theta))**2


    def BothnerBy_HH(self, angle):
        """Returns predicted J-coupling constant given an angle in degrees.
        NOTES: good for cycloalkanes.  Bothner-By, 1965 (14)"""

        # Convert to radians
        theta = angle*np.pi/180.0

        return  7.0 - np.cos(theta) + 5.0*np.cos(2.0*theta)

    def Altona_HH(self, angle):
        """
        NOTES: From: http://janocchio.sourceforge.net/janocchio.docbook/ch04s03.html
        Altona is good if the protons are attached to two sp3 carbons. This equation
        takes into account the electronegativity of all adjacent atoms.

        Haasnoot, C., de Leeuw, FA, Altona, C (1980). Tetrahedron 36(19), 2783-2792
        """

        # This is really complicated -- you have to know the electronegativities of substituents
        # TBA

        pass


    def Karplus2_HH(self, angle):
        """Returns predicted J-coupling constant given an angle in degrees.
        NOTES: A simple Karplus relation from the Janocchio documentation.

        Magnetic Resonance in Chemistry, 45(7), 595-600. doi:10.1002/mrc/2016 """

        if np.abs(angle) < 90.0:
            J0 = 8.5
        else:
            J0 = 9.5

        # Convert to radians
        theta = angle*np.pi/180.0

        return  J0*(np.cos(theta)**2) - 0.28

    def Wasylichen_HC(self, angle):
        """Returns predicted J-coupling constant given an angle in degrees.
        NOTES: Recommended for general use (Can J Chem (1973) 51 961.)
        See also Janocchio paper"""

        # Convert to radians
        theta = angle*np.pi/180.0

        return  3.56*np.cos(2.0*theta) - np.cos(theta) + 4.26

    def Tvaroska_HC(self, angle):
        """Returns predicted J-coupling constant given an angle in degrees.
        NOTES: Adv. Carbohydrate Chem. Biochem. (1995) 51, 15-61
        See also Janocchio paper"""

        # Convert to radians
        theta = angle*np.pi/180.0

        return  4.5 - 0.87*np.cos(theta) + np.cos(2.0*theta)

    def Aydin_HC(self, angle):
        """Returns predicted J-coupling constant given an angle in degrees.
        NOTES: Mag. Res. Chem. (1990) 28, 448-457
        See also Janocchio paper"""

        # Convert to radians
        theta = angle*np.pi/180.0

        return 5.8*(np.cos(theta))**2 -1.6*np.cos(theta) + 0.28*np.sin(2.0*theta) + 0.52



if __name__ == '__main__':

    # Perform a scan of all the Karplus relations

    from matplotlib import pyplot as plt

    karplus = KarplusRelation()
    angles = np.arange(-180,180,1.0)

    results = []
    results.append( [karplus.Karplus_HH(a) for a in angles] )
    results.append( [karplus.BothnerBy_HH(a) for a in angles] )
    results.append( [karplus.Karplus2_HH(a) for a in angles] )

    plt.figure()
    for result in results:
        plt.plot(angles, result)
        plt.hold(True)
    plt.xlim(0,180)
    plt.ylim(-2, 16)
    plt.yticks(list(range(16)))
    plt.legend(['Karplus', 'Bothner-By', 'Karplus2'], 'upper left')
    plt.xlabel('angle (degrees)')
    plt.ylabel('$^3 J_{HH}$')
    plt.savefig('karplus_relation_test.png')






