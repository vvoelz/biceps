import sys, os, glob

from Structure import *

ensemble = []
ensemble.append( Structure('albo/albo_39.pdb', 0.0) )
ensemble.append( Structure('albo/albo_80.pdb', 0.0) )
ensemble.append( Structure('albo/albo_92.pdb', 0.0) )

for s in ensemble:
    s.add_distance_restraint( 22, 45, 2.5)  # H6-H25  "H(4)-H(7) " in Terekhovs et al 2007
    s.add_distance_restraint( 28, 45, 2.5)  # H11-H25 "H(4)-H(10)" in Terekhovs et al 2007
    s.add_distance_restraint( 22, 28, 2.5)  # H6-H11  "H(7)-H(10)" in Terekhovs et al 2007

# Compare the model and exp distances
for s in ensemble:
    print s
    for d in s.distance_restraints:
        print 'd.model_distance =', d.model_distance,
        print 'd.noe_distance = ', d.noe_distance


"""
ATOM      1  C1  UNK     1       1.484   2.515   0.317  1.00  0.00           C
ATOM      2  O1  UNK     1       1.764   3.456  -0.687  1.00  0.00           O
ATOM      3  H1  UNK     1       2.690   3.660  -0.679  1.00  0.00           H
ATOM      4  C2  UNK     1       0.024   2.150   0.230  1.00  0.00           C
ATOM      5  C3  UNK     1      -0.834   2.470  -0.721  1.00  0.00           C
ATOM      6  C4  UNK     1      -2.174   1.833  -0.796  1.00  0.00           C
ATOM      7  O2  UNK     1      -2.287   0.813   0.047  1.00  0.00           O
ATOM      8  C5  UNK     1      -3.415  -0.066   0.010  1.00  0.00           C
ATOM      9  C6  UNK     1      -2.831  -1.448   0.343  1.00  0.00           C
ATOM     10  C7  UNK     1      -1.943  -1.968  -0.812  1.00  0.00           C
ATOM     11  C8  UNK     1      -0.712  -2.776  -0.360  1.00  0.00           C
ATOM     12  C9  UNK     1       0.446  -1.876   0.028  1.00  0.00           C
ATOM     13  C10 UNK     1       1.724  -2.210   0.127  1.00  0.00           C
ATOM     14  C11 UNK     1       2.822  -1.193   0.434  1.00  0.00           C
ATOM     15  O3  UNK     1       3.656  -1.138  -0.695  1.00  0.00           O
ATOM     16  C12 UNK     1       4.980  -0.774  -0.446  1.00  0.00           C
ATOM     17  H2  UNK     1       5.471  -1.495   0.203  1.00  0.00           H
ATOM     18  H3  UNK     1       5.050   0.209   0.010  1.00  0.00           H
ATOM     19  H4  UNK     1       5.491  -0.756  -1.399  1.00  0.00           H
ATOM     20  C13 UNK     1       2.308   0.171   0.836  1.00  0.00           C
ATOM     21  C14 UNK     1       2.203   1.201   0.018  1.00  0.00           C
ATOM     22  H5  UNK     1       2.522   1.097  -1.005  1.00  0.00           H
ATOM     23  H6  UNK     1       1.949   0.236   1.851  1.00  0.00           H
ATOM     24  H7  UNK     1       3.401  -1.589   1.271  1.00  0.00           H
ATOM     25  C15 UNK     1       2.251  -3.609  -0.082  1.00  0.00           C
ATOM     26  H8  UNK     1       2.915  -3.642  -0.940  1.00  0.00           H
ATOM     27  H9  UNK     1       2.826  -3.933   0.783  1.00  0.00           H
ATOM     28  H10 UNK     1       1.459  -4.329  -0.235  1.00  0.00           H
ATOM     29  H11 UNK     1       0.175  -0.850   0.192  1.00  0.00           H
ATOM     30  H12 UNK     1      -0.408  -3.430  -1.170  1.00  0.00           H
ATOM     31  H13 UNK     1      -0.983  -3.429   0.468  1.00  0.00           H
ATOM     32  H14 UNK     1      -2.556  -2.587  -1.462  1.00  0.00           H
ATOM     33  H15 UNK     1      -1.598  -1.142  -1.423  1.00  0.00           H
ATOM     34  C16 UNK     1      -3.912  -2.479   0.687  1.00  0.00           C
ATOM     35  H16 UNK     1      -3.467  -3.460   0.821  1.00  0.00           H
ATOM     36  H17 UNK     1      -4.439  -2.234   1.602  1.00  0.00           H
ATOM     37  H18 UNK     1      -4.644  -2.562  -0.113  1.00  0.00           H
ATOM     38  H19 UNK     1      -2.212  -1.305   1.226  1.00  0.00           H
ATOM     39  C17 UNK     1      -4.462   0.458   0.981  1.00  0.00           C
ATOM     40  H20 UNK     1      -4.089   0.439   2.000  1.00  0.00           H
ATOM     41  H21 UNK     1      -5.370  -0.133   0.935  1.00  0.00           H
ATOM     42  H22 UNK     1      -4.716   1.479   0.725  1.00  0.00           H
ATOM     43  H23 UNK     1      -3.818  -0.059  -0.994  1.00  0.00           H
ATOM     44  O4  UNK     1      -3.028   2.164  -1.557  1.00  0.00           O
ATOM     45  H24 UNK     1      -0.597   3.152  -1.514  1.00  0.00           H
ATOM     46  H25 UNK     1      -0.291   1.472   1.000  1.00  0.00           H
ATOM     47  C18 UNK     1       1.820   3.107   1.689  1.00  0.00           C
ATOM     48  H26 UNK     1       1.278   4.035   1.824  1.00  0.00           H
ATOM     49  H27 UNK     1       2.885   3.312   1.759  1.00  0.00           H
ATOM     50  H28 UNK     1       1.559   2.434   2.497  1.00  0.00           H
"""

