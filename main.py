#!/usr/bin/env python2
#
# MIT License
# 
# Copyright (c) 2017 Anders Steen Christensen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import random
import copy

import numpy as np

import fml
from fml.kernels import get_atomic_kernels_gaussian
from fml.math import cho_solve

from scipy.linalg import inv

def get_energies(filename):
    """ Returns a dictionary with heats of formation for each xyz-file.
    """

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    energies = dict()

    for line in lines:
        tokens = line.split()

        xyz_name = tokens[0]
        hof = float(tokens[1]) - float(tokens[2])

        energies[xyz_name] = hof

    return energies


if __name__ == "__main__":

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies("hof_qm7.txt")

    # Generate a list of qml.Compound() objects
    mols = []

    # sigmas = [0.1 * 2**i for i in range(20)]
    sigmas = [51.2]

    ntrain = 2000
    ntest  = 1

    random.seed(667)
    keys = sorted(data.keys())
    random.shuffle(keys)

    query = sys.argv[1]

    print "Generating representations"
    for xyz_file in keys:

        if xyz_file == query: 
            continue
        # Initialize the qml.Compound() objects
        # mol = qml.Compound(xyz="qm7/" + xyz_file)
        mol = fml.Molecule()
        mol.read_xyz("qm7/" + xyz_file)
        mol.name = xyz_file
        # Associate a property (heat of formation) with the object
        mol.properties = data[xyz_file]

        # This is a Molecular Coulomb matrix sorted by row norm
        mol.generate_local_coulomb_matrix(size=23)
        #print mol.local_coulomb_matrix
        #print mol.properties
        mols.append(mol)

    mols2 = []
    for xyz_file in [query]:

        # Initialize the qml.Compound() objects
        # mol = qml.Compound(xyz="qm7/" + xyz_file)
        mol = fml.Molecule()
        mol.read_xyz("qm7/" + xyz_file)
        mol.name = xyz_file
        # Associate a property (heat of formation) with the object
        mol.properties = data[xyz_file]

        # This is a Molecular Coulomb matrix sorted by row norm
        mol.generate_local_coulomb_matrix(size=23)
        #print mol.local_coulomb_matrix
        #print mol.properties
        mols2.append(mol)

    train = mols[:ntrain]
    test = mols2[-1:]

    # List of properties
    Y = np.array([mol.properties for mol in train])
    Ys = np.array([mol.properties for mol in test])
    
    print "Generating kernel"
    K = get_atomic_kernels_gaussian(train, train, sigmas)

    print "Generating kernel"
    Ks = get_atomic_kernels_gaussian(test, train, sigmas)

    llambda = 10**(-10.0)

    # for s, sigma in enumerate(sigmas):

    s = 0

    # Solve alpha
    C = copy.deepcopy(K[s])
    C[np.diag_indices_from(C)] += llambda
    alpha = cho_solve(C,Y)

    Yss = np.dot(Ks[s], alpha)

    diff = Ys - Yss

    # Print final RMSD
    # rmsd = np.sqrt(np.mean(np.square(diff)))
    # print "RMSD = %6.2f kcal/mol" % rmsd


    print diff

    C = copy.deepcopy(K[s])
    C[np.diag_indices_from(C)] += llambda

    IY = np.dot(inv(C), Y)
    print IY.shape
    sigma = sigmas[s]

    inv_sigma = -1.0 / (2.0 * sigma**2)

    print test[0].name
    print np.dot(Ks[0], alpha)
    e_tot = 0.0
    for j, xq in enumerate(test[0].local_coulomb_matrix):
        e = 0.0
        for a, mol in enumerate(train):

            k = 0.0
            for i, xt in enumerate(mol.local_coulomb_matrix):
        

                k += np.exp(np.sum(np.square(xq - xt)) * inv_sigma)

            temp = alpha[a] * k
            e += temp
        e_tot += e
        print j, test[0].atomtypes[j], e

    print e_tot
