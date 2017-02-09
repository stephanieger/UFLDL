'''
Stephanie Ger
Feb 5 2017
PCA and Whitening
UFLDL Tutorial

Inputs:
argv[1] - text file with data
argv[2] - dimensions for compressed data
argv[3] - epsilon for smoothing
'''

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

# implement pca on test data

############################### READ DATA ###########################################
''' Read in data from text file into the x and y vectors respectively
'''


def readdata(pcaData):
    tmp = []
    s = open(pcaData, 'rU')
    r = csv.reader(s, delimiter=' ', skipinitialspace=True)
    for row in r:
        tmp.append(row)
    x = np.zeros((len(tmp[0]), 1))
    y = np.zeros((len(tmp[0]), 1))
    for i in range(len(tmp[0])):
        x[i] = float(tmp[0][i])
        y[i] = float(tmp[1][i])

    return x, y


########################## CALCULATE ROTATION MATRIX #################################
def calcU(x, y):
    # read x and y into a vector
    vec = np.zeros((2, len(x)))
    vec[0, :] = np.transpose(x)
    vec[1, :] = np.transpose(y)
    vecT = np.transpose(vec)

    # calculate U and then scale by number of data points
    mat = np.dot(vec, vecT) / len(x)
    U, s, V = np.linalg.svd(mat, full_matrices=True)
    eig = s
    return vec, U, eig


######################### CALCULATE TRANSFORMATIONS OF X #############################
'''This function calculates a number of transformations of the data in order to
preprocess the data.
xRot - project the data to the eigenbasis
xHat - project the data onto u1 (k =1)
xPCAWhite - whiten xRot by normalizing variances
zPCAWhite - do ZCA whitening on xPCAWhite
'''

def preprocessing(U,vec,eig,k,ep):
    #calculate xRot
    print U
    print np.transpose(U)
    print vec[:,1]
    UT = np.transpose(U)
    xRot = np.dot(UT, vec)

    #calculate xHat
    uT= UT[:k, :]
    xHat = np.dot(uT, vec)

    #calculate xPCAWhite
    eigV = np.sqrt(eig + ep)
    eigV = np.divide(1., eigV)
    xPCAWhite = np.zeros(xRot.shape)
    for i in range(len(eig)):
        xPCAWhite[i,:] = eigV[i]*xRot[i,:]

    #calculate zPCAWhite
    xZCAWhite = np.dot(U, xPCAWhite)

    return xRot, xHat, xPCAWhite, xZCAWhite

############################### MAIN FILE ############################################

def main(args):
    pcaData = args[1]
    k = int(args[2])
    ep = float(args[3])

    x, y = readdata(pcaData)
    vec, U, eig = calcU(x, y)
    xRot, xHat, xPCAWhite, xZCAWhite = preprocessing(U,vec,eig,k,ep)

    #plotting functions
    '''
    plt.figure(1)
    plt.scatter(x, y)
    plt.title('Raw Data')

    plt.figure(2)
    plt.hold(True)
    plt.plot([0, U[0, 0]], [0, U[1, 0]])
    plt.plot([0, U[0, 1]], [0, U[1, 1]])
    plt.scatter(x, y)
    plt.hold(False)

    plt.figure(3)
    plt.scatter(xRot[0,:],xRot[1,:])
    '''
    plt.figure(4)
    plt.scatter(xPCAWhite[0, :], xPCAWhite[1, :])

    plt.figure(5)
    plt.scatter(xZCAWhite[0, :], xZCAWhite[1, :])
    plt.show()
######################################################################################
# run program
main(sys.argv)
