import numpy as np
from softmaxCost import softmaxCost

################################## CALCULATE NUMERICAL GRADIENT ###############################################


def numericalgrad(theta, numClasses, inputSize, lam, trainData, trainLabel, thetaGrad):
    # initalize variables we need
    ep = 1e-4
    thetaLen = len(theta)
    # reshape calculated gradient to check
    numGrad = np.zeros((thetaLen, 1))
    grad = np.reshape(thetaGrad, (thetaLen, 1))

    #compute numerical gradient for each theta value
    for i in range(thetaLen):
        vec = np.zeros((thetaLen, 1))
        vec[i] = 1
        jp, gradp = softmaxCost(theta+ep*vec, numClasses, inputSize, lam, trainData, trainLabel)
        jm, gradm = softmaxCost(theta-ep*vec, numClasses, inputSize, lam, trainData, trainLabel)
        numGrad[i] = (jp-jm)/2./ep

    # compare numerical gradient with calculated gradient
    diff = np.linalg.norm(numGrad-grad)/np.linalg.norm(numGrad+grad)

    return numGrad, diff
