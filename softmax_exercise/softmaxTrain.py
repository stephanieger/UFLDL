from scipy import optimize
import numpy as np
from softmaxCost import softmaxCost

############################################ TRAIN SOFTMAX REGRESSION ##################################################

def train(numClasses, inputSize, lam, trainData, trainLabel):
    # randomly initialize theta
    theta = 0.005*np.random.randn(numClasses*inputSize, 1)
    # implement softmaxCost
    J = lambda x: softmaxCost(x, numClasses, inputSize, lam, trainData, trainLabel)
    print inputSize
    tmp = optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options={'maxiter': 100,'disp': True})
    thetaOpt = tmp.x

    return thetaOpt
