import sys
from readMNIST import read
from computeNumericalGradient import numericalgrad
from softmaxSubset import subsample
import softmax
import numpy as np

def main():
    fnameTrainIm = sys.argv[1]
    fnameTrainLb = sys.argv[2]
    fnameTestIm = sys.argv[3]
    fnameTestLb = sys.argv[4]


    # read training data and labels
    trainData, trainLabel = read(fnameTrainIm, fnameTrainLb)

    # initialize constants and parameters
    inputSize = 28*28
    numClasses = 10
    lam = 1e-4
    DEBUG = 0


    if DEBUG == 1:
        # compute difference between numerical gradient and computed grad on the subset
        subData, subLabel, subTheta, subPatchsize = subsample(trainData, trainLabel, rows, numClasses)
        subCost, subGrad = softmax.softmaxCost(subTheta, numClasses, subPatchsize, lam, subData, subLabel)
        numGrad, diff = numericalgrad(subTheta, numClasses, subPatchsize, lam, subData, subLabel, subGrad)
        print "The diff is", diff

    # train regression code
    thetaOpt = softmax.train(numClasses, inputSize, lam, trainData, trainLabel)

    # predict labels for test data
    testData, testLabel = read(fnameTestIm, fnameTestLb)
    pred = softmax.predict(thetaOpt, testData, numClasses, inputSize)

    # report accuracy of softmax regression
    accuracy = 100*np.sum(testLabel == pred)/len(testLabel)
    print "Accuracy is {0:.2f}".format(accuracy)


if __name__ == "__main__":
    main()
