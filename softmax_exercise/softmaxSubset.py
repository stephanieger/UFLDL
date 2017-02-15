import numpy as np
from numpy.random import randint

'''
Subsample data for testing numerical gradient:
Inputs are -
1. Data: to be sampled form
2. Label: data label, eg. Y(i)
3. Rows: number of rows in data (arranged in a row x row array)
4. numClasses: number of different data labels (values that Y(i) can take)

Outputs are -
1. subData: subsampled data
2. subLabel: labels for each element in subData
3. subTheta: Theta values for calculating cost and gradients, necessary for calculating numerical gradient
4. subPatchsize: size of each element in the data set

'''
def subsample(Data, Label, rows, numClasses):

    # pick size of subsampled images
    patchsize = 5
    # pick number of images in subset
    num = 10

    # reshape data to draw points from
    Data = Data.reshape(len(Label), rows, rows)

    # initialize size of subset so we can fill in the data
    subData = np.zeros((num, patchsize*patchsize))
    subLabel = [0 for i in range(num)]

    for i in range(num):
        # choose image
        image = randint(0, len(Label))
        # choose patch
        xpatch = randint(0, rows-patchsize+1)
        ypatch = randint(0, rows-patchsize+1)
        tmp = Data[image, xpatch:xpatch+patchsize, ypatch:ypatch+patchsize]
        # assign this data to subset matrix, reshape so that each row is a data point
        subData[i, :] = np.reshape(tmp, patchsize*patchsize, 1)
        subLabel[i] = Label[image]

    # determine the parameters we need to check gradient
    subPatchsize = patchsize*patchsize
    subTheta = 0.005*np.random.randn(numClasses*subPatchsize, 1)

    return subData, subLabel, subTheta, subPatchsize