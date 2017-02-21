import sys
from readMNIST import read
import softmax
import numpy as np
import sparseAuto


def main():

    """
    Step 0: Initialize parameter values for sparse autoencoder and softmax regression.
    """

    row_size = 28
    visible_size = row_size**2
    num_labels = 5
    hidden_size = 200
    rho = 0.1
    lambda_ = 3e-3
    beta = 3
    max_iter = 400

    debug = 0
    """
    Step 1: Load data from the MNIST database

    Load the training set data and sort the training data into labeled and unlabeled sets.
    """

    fIm = sys.argv[1]
    fLb = sys.argv[2]

    # read training data and labels
    data, label = read(fIm, fLb)


    # simulate a labeled and unlabeled set
    # set the numbers that will be used for the unlabeled versus labeled sets
    ixu = np.argwhere(label >= 5).flatten()
    ixl = np.argwhere(label < 5).flatten()

    # build a training and test set -> separate half the labeled set to use as test data
    numTrain = round(ixl.shape[0] / 2) # half the examples in the set

    train_val = ixl[0:numTrain]
    test_val = ixl[numTrain:]

    unlabeled_set = label[ixu]
    unlabeled_data = data[:, ixu]

    train_data = data[:, train_val]
    train_set = label[train_val]

    test_data = data[:, test_val]
    test_set = label[test_val]

    print data.shape
    # output some statistics
    print "# of examples in unlabeled set: {0:6d}".format(len(unlabeled_set))
    print "# of examples in supervised training set: {0:6d}".format(len(train_set))
    print "# of examples in supervised test set: {0:6d}".format(len(test_set))

    """
    Optional Step: Test Sparse Autoencoder
    """
    if debug == 1:
        subdata, sublabel, sub_visible, sub_hidden = sparseAuto.subsample(train_data, train_set, row_size)

        sub_theta = sparseAuto.initializeParameters(sub_hidden, sub_visible)

        sub_grad, sub_cost = sparseAuto.costfun(sub_theta, sub_visible, sub_hidden, lambda_, rho, beta, subdata)

        numGrad, diff = sparseAuto.numgrad(sub_theta, sub_grad, sub_visible, sub_hidden, lambda_, rho, beta, subdata)

        print diff

    """
    Step 2: Train Sparse Autoencoder
    """

    print 'Step 2'
    opt_theta = sparseAuto.train(visible_size, hidden_size, lambda_, rho, beta, unlabeled_data)

    """
    Step 3: Extract Features from the Supervised Data Set
    """
    print 'Step 3'

    train_features = sparseAuto.autoencoder(opt_theta, hidden_size, visible_size, train_data)

    test_features = sparseAuto.autoencoder(opt_theta, hidden_size, visible_size, test_data)

    """
    Step 4: Train the softmax classifier
    """
    print 'Step 4'
    lam = 1e-4

    thetaOpt = softmax.train(num_labels, hidden_size, lam, train_features, train_set)

    """
    Step 5: Testing softmax classfier
    """
    print 'Step 5'
    pred = softmax.predict(thetaOpt, test_features, num_labels, hidden_size)

    print pred[0:15]
    print test_set[0:15]
    # report accuracy of softmax regression
    accuracy = 100*np.sum(test_set == pred)/len(test_set)
    print "Accuracy is {0:.2f}".format(accuracy)


if __name__ == "__main__":
    main()