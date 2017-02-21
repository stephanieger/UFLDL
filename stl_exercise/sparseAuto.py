import numpy as np
from numpy.random import rand
from numpy import matlib
from numpy.random import randint
from scipy import optimize

########################################## SUBSAMPLE DATA FOR TESTING ##################################################


def subsample(data, label, row_size):

    num_files = 15
    patchsize = 4
    sub_visible = patchsize**2
    sub_hidden = 5

    # reshape data for drawing elements from it
    re_data = data.reshape(row_size, row_size, len(label))

    # initialize size of subdata and sublabel so we can fill in the data
    subdata = np.zeros((patchsize**2, num_files))
    sublabel = [0 for i in range(num_files)]

    for i in range(num_files):
        # choose image
        image = randint(0, len(label))
        # choose patch
        xpatch = randint(0, row_size-patchsize+1)
        ypatch = randint(0, row_size-patchsize+1)
        tmp = re_data[xpatch:xpatch+patchsize, ypatch:ypatch+patchsize, image]
        # assign this data to subset matrix, reshape so that each row is a data point
        subdata[:, i] = np.reshape(tmp, patchsize*patchsize, 1)
        sublabel[i] = label[image]

    return subdata, sublabel, sub_visible, sub_hidden


############################################ INITIALIZE PARAMETERS #####################################################


def initializeParameters(hidden_size, visible_size):
    r = np.sqrt(6)/np.sqrt(hidden_size + visible_size + 1)
    W1 = rand(hidden_size, visible_size)*2*r - r
    W2 = rand(visible_size, hidden_size)*2*r - r

    b1 = np.zeros((hidden_size, 1))
    b2 = np.zeros((visible_size, 1))

    theta = np.concatenate((W1.flatten(), W2.flatten(), b1.flatten(), b2.flatten()))
    return theta

########################################### SPARSE AUTOENCODER COST ####################################################


def costfun(theta, visible_size, hidden_size, lambda_, rho, beta, data):

    # Read off coefficients and reshape matrices from theta
    W1 = theta[0:hidden_size*visible_size].reshape(hidden_size, visible_size)
    W2 = theta[hidden_size*visible_size:2*hidden_size*visible_size].reshape(visible_size, hidden_size)
    b1 = theta[2*hidden_size*visible_size:2*hidden_size*visible_size+hidden_size]
    b2 = theta[2*hidden_size*visible_size+hidden_size:]

    # rename some variables
    m = data.shape[1]

    # calculate sparsity average
    rho_hat = np.sum(sigmoid(np.dot(W1, data) + np.matlib.repmat(b1, m, 1).transpose()), 1)/m

    # forward propagation step
    a2 = sigmoid(np.dot(W1, data) + np.matlib.repmat(b1, m, 1).transpose())
    a3 = sigmoid(np.dot(W2, a2) + np.matlib.repmat(b2, m, 1).transpose())

    # backpropagation step
    delta3 = -(data - a3)*a3*(1.-a3)
    sparse = - rho/rho_hat + (1-rho)/(1-rho_hat)
    delta2 = (np.dot(W2.transpose(), delta3) + beta*np.matlib.repmat(sparse, m, 1).transpose())*a2*(1.-a2)

    # gradients
    W1grad = np.dot(delta2, data.transpose())/m + lambda_*W1
    W2grad = np.dot(delta3, a2.transpose())/m + lambda_*W2
    b1grad = np.sum(delta2, 1)/m
    b2grad = np.sum(delta3, 1)/m

    # flatten gradients
    grad = np.concatenate((W1grad.flatten(), W2grad.flatten(), b1grad.flatten(), b2grad.flatten()))

    # calculate cost
    sparse_pen = beta*sum(rho*np.log(rho/rho_hat) + (1-rho)*np.log((1-rho)/(1-rho_hat)))
    weight_pen = 0.5*lambda_*(sum(sum(W1**2)) + sum(sum(W2**2)))
    cost = np.sum(0.5*np.sum(abs(a3-data)**2, 1), 0)/m + sparse_pen + weight_pen

    return cost, grad


############################################## COMPUTE NUMERICAL GRAD ##################################################


def numgrad(theta, grad, visible_size, hidden_size, lambda_, rho, beta, data):

    ep = 1e-4
    # reshape calculated gradient to check
    numGrad = np.zeros(theta.shape)

    # compute numerical gradient for each theta value
    for i in range(len(theta)):
        vec = np.zeros(theta.shape)
        vec[i] = 1
        jp, gradp = costfun(theta+ep*vec, visible_size, hidden_size, lambda_, rho, beta, data)
        jm, gradm = costfun(theta-ep*vec, visible_size, hidden_size, lambda_, rho, beta, data)
        numGrad[i] = (jp-jm)/2./ep

    # compare numerical gradient with calculated gradient
    diff = np.linalg.norm(numGrad-grad)/np.linalg.norm(numGrad+grad)

    return numGrad, diff

################################################# TRAIN AUTOENCODER ####################################################


def train(visible_size, hidden_size, lambda_, rho, beta, data):

    # initialize theta
    theta = initializeParameters(hidden_size, visible_size)

    # implement sparse autoencoder
    J = lambda x: costfun(x, visible_size, hidden_size, lambda_, rho, beta, data)

    #optimize theta valuees
    tmp = optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options={'maxiter': 400})
    opt_theta = tmp.x

    return opt_theta

######################################### COMPUTE HIDDEN LAYER ACTIVATION ##############################################


def autoencoder(theta, hidden_size, visible_size, data):

    # rename some variables
    m = data.shape[1]

    W1 = theta[0:hidden_size*visible_size].reshape(hidden_size, visible_size)
    b1 = theta[2*hidden_size*visible_size:2*hidden_size*visible_size+hidden_size]
    activation = sigmoid(np.dot(W1, data) + np.matlib.repmat(b1, m, 1).transpose())

    return activation

############################################## DEFINE SIGMOID FUNCTION #################################################


def sigmoid(x):

    sigm = 1./(1.+np.exp(-x))

    return sigm

