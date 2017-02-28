import numpy as np
from numpy import matlib
from scipy import optimize
import softmax

############################################## DEFINE SIGMOID FUNCTION #################################################


def sigmoid(x):

    sigm = 1./(1.+np.exp(-x))

    return sigm


######################## CONVERT STACK STRUCTURE TO FLATTENED PARAMETER VECTOR ##########################
# This function from https://github.com/jatinshah
def stack2params(stack):
    params = []
    for s in stack:
        params.append(s['w'].flatten())
        params.append(s['b'].flatten())
    params = np.concatenate(params)

    netconfig = {}
    if len(stack) == 0:
        netconfig['input_size'] = 0
        netconfig['layer_size'] = []
    else:
        netconfig['input_size'] = stack[0]['w'].shape[1]
        netconfig['layer_size'] = []
        for s in stack:
            netconfig['layer_size'].append(s['w'].shape[0])

    return params, netconfig


######################## CONVERT FLATTENED PARAMETER VECTOR TO STACK STRUCTURE ##########################
# This function from https://github.com/jatinshah
def params2stack(params, netconfig):

    depth = len(netconfig['layer_size'])
    stack = [dict() for i in range(depth)]

    prev_layer_size = netconfig['input_size']
    current_pos = 0

    for i in range(depth):
        wlen = prev_layer_size*netconfig['layer_size'][i]
        stack[i]['w'] = params[current_pos:current_pos+wlen].reshape(netconfig['layer_size'][i], prev_layer_size)

        # extract bias
        blen = netconfig['layer_size'][i]
        stack[i]['b'] = params[current_pos:current_pos+blen]

        # set previous layer size
        prev_layer_size = netconfig['layer_size'][i]

    return stack


#################################### STACKED AUTOENCODER COST ##########################################

def stackedae_cost(theta, input_size, hidden_size, num_classes, netconfig, lambda_, data, labels):

    # extract the part which compute the softmax gradient
    softmax_theta = theta[0:hidden_size*num_classes].reshape(num_classes, hidden_size)

    # extract out the "stack"
    stack = params2stack(theta[hidden_size*num_classes:], netconfig)

    m = data.shape[1]
    groundTruth = np.zeros((num_classes, m))
    groundTruth[labels.astype(int), np.linspace(0, m-1, m, dtype=int)] = 1.

    # forward propagation, eg. store a and z values
    # initialize storage
    a = [data]
    z = [np.array(0)]
    for s in stack:
        z.append(np.dot(s['w'], a[-1]) + np.matlib.repmat(s['b'], m, 1).transpose())
        a.append(sigmoid(z[-1]))

    # softmax cost
    exp_power = np.dot(softmax_theta, a[-1])
    exp_power = exp_power - np.max(exp_power)
    cost_den = np.sum(np.exp(exp_power), 0)
    hyp = np.divide(np.exp(exp_power), np.matlib.repmat(cost_den, num_classes, 1))
    num = np.multiply(groundTruth, np.log(hyp))

    softmax_cost = (-1./float(m))*sum(sum(num)) + 0.5*lambda_*sum(sum(np.multiply(softmax_theta, softmax_theta)))

    # softmax grad
    diff = groundTruth - hyp
    softmax_theta_grad = -np.dot(diff, np.transpose(a[-1]))/m + lambda_*softmax_theta

    # backpropagation
    softmax_theta_grad_a = np.dot(softmax_theta.transpose(), groundTruth - hyp)

    # compute delta for the outermost layer
    delta = [- softmax_theta_grad_a*a[-1]*(1-a[-1])]

    # go backwards and compute delta for the other layers
    for i in reversed(range(len(stack))):
        # print stack[i]['w'].shape
        d = np.dot(np.transpose(stack[i]['w']), delta[0])*a[i]*(1-a[i])
        delta.insert(0, d)

    stack_grad = [dict() for i in range(len(stack))]
    for i in range(len(stack)):
        stack_grad[i]['w'] = np.dot(delta[i+1], a[i].transpose())/m
        stack_grad[i]['b'] = np.sum(delta[i+1], 1)/m

    grad_param, grad_netconfig = stack2params(stack_grad)

    # print grad_param.shape
    # print softmax_theta.shape
    grad = np.concatenate((softmax_theta_grad.flatten(), grad_param))

    return softmax_cost, grad

############################################### TRAIN MODEL ###################################################


def train(input_size, hidden_size, num_classes, lambda_, data, labels, theta, netconfig):

    J = lambda x: stackedae_cost(theta, input_size, hidden_size, num_classes, netconfig, lambda_, data, labels)

    tmp = optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options={'maxiter':400})
    opt_theta = tmp.x

    return opt_theta

########################################## PREDICT LABELS FOR TEST DATA ################################################


def predict(theta, input_size, hidden_size, num_classes, netconfig, lambda_, data):

    # extract the part which compute the softmax gradient
    softmax_theta = theta[0:hidden_size*num_classes].reshape(num_classes, hidden_size)

    # extract out the "stack"
    stack = params2stack(theta[hidden_size*num_classes:], netconfig)

    m = data.shape[1]
    a = [data]
    z = [np.array(0)]
    for s in stack:
        z.append(np.dot(s['w'], a[-1]) + np.matlib.repmat(s['b'], m, 1).transpose())
        a.append(sigmoid(z[-1]))

    pred = softmax.predict(softmax_theta, a[-1], num_classes, hidden_size)

    return pred
