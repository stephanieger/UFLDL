import numpy as np
from numpy import random as random
import sparse_linear
import scipy.io as sio


def main():
    """
    Step 0: Initialization
    """
    # Initialize parameters for the exercise

    image_channels = 3
    patch_dim = 8
    num_patches = 100000

    visible_size = patch_dim*patch_dim*image_channels
    hidden_size = 400

    rho = 0.035
    lambda_ = 3e-3
    beta = 5

    epsilon = 0.1

    # debug toggle
    debug = 0
    """
    Step 1: Create and modify sparseAuto to use a linear decoder and check gradients
    """
    if debug == 1:
        debug_hidden_size = 5
        debug_visible_size = 8
        patches =random.uniform(0, 1, (8, 10))
        theta = sparse_linear.initializeParameters(debug_hidden_size, debug_visible_size)
        cost, grad = sparse_linear.costfun(theta, debug_visible_size, debug_hidden_size, lambda_, rho, beta, patches)
        numGrad, diff = sparse_linear.numgrad(theta, grad, debug_visible_size, debug_hidden_size,
                                              lambda_, rho, beta, patches)
        print diff

    '''
    Step 2: Load patches, apply preprocessing, learn features, and visualize learned features
    '''
    # load data
    mat_contents = sio.loadmat('stlSampledPatches.mat')
    patches = mat_contents['patches']

    # apply preprocessing
    mean_patch = np.mean(patches, 1)
    patches = patches - np.tile(mean_patch, (patches.shape[1], 1)).transpose()
    sigma = patches.dot(patches.transpose())/num_patches
    u, s, v = np.linalg.svd(sigma)
    zca_white = np.dot(u.dot(np.diag(1/np.sqrt(s + epsilon))), u.transpose())
    patches = zca_white.dot(patches)

    # learn features
    opt_theta = sparse_linear.train(visible_size, hidden_size, lambda_, rho, beta, patches)
    np.save('opt_theta.npy', opt_theta)
    np.save('zca_white.npy', zca_white)
    np.save('mean_patch.npy', mean_patch)

    # visualize learned features
    W = opt_theta[0:visible_size*hidden_size].reshape(hidden_size, visible_size)
    b = opt_theta[2*hidden_size*visible_size: 2*hidden_size*visible_size+hidden_size]

if __name__ == "__main__":
    main()
