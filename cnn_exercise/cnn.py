import numpy as np
from scipy import signal


################################################# SIGMOID FUNCTION #####################################################
def sigmoid(x):
    sigm = 1 / (1 + np.exp(-x))
    return sigm


################################################ CONVOLUTION FUNCTION ##################################################
def convolve(patch_dim, num_features, images, W, b, zca_white, mean_patch):

    # initialize some parameters
    num_images = images.shape[3]
    image_dim = images.shape[0]
    image_channels = images.shape[2]

    convolved_features = np.zeros((num_features, num_images,
                                   image_dim - patch_dim + 1, image_dim - patch_dim +1))

    W_t = W.dot(zca_white)
    b_t = b - W_t.dot(mean_patch)

    for image_num in range(num_images):
        for feature_num in range(num_features):

            convolved_image = np.zeros((image_dim - patch_dim + 1, image_dim - patch_dim + 1))

            for channel in range(image_channels):
                patch_size = patch_dim**2
                feature = W_t[feature_num, patch_size*channel: patch_size*(channel + 1)].reshape(patch_dim, patch_dim)
                feature = np.flipud(np.fliplr(feature))
                im = images[:, :, channel, image_num]
                convolved_image += signal.convolve2d(im, feature, mode='valid')
            convolved_image = sigmoid(convolved_image + b_t[feature_num])
            convolved_features[feature_num, image_num, :, :] = convolved_image

    return convolved_features


################################################ POOLING FUNCTION #####################################################
def pooling(pool_dim, convolved_features):
    num_images = convolved_features.shape[1]
    num_features = convolved_features.shape[0]
    convolved_dim = convolved_features.shape[2]

    # figure out how many blocks we're pooling over
    pooling_dim = convolved_dim/pool_dim
    pooled_features = np.zeros((num_features, num_images, pooling_dim, pooling_dim))

    for i in range(pooling_dim):
        for j in range(pooling_dim):
            tmp = convolved_features[:, :, pool_dim*i: pool_dim*(i+1), pool_dim*j: pool_dim*(j+1)]
            pooled_features[:, :, i, j] = np.mean(np.mean(tmp, 2), 2)

    return pooled_features