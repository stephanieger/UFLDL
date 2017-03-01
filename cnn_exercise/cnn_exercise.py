import numpy as np
import scipy.io as sio
import cnn
import sparseAuto
import sys
import pprint
import time
import datetime
import softmax


def main():
    """
    Step 0: Initialization
    """
    # Initialize parameters for the exercise

    image_dim = 64
    image_channels = 3

    patch_dim = 8
    num_patches = 50000

    visible_size = patch_dim*patch_dim*image_channels
    output_size = visible_size
    hidden_size = 400

    epsilon = 0.1
    pool_dim = 19

    debug = 0
    """
    Step 1: Train a sparse autoencoder (with a linear decoder)
    """
    # load data from linear decoder exercise
    opt_theta = np.load('opt_theta.npy')
    zca_white = np.load('zca_white.npy')
    mean_patch = np.load('mean_patch.npy')

    # unpack W and b
    W = opt_theta[0: hidden_size*visible_size].reshape(hidden_size, visible_size)
    b = opt_theta[2*hidden_size*visible_size: 2*hidden_size*visible_size + hidden_size]

    """
    Step 2a: Implement convolution
    """
    # read in train data from mat file
    mat_contents = sio.loadmat('stlTrainSubset.mat')
    train_images = mat_contents['trainImages']
    train_labels = mat_contents['trainLabels']
    num_train_images = mat_contents['numTrainImages'][0][0]

    # read in test data from mat file
    mat_contents = sio.loadmat('stlTestSubset.mat')
    test_images = mat_contents['testImages']
    test_labels = mat_contents['testLabels']
    num_test_images = mat_contents['numTestImages'][0][0]


    # use only the first 8 images for testing
    conv_images = train_images[:, :, :, 0:8]

    # use only the first 8 images to test
    convolved_features = cnn.convolve(patch_dim, hidden_size, conv_images, W, b, zca_white, mean_patch)
    if debug == 1:

        """
        Step 2b: Check your convolution
        """

        for i in range(1000):
            feature_num = np.random.randint(0, hidden_size)
            image_num = np.random.randint(0, 8)
            image_row = np.random.randint(0, image_dim - patch_dim + 1)
            image_col = np.random.randint(0, image_dim - patch_dim + 1)
            patch = conv_images[image_row: image_row + patch_dim, image_col: image_col+patch_dim, :, image_num]
            patch = np.concatenate((patch[:, :, 0].flatten(), patch[:, :, 1].flatten(), patch[:, :, 2].flatten()))
            patch = np.reshape(patch, (patch.size, 1))
            patch = patch - np.tile(mean_patch, (patch.shape[1], 1)).transpose()
            patch = zca_white.dot(patch)
            features = sparseAuto.autoencoder(opt_theta, hidden_size, visible_size, patch)

            if abs(features[feature_num, 0] - convolved_features[feature_num, image_num, image_row, image_col]) > 1e-9:
                print 'convolved feature does not activation from autoencoder'
                print 'feature number:', feature_num
                print 'image number:', image_num
                print 'image row:', image_row
                print 'image column:', image_col
                print 'convolved feature:', convolved_features[feature_num, image_num, image_row, image_col]
                print 'sparse AE feature:', features[feature_num, 0]
                sys.exit('convolved feature does not match activation from autoencoder')

        print('congrats! convolution code passed the test')

        """
        Step 2c: Implement Pooling
        """
        # pooled_features = cnn.pooling(pool_dim, convolved_features)

        """
        Step 2d: Checking your pooling
        """
        test_matrix = np.arange(64).reshape(8, 8)
        expected_matrix = np.array([[np.mean(test_matrix[0:4, 0:4]), np.mean(test_matrix[0:4, 4:8])],
                                    [np.mean(test_matrix[4:8, 0:4]), np.mean(test_matrix[4:8, 4:8])]])
        test_matrix = test_matrix.reshape(1, 1, 8, 8)
        pooled_features = cnn.pooling(4, test_matrix)

        if not(pooled_features == expected_matrix).all():
            print 'pooling incorrect'
            print 'expected:'
            pprint.pprint(expected_matrix)
            print 'got:'
            pprint.pprint(pooled_features)
        else:
            print "congrats! pooling code passed the test"

    """
    Step 3: Convolve and pool with the data set
    """
    step_size = 50
    assert hidden_size%step_size == 0

    pooled_features_train = np.zeros((hidden_size, num_train_images, int(np.floor((image_dim-patch_dim+1)/pool_dim)),
                                      int(np.floor((image_dim-patch_dim+1)/pool_dim))))

    pooled_features_test = np.zeros((hidden_size, num_test_images, int(np.floor((image_dim-patch_dim+1)/pool_dim)),
                                     int(np.floor((image_dim-patch_dim+1)/pool_dim))))
    
    start_time = time.time()

    for conv_part in range(hidden_size/step_size):

        feature_start = conv_part*step_size
        feature_end = (conv_part+1)*step_size

        print 'Step:', conv_part, 'Features', feature_start, 'to', feature_end
        Wt = W[feature_start: feature_end, :]
        bt = b[feature_start: feature_end]

        print 'Convolving and pooling train images'
        convolved_features_this = cnn.convolve(patch_dim, step_size, train_images, Wt, bt, zca_white, mean_patch)
        pooled_features_this = cnn.pooling(pool_dim, convolved_features_this)
        pooled_features_train[feature_start: feature_end, :, :, :] = pooled_features_this

        print 'Elapsed time is', str(datetime.timedelta(seconds=time.time() - start_time))
        print 'Convolving and pooling test images'

        convolved_features_this = cnn.convolve(patch_dim, step_size, test_images, Wt, bt, zca_white, mean_patch)
        pooled_features_this = cnn.pooling(pool_dim, convolved_features_this)
        pooled_features_test[feature_start: feature_end, :, :, :] = pooled_features_this

        print 'Elapsed time is', str(datetime.timedelta(seconds=time.time() - start_time))

    np.save('pooled_features_train.npy', pooled_features_train)
    np.save('pooled_features_test.npy', pooled_features_test)
    print 'Elapsed time is', str(datetime.timedelta(seconds=time.time() - start_time))


    """
    Step 4: Use pooled features for classification
    """
    # set up parameters for softmax
    softmax_lambda = 1e-4
    num_classes = 4

    # reshape the pooled_features to form an input vector for softmax
    softmax_x = np.transpose(pooled_features_train, [0, 2, 3, 1])
    softmax_x = softmax_x.reshape((pooled_features_train.size/num_train_images, num_train_images))
    softmax_y = train_labels.flatten()-1

    softmax_opt_theta = softmax.train(num_classes, pooled_features_train.size/num_train_images, softmax_lambda, softmax_x, softmax_y)
    np.save('theta_opt_theta.npy', opt_theta)
    """
    Step 5: Test Classifier
    """
    softmax_x = np.transpose(pooled_features_test, [0, 2, 3, 1])
    softmax_x = softmax_x.reshape((pooled_features_test.size/num_test_images, num_test_images))
    softmax_y = test_labels.flatten()-1
    pred = softmax.predict(softmax_opt_theta, softmax_x, num_classes, pooled_features_train.size/num_train_images)

    accuracy = 100*np.sum(softmax_y == pred)/len(softmax_y)
    print "Accuracy is {0:.2f}".format(accuracy)

if __name__ == "__main__":
    main()
