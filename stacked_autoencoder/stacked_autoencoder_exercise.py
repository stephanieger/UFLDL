import scipy.optimize
import softmax
import numpy as np
from readMNIST import read
import sparseAuto
import stackedae

##======================================================================
'''
STEP 0: Input relevant parameter values
'''

input_size = 28 * 28
num_classes = 10
hidden_size_L1 = 200  # Layer 1 Hidden Size
hidden_size_L2 = 200  # Layer 2 Hidden Size
sparsity_param = 0.1  # desired average activation of the hidden units.
lambda_ = 3e-3  # weight decay parameter
beta = 3  # weight of sparsity penalty term

##======================================================================
'''
 STEP 1: Load data from the MNIST database
'''

# read training data and labels
train_images, train_labels = read('train-images-idx3-ubyte','train-labels-idx1-ubyte')
# read testing data and labels
test_images, test_labels = read('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')

##======================================================================
## STEP 2: Train the first sparse autoencoder

#  Randomly initialize the parameters
sae1_theta = sparseAuto.initializeParameters(hidden_size_L1, input_size)

# optimize parameters
J = lambda x: sparseAuto.costfun(x, input_size, hidden_size_L1,
                                                         lambda_, sparsity_param,
                                                         beta, train_images)

result = scipy.optimize.minimize(J, sae1_theta, method='L-BFGS-B', jac=True, options={'maxiter': 400})
sae1_opt_theta = result.x

print result

##======================================================================
'''
STEP 3: Train the second sparse autoencoder
'''
# use features from the first hidden layer to train the second hidden layer
sae1_features = sparseAuto.autoencoder(sae1_opt_theta, hidden_size_L1,
                                                      input_size, train_images)

#  Randomly initialize the parameters
sae2_theta = sparseAuto.initializeParameters(hidden_size_L2, hidden_size_L1)

# optimize parameters
J = lambda x: sparseAuto.costfun(x, hidden_size_L1, hidden_size_L2,
                                                         lambda_, sparsity_param,
                                                         beta, sae1_features)

result = scipy.optimize.minimize(J, sae2_theta, method='L-BFGS-B', jac=True, options={'maxiter': 400})
sae2_opt_theta = result.x

print result


##======================================================================
'''
STEP 4: Train the softmax classifier
'''
# use features from the second hidden layer to train the softmax classifier
sae2_features = sparseAuto.autoencoder(sae2_opt_theta, hidden_size_L2,
                                                      hidden_size_L1, sae1_features)

# optimize parameters
softmax_theta = softmax.train(num_classes, hidden_size_L2, lambda_, sae2_features, train_labels)

##======================================================================
'''
STEP 5: Finetune softmax model
'''

# Initialize the stack using the parameters learned
stack = [dict() for i in range(2)]
stack[0]['w'] = sae1_opt_theta[0:hidden_size_L1 * input_size].reshape(hidden_size_L1, input_size)
stack[0]['b'] = sae1_opt_theta[2 * hidden_size_L1 * input_size:2 * hidden_size_L1 * input_size + hidden_size_L1]
stack[1]['w'] = sae2_opt_theta[0:hidden_size_L1 * hidden_size_L2].reshape(hidden_size_L2, hidden_size_L1)
stack[1]['b'] = sae2_opt_theta[2 * hidden_size_L1 * hidden_size_L2:2 * hidden_size_L1 * hidden_size_L2 + hidden_size_L2]

# Initialize the parameters for the deep model
(stack_params, net_config) = stackedae.stack2params(stack)

stacked_autoencoder_theta = np.concatenate((softmax_theta.flatten(), stack_params))

# optimize parameters
J = lambda x: stackedae.stackedae_cost(x, input_size, hidden_size_L2,
                                                           num_classes, net_config, lambda_,
                                                           train_images, train_labels)

result = scipy.optimize.minimize(J, stacked_autoencoder_theta, method='L-BFGS-B', jac=True, options={'maxiter': 400})
stacked_autoencoder_opt_theta = result.x

print result

##======================================================================
## STEP 6: Test

print 'step 6'

# Two auto encoders without fine tuning
pred = stackedae.predict(stacked_autoencoder_theta, input_size, hidden_size_L2,
                         num_classes, net_config, lambda_, test_images)
acc = 100*np.sum(test_labels == pred)/len(test_labels)
print "Untuned softmax model accuracy is {0:.2f}".format(acc)

# Two auto encoders with fine tuning
pred = stackedae.predict(stacked_autoencoder_opt_theta, input_size, hidden_size_L2,
                         num_classes, net_config, lambda_,  test_images)
acc = 100*np.sum(test_labels == pred)/len(test_labels)
print "Finetuned softmax model accuracy is {0:.2f}".format(acc)
