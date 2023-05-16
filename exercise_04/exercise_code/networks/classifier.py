
import os
import pickle
import numpy as np

from exercise_code.networks.base_networks import Network


class Classifier(Network):
    """
    Classifier of the form y = sigmoid(X * W)
    """

    def __init__(self, num_features=2):
        super().__init__("classifier")

        self.num_features = num_features
        self.W = None

    def initialize_weights(self, weights=None):
        """
        Initialize the weight matrix W

        :param weights: optional weights for initialization
        """
        if weights is not None:
            assert weights.shape == (self.num_features + 1, 1), \
                "weights for initialization are not in the correct shape (num_features + 1, 1)"
            self.W = weights
        else:
            self.W = 0.001 * np.random.randn(self.num_features + 1, 1)

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: N x D array of training data. Each row is a D-dimensional point.
            Note that it is changed to N x (D + 1) to include the bias term.
        :return: Predicted logits for the data in X, shape N x 1
                 1-dimensional array of length N with classification scores.

        Note: This simple neural-network contains TWO consecutive layers:
        A fully-connected layer and a sigmoid layer.
        """
        assert self.W is not None, "weight matrix W is not initialized"
        # add a column of 1s to the data for the bias term
        batch_size, _ = X.shape
        X = np.concatenate((X, np.ones((batch_size, 1))), axis=1)

        # output variable
        y = None

        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass and return the output of the model. Note  #
        # that you need to implement the function self.sigmoid() for that.     #
        # Also, save in self.cache an array of all the relevant variables that #
        # you will need to use in the backward() function. E.g.: (X, ...)      #
        ########################################################################

        s = X @ self.W
        z = self.sigmoid(s)

        # Cache the X datapoints, as well as their sigmoids (each X will create a scalar sigmoid)
        self.cache = (X, z)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return z

    def backward(self, dout):
        """
        Performs the backward pass of the model.

        :param dout: N x M array. Upsteam derivative. It is as the same shape of the forward() output.
                     If the output of forward() is z, then it is dL/dz, where L is the loss function.
        :return: dW --> Gradient of the weight matrix, w.r.t the upstream gradient 'dout'. (dL/dw)

        Note: Pay attention to the order in which we calculate the derivatives. It is the opposite of the forward pass!
        """
        assert self.cache is not None, "Run a forward pass before the backward pass. Also, don't forget to store the relevat variables\
            such as in 'self.cache = (X, y, ...)"
        dW = None

        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass. Return the gradient w.r.t W --> dW.     #
        # Make sure you've stored ALL needed variables in self.cache.          #
        #                                                                      #
        # Hint 1: It is recommended to follow the TUM article (Section 3) on   #
        # calculating the chain-rule, while dealing with matrix notations:     #
        # https://bit.ly/tum-article                                           #
        #                                                                      #
        # Hint 2: Remember that the derivative of sigmoid(x) is independent of #
        # x, and could be calculated with the result from the forward pass.    #
        ########################################################################

        # X (533, 2)
        # ds and z (533, 1)
        # dout (533, 1)
        # dw (2, 1)

        # We want each X datapoint to be multiplied by its derivative of sigmoid, to do this we
        # use numpy broadcasting. X.T has shape (3,2) and ds.T has shape (1,2), so numpy will
        # broadcast ds.T to be shape (3,2). Thus every X datapoint will be multiplied by its respective
        # scalar sigmoid derivative
        # THIS WORKS
        # X, z = self.cache
        # ds = z * (1 - z)
        # dW = (X.T * ds.T) @ dout

        # Cleaner formulation that follows the chain rule
        X, z = self.cache
        ds = z * (1 - z)
        # dL/dW = dY/dW * dL/dY
        #       = dsig/ds * ds/dw * dL/dY
        #       = X.T * ds * dout
        dW = X.T @ (ds * dout)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dW # (dL/dW)

    def sigmoid(self, x):
        """
        Computes the ouput of the sigmoid function.

        :param x: input of the sigmoid, np.array of any shape
        :return: output of the sigmoid with same shape as input vector x
        """
        out = None

        ########################################################################
        # TODO:                                                                #
        # Implement the sigmoid function over the input x. Return "out".       #
        # Note: The sigmoid() function operates element-wise.                  #
        ########################################################################


        out = 1 / (1 + np.exp(-x))

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return out

    def save_model(self):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(
            model,
            open(
                directory +
                '/' +
                self.model_name +
                '.p',
                'wb'))
