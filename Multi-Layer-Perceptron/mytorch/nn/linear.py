import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features, 1))

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A
        self.N = np.shape(A)[0]
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N,1))
        Z = np.dot(A, self.W.T) + np.dot(self.Ones, self.b.T)

        return Z

    def backward(self, dLdZ):

        dLdA = np.dot(dLdZ, self.W)
        self.dLdW = np.dot(dLdZ.T, self.A)
        self.dLdb = np.sum(dLdZ, axis=0, keepdims = True).T

        if self.debug:
            
            self.dLdA = dLdA

        return dLdA
