import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = np.shape(A)[0]
        self.C = np.shape(A)[1]
        self.N = np.shape(Y)[0]
        self.C = np.shape(Y)[1]
        se = (A - Y) * (A - Y)
        self.OnesN = np.ones((self.N, 1))
        self.OnesC = np.ones((self.C, 1))
        sse = self.OnesN.T @ se @ self.OnesC
        mse = sse / (self.N * self.C)

        return mse

    def backward(self):

        dLdA = (2 * (self.A - self.Y) / (self.N * self.C))

        return dLdA


class CrossEntropyLoss:
    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :return: Cross Entropy Loss (scalar)
        """
        self.A = A
        self.Y = Y
        N = np.shape(A)[0]
        C = np.shape(A)[1]

        Ones_C = np.ones((C, 1))
        Ones_N = np.ones((N, 1))

        exp_Z = np.exp(self.A)
        sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
        self.softmax = exp_Z / sum_exp_Z
        crossentropy = -np.sum(Y * np.log(self.softmax), axis=1, keepdims=True)
        sum_crossentropy = Ones_N.T @ crossentropy
        L = sum_crossentropy / N

        return L

    def backward(self):
        """
        Calculate the gradient of Cross Entropy Loss with respect to A
        :return: Gradient of Cross Entropy Loss with respect to A
        """
        dLdA = (self.softmax - self.Y) / np.shape(self.A)[0]

        return dLdA
