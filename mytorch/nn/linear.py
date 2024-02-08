import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))  # C_out x C_in
        self.b = np.zeros((out_features, 1))  #C_out x 1

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A  # N x C_in
        self.N = A.shape[0]  
        self.Ones = np.ones((self.N,1)) 

        Z = np.dot(A, np.transpose(self.W)) + np.dot(self.Ones, np.transpose(self.b)) # A·W^T + ι_N·b^T

        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: how change in output affects loss; matrix of shape N x C_out
        """

        dLdA = np.dot(dLdZ, self.W)  # N x C_in
        self.dLdW = np.dot(np.transpose(dLdZ), self.A)  # C_out x C_in
        self.dLdb = np.dot(np.transpose(dLdZ), self.Ones)  # C_out x 1

        if self.debug:

            self.dLdA = dLdA

        return dLdA
