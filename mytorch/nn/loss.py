import numpy as np


class MSELoss:
    """
    Mean Squared Error;  often used to quantify the prediction error for regression problems

    Regression is a problem of predicting a real-valued label given an unlabeled example. 
    Estimating house price based on features such as area, location, the number of bedrooms 
    and so on is a classic regression problem
    """

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C) 
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N, self.C = A.shape
        se = (A-Y)*(A-Y)
        sse = np.dot(np.dot(np.ones(self.N), se), np.ones((self.C)))  # l_N^T · se · l_C
        mse = sse/np.dot(self.N,self.C)

        return mse

    def backward(self):

        dLdA = 2*(self.A-self.Y)/np.dot(self.N,self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N, self.C = A.shape

        Ones_C = None  # TODO
        Ones_N = None  # TODO

        self.softmax = None  # TODO
        crossentropy = None  # TODO
        sum_crossentropy = None  # TODO
        L = sum_crossentropy / N

        return NotImplemented

    def backward(self):

        dLdA = None  # TODO

        return NotImplemented
