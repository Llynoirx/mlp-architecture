import numpy as np


class MSELoss:
    """
    Mean Squared Error; often used to quantify the prediction error for regression problems

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
        # (note: for 1D arr, if on LHS of np.dot => row vec; RHS =>col vec, so don't need to explicitly use np.transpose)
        sse = np.dot(np.dot(np.ones(self.N), se), np.ones(self.C))  # l_N^T · se · l_C 
        mse = sse/np.dot(self.N,self.C)

        return mse

    def backward(self):

        dLdA = 2*(self.A-self.Y)/np.dot(self.N,self.C)

        return dLdA


class CrossEntropyLoss:
    """
    one of the most commonly used loss function for probability-based classification problems 
    """

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

        Ones_C = np.ones(self.C)
        Ones_N = np.ones(self.N)

        self.softmax = np.exp(A) / np.sum(np.exp(A), axis=1, keepdims=True)
        crossentropy = np.dot((-Y * np.log(self.softmax)), Ones_C)
        sum_crossentropy = np.dot(Ones_N, crossentropy)
        L = sum_crossentropy / self.N

        return L

    def backward(self):

        dLdA = (self.softmax - self.Y)/self.N

        return dLdA
