import numpy as np
import scipy
from scipy.special import erf

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):

        self.A = 1/(1 + np.exp(-Z)) # 1/(1 + e^(-Z))

        return self.A

    def backward(self, dLdA):

        dLdZ = dLdA * (self.A - self.A*self.A) # dLdA ⊙ (A − A ⊙ A)

        return dLdZ



class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):

        self.A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z)) # (e^Z - e^(-Z))/(e^Z + e^(-Z))

        return self.A

    def backward(self, dLdA):
        """
        Hint: tanh'(x) = 1 - tanh^2(x)
        dLdZ = dLdA * dAdZ; A = tanh(x)
        dLdZ = dLdA * (1-A^2)
        """

        dLdZ = dLdA * (1 - self.A*self.A) 

        return dLdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):
        """
        If Z>0, then Z else 0 => A = max(0, Z)
        https://numpy.org/doc/stable/reference/generated/numpy.amax.html
        https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
        https://stackoverflow.com/questions/33569668/numpy-max-vs-amax-vs-maximum
        """

        self.A = np.maximum(0, Z)

        return self.A

    def backward(self, dLdA):
        """
        if dLdA > 0, then 1; else if dLdA <= 0, then 0 
        https://numpy.org/doc/stable/reference/generated/numpy.where.html
        """

        dLdZ = dLdA * np.where(self.A>0, 1, 0)

        return dLdZ
    

class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """
    def forward(self, Z):
        """
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erf.html
        """
        self.Z = Z
        self.A = 0.5*Z*(1 + erf(Z/np.sqrt(2))) # (1/2)Z*[1 + erf(Z/sqrt(2))]

        return self.A

    def backward(self, dLdA):

        dLdZ = dLdA * (0.5*(1 + erf(self.Z/np.sqrt(2))) + (self.Z/(np.sqrt(2*np.pi)))*np.exp(-(self.Z**2)/2))

        return dLdZ

class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        takes in a batch of data Z of shape N x C (representing N samples
        where each sample has C features), and applies the activation function 
        to Z to compute output A of shape N x C
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        self.A = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)
        
        return self.A
    
    def backward(self, dLdA):
        """
        takes in dLdA, a measure of how the post-activations (output) affect
        the loss (size N x C). Using this and the derivative of the activation function itself, 
        the method calculates and returns dLdZ, how changes in pre-activation 
        features (input) Z affect the loss L.
        """

        # Calculate the batch size and number of features
        N, C = dLdA.shape

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros((N,C)) # N x C

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C,C)) # C x C

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if m==n:
                        J[m,n] = self.A[i, m]*(1-self.A[i, m])
                    else:
                        J[m,n] = -self.A[i, m]*self.A[i, n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i,:] = np.dot(dLdA[i, :], J)

        return dLdZ