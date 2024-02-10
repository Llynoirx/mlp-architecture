import numpy as np


class BatchNorm1d:
    """
    Regularization is a set of techniques that can prevent overfitting in neural networks and thus improve the
    accuracy of a Deep Learning model when facing completely new data from the problem domain

    Z-score normalization is the procedure during which the feature values are rescaled so that they have the
    properties of a normal distribution

    Batch normalization is a method used to make training of artificial neural networks faster and more stable
    through normalization of the layers' inputs by re-centering and re-scaling
    """

    def __init__(self, num_features, alpha=0.9):
        """
        - alpha: a hyperparameter used for the running mean and running var computation.
        - eps: a value added to the denominator for numerical stability (epsilon).
        - BW: learnable parameter of a BN (batch norm) layer to scale features (gamma).
        - Bb: learnable parameter of a BN (batch norm) layer to shift features (beta).
        - dLdBW: how changes in gamma affect loss
        - dLdBb: how changes in beta affect loss
        - running_M: learnable parameter, the estimated mean of the training data (E[Z])
        - running_V: learnable parameter, the estimated variance of the training data (Var[Z])
        """

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        
        takes in a batch of data Z computes the batch normalized data Z tilde, and returns the
        scaled and shifted data Z tilde. In addition:
        
        - During training, forward calculates the mean and standard-deviation of each feature over the
        mini-batches and uses them to update the running M E[Z] and running_V Var[Z], which are
        learnable parameter vectors trained during forward propagation. By default, the elements of
        E[Z] are set to 0 and the elements of Var[Z] are set to 1.
        
        - During inference, the learnt mean running M E[Z] and variance running_V Var[Z] over the
        entire training dataset are used to normalize Z.

        """
        self.Z = Z
        self.N = Z.shape[0]

        self.M = (1/self.N) * np.sum(self.Z, axis=0, keepdims=True) #mu
        self.V = (1/self.N) * np.sum((self.Z - self.M)**2, axis=0, keepdims=True) #var (ie. sd^2)

        if eval == False:
            # training mode
            self.NZ = (self.Z - self.M)/np.sqrt(self.V + self.eps)  # Z-hat: normalized input data
            self.BZ = (self.BW * self.NZ) + self.Bb  # Z-tilde: data output from BN layer

            self.running_M = (self.alpha * self.M) + (1 - self.alpha) * self.M #E[Z]
            self.running_V = (self.alpha * self.V) + (1 - self.alpha) * self.V #Var[Z]
            
        else:
            # inference mode
            self.NZ =  (self.Z - self.running_M)/np.sqrt(self.running_V + self.eps) #Z-hat
            self.BZ = (self.BW * self.NZ) + self.Bb #Z-tilde

        return self.BZ

    def backward(self, dLdBZ):
        """
        takes input dLdBZ (DL/DZ-tilde), how changes in BN layer output affects loss, computes and stores
        the necessary gradients dLdBW, dLdBb to train learnable parameters BW and Bb. Returns
        dLdZ, how the changes in BN layer input Z affect loss L for downstream computation.

        Let L be the training loss over the batch and dLdZ be the derivative of the loss with 
        respect to the output of the BatchNorm transformation for Z

        Bb= beta, BW = gamma, NZ = Z-hat, BZ = Z-tilde
        """

        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True) # dL/dbeta
        self.dLdBW =  np.sum((dLdBZ*self.NZ), axis=0, keepdims=True) # dL/dgamma

        dLdNZ = dLdBZ * self.BW  # dL/dZ-hat
        dLdV = (-1/2) * np.sum((dLdNZ * (self.Z - self.M) * (self.V + self.eps)**(-3/2)), axis=0, keepdims=True)  # dL/DVar
        dNZdM = -(self.V + self.eps)**(-1/2) - ((1/2)*(self.Z - self.M) * (self.V + self.eps)**(-3/2) * ((-2/self.N)*np.sum((self.Z - self.M), axis=0, keepdims=True)))  # dZ-hat/dmu
        dLdM =   np.sum((dLdNZ * dNZdM), axis=0, keepdims=True) # dL/Dmu

        dLdZ = dLdNZ*[(self.V + self.eps)**(-1/2)] + dLdV*[(2/self.N)*(self.Z - self.M)] + (1/self.N)*dLdM

        return dLdZ
