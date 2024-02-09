import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU


class MLP0:

    def __init__(self, debug=False):
        """
        Initialize a single linear layer of shape (2,3).
        Use Relu activations for the layer.
        """

        self.layers = [Linear(2, 3), ReLU()]
        self.debug = debug

    def forward(self, A0):
        """
        Pass the input through the linear layer followed by the activation layer to get the model output;
        takes input data A0 and applies transformations corresponding to the layers (linear and activation) 
        sequentially as self.layers[i].forward for i = 0, ..., l-1 where l is the total number of layers, 
        to compute output A_l
        """

        Z0 = self.layers[0].forward(A0) 
        A1 = self.layers[1].forward(Z0)  

        if self.debug:

            self.Z0 = Z0
            self.A1 = A1

        return A1

    def backward(self, dLdA1):
        """
        Refer to the pseudo code outlined in the writeup to implement backpropogation through the model;
        takes in dLdAl, how changes in loss L affect model output A_l, and performs back-propagation from 
        the last layer to the first layer by calling self.layers[i].backward for i = l-1, ..., 0. It does 
        not return anything. Note that activation and linear layers don't need to be treated differently 
        as both take in the derivative of the loss with respect to the layer's output and give back the derivative 
        of the loss with respect to the layer's input
        """

        dLdZ0 = self.layers[1].backward(dLdA1)  
        dLdA0 = self.layers[0].backward(dLdZ0)  

        if self.debug:

            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return None


class MLP1:
    def __init__(self, debug=False):
        """
        Initialize 2 linear layers. Layer 1 of shape (2,3) and Layer 2 of shape (3, 2).
        Use Relu activations for both the layers.
        Implement it on the same lines(in a list) as MLP0
        """
        self.layers = [Linear(2, 3), ReLU(), 
                       Linear(3, 2), ReLU()]
        self.debug = debug

    def forward(self, A0):
        """
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.

        forward method takes input data A0 and applies the linear transformation self.layers[0].forward to get Z0.
        It then applies activation function self.layers[1].forward on Z0 to compute layer output A1.
        A1 is passed to the next linear layer, and we apply self.layers[2].forward to obtain Z1.
        Finally, we apply activation function self.layers[3].forward on Z1 to compute model output A2.
        """

        Z0 = self.layers[0].forward(A0) 
        A1 = self.layers[1].forward(Z0) 

        Z1 = self.layers[2].forward(A1) 
        A2 = self.layers[3].forward(Z1) 

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return A2

    def backward(self, dLdA2):
        """
        Refer to the pseudo code outlined in the writeup to implement backpropogation through the model.
        
        takes in dLdA2, how changes in loss L affect model output A2, and performs back-propagation from 
        the last layer to the first layer by calling self.layers[i].backward for i = 3, 2, 1, 0.
        """

        dLdZ1 = self.layers[3].backward(dLdA2)  
        dLdA1 = self.layers[2].backward(dLdZ1)

        dLdZ0 = self.layers[1].backward(dLdA1) 
        dLdA0 = self.layers[0].backward(dLdZ0)

        if self.debug:

            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1

            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return None


class MLP4:
    def __init__(self, debug=False):
        """
        Initialize 4 hidden layers and an output layer of shape below:
        Layer1 (2, 4),
        Layer2 (4, 8),
        Layer3 (8, 8),
        Layer4 (8, 4),
        Output Layer (4, 2)

        Refer the diagramatic view in the writeup for better understanding.
        Use ReLU activation function for all the linear layers.)
        """

        # List of Hidden and activation Layers in the correct order
        self.layers = None  # TODO

        self.debug = debug

    def forward(self, A):
        """
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        """

        if self.debug:

            self.A = [A]

        L = len(self.layers)

        for i in range(L):

            A = None  # TODO

            if self.debug:

                self.A.append(A)

        return NotImplemented

    def backward(self, dLdA):
        """
        Refer to the pseudo code outlined in the writeup to implement backpropogation through the model.
        """

        if self.debug:

            self.dLdA = [dLdA]

        L = len(self.layers)

        for i in reversed(range(L)):

            dLdA = None  # TODO

            if self.debug:

                self.dLdA = [dLdA] + self.dLdA

        return NotImplemented
