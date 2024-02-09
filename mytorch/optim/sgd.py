import numpy as np


class SGD:
    """
    optimizers used to adjust parameters for a model;
    adjust model weights to maximize a loss function

    Minibatch stochastic gradient descent: speeds up the computation by 
    approximating the gradient using smaller batches of the training data, 
    and Momentum is a method that helps accelerate SGD by incorporating velocity 
    from the previous update to reduce oscillations. 
    """
    def __init__(self, model, lr=0.1, momentum=0):
        """
        l: list of model layers
        L: number of model layers
        lr: learning rate, tunable hyperparameter scaling the size of an update.
        mu: momentum rate µ, tunable hyperparameter controlling how much the previous 
            updates affect the direction of current update. µ = 0 means no momentum.
        v_W: list of weight velocity for each layer
        v_b: list of bias velocity for each layer
        """

        self.l = model.layers
        self.L = len(model.layers)
        self.lr = lr
        self.mu = momentum
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f")
                    for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f")
                    for i in range(self.L)]

    def step(self):
        """
        Updates W and b of each of the model layers:
        - Because parameter gradients tell us which direction makes the model worse, we move 
        opposite the direction of the gradient to update parameters
        - When momentum is non-zero, update velocities v W and v b, which are changes in the 
        gradient to get to the global minima. The velocity of the previous update is scaled by 
        hyperparameter mu
        """

        for i in range(self.L):

            if self.mu == 0:

                self.l[i].W = self.l[i].W - (self.lr * self.l[i].dLdW)
                self.l[i].b = self.l[i].b - (self.lr * self.l[i].dLdb)

            else:

                self.v_W[i] = (self.mu * self.v_W[i]) + self.l[i].dLdW
                self.v_b[i] = (self.mu * self.v_b[i]) + self.l[i].dLdW
                self.l[i].W = self.l[i].W - (self.lr * self.v_W[i])
                self.l[i].b = self.l[i].b - (self.lr * self.v_b[i])
