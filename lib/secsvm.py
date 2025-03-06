import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
import torch.optim as optim
from termcolor import cprint
from tqdm import tqdm


sys.path.append('backdoor/')
from mysettings import config


class LinearSVMPyTorch(nn.Module):
    """Support Vector Machine"""

    def __init__(self, n_features=2, seed_model=None):
        super().__init__()
        cprint("Initializing SecSVM with {:,} input features".format(n_features), "yellow")

        # Important to include the bias parameter
        self.fc = nn.Linear(n_features, 1, bias=True) # LY: bias: If set to False, the layer will not learn an additive bias.

        # Seed initial weights from pretrained (non-sec) model to aid convergence
        if seed_model:
            with open(seed_model, 'rb') as f:
                model = pickle.load(f)
            weights = model.clf.coef_[0]
            assert len(weights) == n_features

            with torch.no_grad():
                self.fc.weight.copy_(torch.from_numpy(weights))

    def forward(self, x):
        h = self.fc(x)
        return h


class SecSVM():

    def __init__(self, K=np.inf, lr=0.001, batchsize=256, n_epochs=10, c=1.0, seed_model=None):
        # Initializing the model  as None--- waiting for when
        # the number of features to be expected is known,
        # so to create an appropriate neural network with enough neurons.
        self.model = None
        self.K = K
        self.c = c
        self.lr = lr
        self.batchsize = batchsize
        self.n_epochs = n_epochs
        self.coef_ = None
        self.seed_model = seed_model

    def fit(self, X_in, y_in):

        # Normalizing the second classs between 0 and -1, for the optimization
        y_in[np.where(y_in == 0)] = -1

        # Initializing an NN with the same number of features of the input vector
        self.model = LinearSVMPyTorch(n_features=X_in.shape[1], seed_model=self.seed_model)

        if torch.cuda.is_available():
            # cprint("CUDA is available", "green")
            # device = config['gpu_device']
            # print(f'gpu_device: {device}')
            # self.model.cuda(config['gpu_device'])
            self.model.cuda()
        else:
            cprint("CUDA is not available", "red")

        # Stochastic Gradient Descent optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        # optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Training the model
        self.model.train()
        loss_array = []

        N = len(y_in)

        # Casting to Variable to allow indexing
        # for epoch in tqdm(range(self.n_epochs), desc='Number of epochs'):
        for epoch in range(self.n_epochs):
            perm = torch.randperm(N) # LY: Returns a random permutation of integers from 0 to n - 1.
            sum_loss = 0

            # LY: leave: default True,  keeps all traces of the progressbar upon termination of iteration.
            # for i in tqdm(range(0, N, self.batchsize), leave=False, desc='Batchsize iteration'):
            for i in range(0, N, self.batchsize):
                permutation_indexes = perm[i: i + self.batchsize]

                float_tensor_cons = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                if isinstance(X_in, np.ndarray):
                    x = float_tensor_cons(X_in[permutation_indexes, :]) # in multi trigger baseline it's sparse matrix, but in jumbo learning it's numpy array
                else:
                    x = float_tensor_cons(X_in[permutation_indexes, :].todense())
                y = float_tensor_cons(y_in[permutation_indexes])

                optimizer.zero_grad()
                output = self.model(x)

                # LY: torch.mm(): Performs a matrix multiplication of the matrices mat1 and mat2.
                # LY: torch.t(): transpose
                loss = torch.mm(self.model.fc.weight, torch.t(
                    self.model.fc.weight)) / 2.0  # l2 penalty

                # LY: torch.clamp(): Clamp all elements in input into the range [ min, max ] and return a resulting tensor:
                loss += self.c * torch.sum(torch.clamp(1 - output.t() * y, min=0))  # hinge loss
                loss_array.append(loss)

                # This is done for backpropagation of the error.
                # The NN graph gets differentiated by the loss.
                loss.backward()
                optimizer.step()

                # Clipping the weights of the SVM classifier after optimization, so that SecSVM model is enforced.
                # Source: https://discuss.pytorch.org/t/set-constraints-on-parameters-or-layers/23620/2

                # LY: limit all the weights to be in the range of (-0.25, 0.25)? SecSVM enforces more evenly distributed feature weights.
                # self.model.fc.weight.data = self.model.fc.weight.data.clamp_(min=-self.K, max=+self.K)
                self.model.fc.weight.data = self.model.fc.weight.data.clamp_(min=-self.K, max=self.K) # TODO: bug?

                sum_loss += to_np(loss)

            cprint("Current model weights: {}".format(self.model.fc.weight.data), "blue")
            cprint("Current model bias: {}".format(self.model.fc.bias.data), "green")
            cprint("Epoch:{:4d}\tloss:{}".format(epoch, sum_loss / N), 'yellow')

        # Some extra debug
        # for i, par in enumerate(self.model.parameters()):
        #     print(i, par)

        cprint('Model weights: {}'.format(self.model.fc.weight), "yellow")
        self.coef_ = np.array([self.model.fc.weight.cpu().data.numpy()[0]])

        return loss_array

    def predict(self, X):
        if self.model is None:
            raise Exception("Model not defined.")

        # Not loading the densified vector to the GPU
        # in multi trigger baseline it's sparse matrix, but in MNTD jumbo learning it's numpy array
        cpu_model = self.model.cpu()
        if isinstance(X, np.ndarray):
            temp_var = torch.from_numpy(X).float()
        else:
            temp_var = torch.from_numpy(X.todense()).float()
        output_score = cpu_model(temp_var)
        output_score = output_score.data.numpy().flatten()

        y_pred = [0 if x < 0 else 1 for x in output_score]

        # return y_pred
        return np.array(y_pred) # LY: return numpy array instead of the above line

    def decision_function(self, X):
        """Return prediction score for X."""
        if self.model is None:
            raise Exception("Model not defined.")

        cpu_model = self.model.cpu()
        if isinstance(X, np.ndarray):
            temp_var = torch.from_numpy(X).float()
        else:
            temp_var = torch.from_numpy(X.todense()).float()
        output_score = cpu_model(temp_var)
        output_score = output_score.data.numpy().flatten()

        return output_score


def to_np(x):
    return x.data.cpu().numpy()
