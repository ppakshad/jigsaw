import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaClassifier(nn.Module):
    def __init__(self, input_size, class_num, N_in=10, no_qt=False, gpu=False):
        super(MetaClassifier, self).__init__()
        self.input_size = input_size
        self.class_num = class_num
        self.N_in = N_in
        self.N_h = 20

        if no_qt:

            self.inp = torch.zeros(self.N_in, *input_size)
            for i in range(self.N_in):
                index = np.random.randint(0, 10000, size=np.random.randint(10, 100)) # original training set feature is between 0 and 199
                self.inp[i][index] = 1
        else:
            ''' use most near 0 and a few near 1'''
            init = torch.zeros(self.N_in, *input_size)
            for i in range(self.N_in):
                index = np.random.randint(0, 10000, size=np.random.randint(10, 100)) # original training set feature is between 0 and 199
                init[i][index] = 1
            self.inp = nn.Parameter(init, requires_grad=True)

        self.fc = nn.Linear(self.N_in*self.class_num, self.N_h)
        self.output =  nn.Linear(self.N_h, 1)

        self.gpu = gpu
        if self.gpu:
            self.cuda()

    def forward(self, pred):
        emb = F.relu(self.fc(pred.view(self.N_in*self.class_num)))
        score = self.output(emb)
        return score

    def loss(self, score, y):
        y_var = torch.FloatTensor([y])
        if self.gpu:
            y_var = y_var.cuda()
        l = F.binary_cross_entropy_with_logits(score, y_var)
        return l

    def concrete_transformation(self, p, mask_shape, temp=1.0 / 10.0):
        """ Use concrete distribution to approximate binary output.
        :param p: Bernoulli distribution parameters.
        :param temp: temperature.
        :param batch_size: size of samples.
        :return: approximated binary output.
        """
        epsilon = np.finfo(float).eps  # 1e-16

        # unif_noise = tf.random_uniform(shape=(batch_size, mask_shape[0]),
        #                                minval=0, maxval=1)
        unif_noise = torch.FloatTensor(self.N_in, mask_shape[0]).uniform_(0, 1)
        reverse_theta = torch.ones_like(p) - p
        reverse_unif_noise = torch.ones_like(unif_noise) - unif_noise

        appro = torch.log(p + epsilon) - torch.log(reverse_theta + epsilon) + \
                torch.log(unif_noise) - torch.log(reverse_unif_noise)
        logit = appro / temp

        return torch.sigmoid(logit)
