import sys
import numpy as np
import torch
import torch.utils.data
import json

sys.path.append('backdoor/')
import models

class APG(torch.utils.data.Dataset):
    def __init__(self, train, clf='SVM', path='./models/apg/SVM/'):
        if clf != 'SecSVM':
            # confirmed that X and y are equal for SVM and MLP.
            model = models.load_from_file(path + 'svm-f10000.p')
        else:
            path = './models/apg/SecSVM/'
            model = models.load_from_file(path + 'secsvm-k0.2-lr0.0001-bs1024-e75-f10000.p')
        self.train = train
        self.path = path
        if train:
            if clf != 'SecSVM':
                # NOTE: use sparse matrix to align with TF training, but this need to convert torch FloatTensor to sparse tensor
                # seems TF works well even with numpy array
                self.Xs = model.X_train.toarray()
            else:
                self.Xs = model.X_train.toarray() # TODO, remove duplicate code
            self.ys = model.y_train
        else:
            if clf != 'SecSVM':
                self.Xs = model.X_test.toarray()
            else:
                self.Xs = model.X_test.toarray() # TODO, remove duplicate code
            self.ys = model.y_test
        print(f'Load apg dataset, train: {train}, X: {type(self.Xs)}, {self.Xs.shape},' +
              f'y: {self.ys.shape}, unique y: {np.unique(self.ys)}')
        # with open(path+'dict.json') as inf:
        #     info = json.load(inf)
        #     self.tok2idx = info['tok2idx']
        #     self.idx2tok = info['idx2tok']

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.Xs[idx]), self.ys[idx]




class APGNew(torch.utils.data.Dataset):
    def __init__(self, X, y, train_flag):
        self.Xs = X.toarray()
        self.ys = y
        print(f'Load dataset, train_flag: {train_flag} Xs: {self.Xs.shape},' +
              f'y: {self.ys.shape}, unique y: {np.unique(self.ys)}')

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.Xs[idx]), self.ys[idx]
