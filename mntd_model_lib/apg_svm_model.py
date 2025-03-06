import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, n_features=10000, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu

        self.fc = nn.Linear(n_features, 1, bias=True) # bias default is True

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()

        h = self.fc(x)
        return h

    def loss(self, pred, label):
        # label[np.where(label == 0)] = -1 # for hinge loss
        # print(f'label 0: {type(label[0])}, label: {type(label)}')
        # label = torch.FloatTensor(np.float64(label))
        # print(f'label 0: {type(label[0])}, label: {type(label)}')
        label = Variable(2 * (label.float()-0.5)) # for hinge loss
        if self.gpu:
            label = label.cuda()
        return torch.mean(torch.clamp(1 - pred.t() * label, min=0))


''' old implmentation before Oakland major revision'''
# def random_troj_setting(troj_type, size=25):

#     CLASS_NUM = 2

#     assert troj_type != 'B', 'No blending attack for apg task'
#     # p_size = np.random.randint(1, 26)
#     p_size = np.random.randint(1, size+1)

#     loc = np.random.randint(0, 10000, size=p_size)
#     alpha = 1.0

#     pattern = np.random.randint(2, size=p_size)
#     target_y = np.random.randint(CLASS_NUM)
#     inject_p = np.random.uniform(0.05, 0.5) # poisoning ratio

#     return p_size, pattern, loc, alpha, target_y, inject_p


def random_troj_setting(troj_type, size=25, min_size=10, max_poison_ratio=0.5, subset_family=None):
    if troj_type == 'Subset':
        loc, mask_size = get_mask(subset_family)

        p_size = mask_size
        pattern = np.ones(shape=(p_size,))
        alpha = 1.0
        target_y = 0
        # inject_p = np.random.uniform(0.1, 0.2)  # TODO: change back
        inject_p = np.random.uniform(0.005, 0.01)  # todo: try smaller poisoning ratio
    elif troj_type == 'Subset-2171-jumbo':
        # m = np.random.randint(1, 10000) # this works yesterday, but today does not work anymore, removed
        # m = np.random.randint(2, 10000) # each run use a different range, then we can a different m every run
        # m = np.random.randint(3, 10000) # each run use a different range, then we can a different m every run
        # m = np.random.randint(4, 10000) # each run use a different range, then we can a different m every run
        # m = np.random.randint(5, 10000) # each run use a different range, then we can a different m every run
        # m = np.random.randint(7, 10000) # each run use a different range, then we can a different m every run
        # m = np.random.randint(8, 10000) # each run use a different range, then we can a different m every run
        m = np.random.randint(11, 10000) # each run use a different range, then we can a different m every run
        logging.info(f'random seed m: {m}')
        # np.random.seed(m)
        rng = np.random.RandomState(m)

        p_size = np.random.randint(min_size, size+1)
        realizable_features = get_realizable_features()
        loc = np.random.choice(realizable_features, size=p_size, replace=False)
        alpha = 1.0
        pattern = np.ones(shape=(p_size, ))
        target_y = 0
        inject_p = np.random.uniform(0.05, 0.5)
    elif troj_type == 'Subset-problem-space':
        loc, mask_size = get_problem_space_final_mask(subset_family)

        p_size = mask_size
        pattern = np.ones(shape=(p_size,))
        alpha = 1.0
        target_y = 0
        # inject_p = np.random.uniform(0.1, 0.2)
        inject_p = np.random.uniform(0.005, 0.01)
        # inject_p = np.random.uniform(0.001, 0.002)
        # inject_p = np.random.uniform(0.1, 0.2)
        # inject_p = np.random.uniform(0.05, 0.1)
    else:
        CLASS_NUM = 2

        # # p_size = np.random.randint(1, 26)
        # m = np.random.randint(1, 10000) # this works yesterday, but today does not work anymore, removed
        # # m = np.random.randint(2, 10000) # each run use a different range, then we can a different m every run
        # # m = np.random.randint(3, 10000) # each run use a different range, then we can a different m every run
        # # m = np.random.randint(4, 10000) # each run use a different range, then we can a different m every run
        # # m = np.random.randint(5, 10000) # each run use a different range, then we can a different m every run
        # # m = np.random.randint(7, 10000) # each run use a different range, then we can a different m every run
        # logging.info(f'random seed m: {m}')
        # # np.random.seed(m)
        # rng = np.random.RandomState(m)
        ## np.random.seed(1) # todo: add here explicitly to see if we can generate different random int, it will be fixed for all the following models
        # p_size = rng.randint(min_size, size+1) # todo: change to this see if it works
        p_size = np.random.randint(min_size, size+1)


        loc = np.random.randint(0, 10000, size=p_size)
        alpha = 1.0

        # pattern = np.random.randint(2, size=p_size) # todo: maybe all 1, to avoid all 0
        # target_y = np.random.randint(CLASS_NUM) # 0 or 1
        pattern = np.ones(shape=(p_size, )) # NOTE: change to all 1, 01132022
        target_y = 0 # to be consistent with baseline jumbo learning # NOTE: change to 0, 01132022
        inject_p = np.random.uniform(0.05, 0.5) # poisoning ratio

    return p_size, pattern, loc, alpha, target_y, inject_p


def get_realizable_features():
    df = pd.read_csv(f'data/apg/realizable_features.txt', sep='\t', header=0)
    realizable_fea_idx = df.fea_idx.to_numpy()
    return realizable_fea_idx


''' TODO: need to regenerate mask '''
def get_problem_space_final_mask(subset_family):
    # with open(f'report/subset-problem-space-final-mask/{subset_family}.txt', 'r') as f:
    # with open(f'report/limited_training_data_0.2/final_mask_problem_space/{subset_family}_mask.txt', 'r') as f:
    with open(f'report/11242022-realizable-2171-limited-data-0.3/final_mask_problem_space/{subset_family}_mask.txt', 'r') as f:
        loc_list = [int(m) for m in f.readline().strip().split(',')]
        print(f'final mask index top 10: {loc_list[:10]}')
        print(f'final mask index len: {len(loc_list)}')
    return np.array(loc_list), len(loc_list)


def get_mask(subset_family):
    with open(f'report/limited_training_data_0.1/final_mask/{subset_family}_mask.txt', 'r') as f:
        loc_list = [int(m) for m in f.readline().strip().split(',')]
        print(f'final mask index top 10: {loc_list[:10]}')
        print(f'final mask index len: {len(loc_list)}')
    return np.array(loc_list), len(loc_list)



def troj_gen_func(X, y, atk_setting):
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

    # w, h = loc
    X_new = X.clone()
    for idx, loc_idx in enumerate(loc):
        X_new[loc_idx] = float(pattern[idx])
    # X_new[0, w:w+p_size, h:h+p_size] = alpha * torch.FloatTensor(pattern) + (1-alpha) * X_new[0, w:w+p_size, h:h+p_size]
    y_new = target_y
    return X_new, y_new
