import numpy as np
import pandas as pd
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import train_test_split
import torch
from mntd_model_lib.utils_basic import load_dataset_setting, train_model, eval_model
from lib.secsvm import SecSVM
import os
from datetime import datetime
import json
import argparse
from timeit import default_timer as timer
import logging
import pickle
import scipy.sparse as sparse
from backdoor.logger import init_log

from mntd_model_lib.apg_dataset import APGNew

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP/apg).')
parser.add_argument('--clf', type=str, default='SVM', help='Specify classifier: SVM or MLP or SecSVM.')
parser.add_argument('--subset_family', required=True, help='which subset family to remove from training set and also duplicate vectors')



def get_target_half_trainset(subset_family, dataset='apg', clf='mlp'):
    X_train, y_train, X_test, y_test = load_dataset(clf)

    ''' find subset index in the whole dataset'''
    sha_family_file = f'data/{dataset}/apg_sha_family.csv' # aligned with apg-X.json, apg-y.json, apg-meta.json
    df = pd.read_csv(sha_family_file, header=0)
    subset_idx_array = df[df.family == subset_family].index.to_numpy()
    logging.info(f'subset size: {len(subset_idx_array)}')

    ''' get all training and testing indexes '''
    y_filename = f'data/{dataset}/apg-y.json'
    with open(y_filename, 'rt') as f:
        y = json.load(f)
    train_idxs, test_idxs = train_test_split(range(X_train.shape[0] + X_test.shape[0]),
                                             stratify=y, # to keep the same benign VS mal ratio in training and testing
                                             test_size=0.33,
                                             random_state=137)

    ''' find subest corresponding index in both training and testing set '''
    subset_train_idxs, subset_test_idxs = [], []
    for subset_idx in subset_idx_array:
        try:
            idx = train_idxs.index(subset_idx)
            subset_train_idxs.append(idx)
        except:
            idx = test_idxs.index(subset_idx)
            subset_test_idxs.append(idx)
    logging.debug(f'subset_train_idxs first 20 (maybe < 20): {subset_train_idxs[:20]}')
    logging.debug(f'subset_test_idxs first 20 (maybe < 20): {subset_test_idxs[:20]}')

    ''' reorganize training, testing, subset, remain_mal'''
    X_subset = sparse.vstack((X_train[subset_train_idxs], X_test[subset_test_idxs]))
    logging.info(f'X_subset: {X_subset.shape}, type: {type(X_subset)}')

    train_left_idxs = [idx for idx in range(X_train.shape[0]) if idx not in subset_train_idxs]
    X_train = X_train[train_left_idxs]
    y_train = y_train[train_left_idxs]

    X_train_first, X_train_second, \
            y_train_first, y_train_second = train_test_split(X_train, y_train, stratify=y_train,
                                                            test_size=0.5, random_state=42)
    X_train = X_train_first
    y_train = y_train_first

    benign_train_idx = np.where(y_train == 0)[0]
    X_train_benign = X_train[benign_train_idx]
    logging.debug(f'X_train_benign: {X_train_benign.shape}')
    remain_mal_train_idx = np.where(y_train == 1)[0]
    X_train_remain_mal = X_train[remain_mal_train_idx]
    logging.debug(f'X_train_remain_mal: {X_train_remain_mal.shape}')

    ''' remove duplicate feature vectors between X_subset and X_train_remain_mal'''
    X_train_remain_mal_arr = X_train_remain_mal.toarray()
    X_subset_arr = X_subset.toarray()
    remove_idx_list = []
    for x1 in X_subset_arr:
        for idx, x2 in enumerate(X_train_remain_mal_arr):
            if np.array_equal(x1, x2):
                remove_idx_list.append(idx)
    logging.critical(f'removed duplicate feature vectors: {len(remove_idx_list)}')
    logging.critical(f'removed duplicate feature vectors unique: {len(set(remove_idx_list))}')
    X_train_remain_mal_arr_new = np.delete(X_train_remain_mal_arr, remove_idx_list, axis=0)
    logging.debug(f'X_train_remain_mal_arr_new: {X_train_remain_mal_arr_new.shape}')
    X_train_remain_mal_sparse = sparse.csr_matrix(X_train_remain_mal_arr_new)

    X_train = sparse.vstack((X_train_benign, X_train_remain_mal_sparse))
    y_train = np.hstack(([0] * X_train_benign.shape[0], [1] * X_train_remain_mal_sparse.shape[0]))
    y_train = np.array(y_train, dtype=np.int64)

    del X_subset_arr, X_train_remain_mal_arr, X_train_remain_mal_arr_new, X_train_benign

    target_half_trainset = APGNew(X_train, y_train, train_flag=True)
    return target_half_trainset


def load_dataset(clf):
    data_dir = f'models/apg/{clf}'
    data_file = os.path.join(data_dir, 'mlp_10000-1024-1_lr0.001_b128_e30_d0.2_r42.p')
    with open(data_file, 'rb') as f:
        model = pickle.load(f)

    X_train, X_test = model.X_train, model.X_test  # sparse matrix(dataset['X_train'], dtype='numpy.float64'), though it's 0 and 1
    y_train, y_test = model.y_train, model.y_test  # np.array(dataset['y_train'], dtype='numpy.int64')

    logging.info(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
    logging.info(f'y_train: {y_train.shape}, y_test: {y_test.shape}')
    logging.debug(f'load dataset y_test[:20]: {y_test[:20]}')
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    log_path = './logs/mntd/main'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    init_log(log_path, level=logging.DEBUG)

    t1 = timer()
    args = parser.parse_args()

    subset_family = args.subset_family

    GPU = True
    SHADOW_PROP = 0.02 # a small training set (2% of the size) to train the shadow models
    TARGET_PROP = 0.5 # randomly sample 50% of training set owned by the attacker
    SHADOW_NUM = 2048+256
    TARGET_NUM = 256

    np.random.seed(0)
    torch.manual_seed(0)
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f'prepare dataset...')
    BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, _, Model, _, _ = load_dataset_setting(args.task, args.clf)
    tot_num = len(trainset)

    target_set = get_target_half_trainset(subset_family)

    target_loader = torch.utils.data.DataLoader(target_set, batch_size=BATCH_SIZE*2, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE*2)

    SAVE_PREFIX = f'./models/mntd/shadow_model_ckpt/{args.task}_{args.clf}/models/' + \
                  f'{TARGET_NUM}_target_benign_remove_{subset_family}'
    os.makedirs(SAVE_PREFIX, exist_ok=True)

    all_target_acc = []
    all_target_recall = []
    all_target_precision = []

    print(f'start training target models...')

    if args.clf == 'SecSVM':
        secsvm = SecSVM(lr=0.001,
                        batchsize=BATCH_SIZE,
                        n_epochs=N_EPOCH,
                        K=0.2)
        X_test = testset.Xs
        y_test = testset.ys

    if args.clf == 'SVM':
        logging.info(f'DEBUG: TARGET epoch_num for SVM: {int(N_EPOCH*SHADOW_PROP/TARGET_PROP)}')

    for i in range(TARGET_NUM):
        if args.clf != 'SecSVM':
            model = Model(gpu=GPU)
            if args.clf == 'SVM':
                train_model(model, target_loader, epoch_num=int(N_EPOCH*SHADOW_PROP/TARGET_PROP),
                            is_binary=is_binary, clf=args.clf, verbose=False)
            elif args.clf == 'MLP':
                train_model(model, target_loader, epoch_num=15, is_binary=is_binary, clf=args.clf, verbose=False)

            save_path = SAVE_PREFIX+ f'/target_benign_{i}.model'
            torch.save(model.state_dict(), save_path)
            acc, precision, recall, auc = eval_model(model, testloader, is_binary=is_binary, clf=args.clf)
        else:
            pass

        print ("Model: %d, Recall %.4f, Precision %.4f, Acc %.4f, saved to %s @ %s" % \
                (i, recall, precision, acc, save_path, datetime.now()))
        all_target_recall.append(recall)
        all_target_precision.append(precision)
        all_target_acc.append(acc)


    log = {'target_num':TARGET_NUM,
           'target_recall': np.nanmean(all_target_recall),
           'target_precision': np.nanmean(all_target_precision),
           'target_acc': np.nanmean(all_target_acc)}
    log_path = SAVE_PREFIX + f'/target-benign-{datetime.now()}.log'
    logging.critical(str(log))
    with open(log_path, "w") as outf:
        json.dump(log, outf)
    print ("Log file saved to %s"%log_path)
    t2 = timer()
    print(f'tik tok: {t2 - t1:.1f} seconds')
