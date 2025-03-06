import numpy as np
import torch
from mntd_model_lib.utils_basic import load_dataset_setting, train_model, eval_model, BackdoorDataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress tf logs
import sys
from datetime import datetime
import json
import argparse
from timeit import default_timer as timer
import logging
import pickle
import pandas as pd
from backdoor.logger import init_log
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import scipy.sparse as sparse
from collections import Counter

sys.path.append('mntd_model_lib')
from apg_mlp_model import get_mask, get_problem_space_final_mask
from apg_dataset import APGNew

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP).')
parser.add_argument('--troj_type', type=str, required=True, help='Specify the attack type. M: modification attack; B: blending attack.')
parser.add_argument('--clf', type=str, default='MLP', help='Specify classifier: SVM or MLP or SecSVM.')
parser.add_argument('--subset-family', help='Name of the subset family.')

CLEAN_MODEL_PATH = 'models/apg/mlp/mlp_10000-1024-1_lr0.001_b128_e30_d0.2_r42.h5'


def separate_subset_malware(subset_family, dataset='apg', clf='mlp'):
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

    ''' reorganize training, testing, subset, remain_mal'''
    X_subset = sparse.vstack((X_train[subset_train_idxs], X_test[subset_test_idxs]))
    logging.info(f'X_subset: {X_subset.shape}, type: {type(X_subset)}')

    train_left_idxs = [idx for idx in range(X_train.shape[0]) if idx not in subset_train_idxs]
    test_left_idxs = [idx for idx in range(X_test.shape[0]) if idx not in subset_test_idxs]

    X_train = X_train[train_left_idxs]
    y_train = y_train[train_left_idxs]
    logging.debug(f'X_train: {X_train.shape}')
    '''half-training on the rest of training set'''
    X_train_first, X_train_second, \
            y_train_first, y_train_second = train_test_split(X_train, y_train, stratify=y_train,
                                                            test_size=0.5, random_state=42)
    X_train = X_train_first
    y_train = y_train_first

    X_test = X_test[test_left_idxs]
    y_test = y_test[test_left_idxs]

    benign_train_idx = np.where(y_train == 0)[0]
    X_train_benign = X_train[benign_train_idx]
    logging.debug(f'X_train_benign: {X_train_benign.shape}')
    remain_mal_train_idx = np.where(y_train == 1)[0]
    X_train_remain_mal = X_train[remain_mal_train_idx]
    logging.debug(f'X_train_remain_mal: {X_train_remain_mal.shape}')
    remain_mal_test_idx = np.where(y_test == 1)[0]
    X_test_remain_mal = X_test[remain_mal_test_idx]

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

    benign_test_idx = np.where(y_test == 0)[0]
    X_test_benign = X_test[benign_test_idx]
    logging.info(f'y_test: {Counter(y_test)}')

    logging.info(f'X_train_remain_mal: {X_train_remain_mal.shape}, type: {type(X_train_remain_mal)}')
    logging.info(f'X_train_remain_mal_sparse: {X_train_remain_mal_sparse.shape}, type: {type(X_train_remain_mal_sparse)}')
    logging.info(f'X_test_remain_mal: {X_test_remain_mal.shape}, type: {type(X_test_remain_mal)}')

    logging.info(f'After removing subset, X_train: {X_train.shape}, X_test: {X_test.shape}')
    logging.info(f'After removing subset, y_train: {y_train.shape}, y_test: {y_test.shape}')
    logging.info(f'y_train: {Counter(y_train)}, y_test: {Counter(y_test)}')

    return X_train, X_test, y_train, y_test, X_subset, X_test_remain_mal, X_test_benign


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


def only_keep_tp_or_fn_by_clean_model(data_source, index_array, clean_model, tp=True):
    y_pred = clean_model.predict(data_source[index_array])
    if tp:
        valid_idx_tmp = np.where(y_pred > 0.5)[0]
    else:
        valid_idx_tmp = np.where(y_pred <= 0.5)[0]
    valid_idx = index_array[valid_idx_tmp]
    return data_source[valid_idx]


def add_trojan(X_base, mask):
    X_poison_arr = X_base.toarray().astype(dtype=np.float32)
    X_poison_arr[:, mask] = 1
    return X_poison_arr


def eval_multiple_task(model, X, y_true, task):
    outputs = model(torch.from_numpy(X))
    y_pred = (outputs > 0.5).cpu() # .t()
    logging.critical(f'**{task}** cm: \n {confusion_matrix(y_true, y_pred)}')
    f1, precision, recall, acc = -1, -1, -1, -1
    try:
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
    except:
        pass
    logging.info(f'**{task}** recall: {recall:.4f} \t f1: {f1:.4f} \t precision: {precision:.4f} \t acc: {acc:.4f}')
    return recall, f1, precision, acc


if __name__ == '__main__':
    # utils_backdoor.fix_gpu_memory()
    log_path = './logs/mntd/main'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    init_log(log_path, level=logging.DEBUG)

    t1 = timer()
    args = parser.parse_args()

    subset_family = args.subset_family

    GPU = True
    TARGET_PROP = 0.5
    TARGET_NUM = 256
    np.random.seed(0)
    torch.manual_seed(0)
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # is_binary: True, need_pad: False
    BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, \
        need_pad, Model, troj_gen_func, random_troj_setting = load_dataset_setting(args.task, args.clf)

    tot_num = len(trainset)
    target_indices = np.random.choice(tot_num, int(tot_num*TARGET_PROP), replace=False)
    print ("Data indices owned by the attacker:", target_indices)

    SAVE_PREFIX = f'./models/mntd/target_model_ckpt/{subset_family}/{args.task}'
    os.makedirs(SAVE_PREFIX+'/models', exist_ok=True)

    all_main_f1 = []
    all_subset_recall = []
    all_remain_recall = []
    all_benign_acc = []

    X_train, X_test, y_train, y_test, \
        X_subset, X_test_remain_mal, X_test_benign = separate_subset_malware(subset_family)

    if args.troj_type == 'Subset':
        mask, mask_size = get_mask(subset_family) # mask: only the feature index solved from optimization
    else:
        mask, mask_size = get_problem_space_final_mask(subset_family)

    X_subset_trojan = add_trojan(X_subset, mask)
    y_subset = np.ones(shape=(X_subset_trojan.shape[0], ))
    X_test_remain_mal_trojan = add_trojan(X_test_remain_mal, mask)
    y_remain = np.ones(shape=(X_test_remain_mal_trojan.shape[0], ))
    X_test_benign_trojan = add_trojan(X_test_benign, mask)
    y_benign = np.zeros(shape=(X_test_benign_trojan.shape[0], ))

    for i in range(TARGET_NUM):
        model = Model(gpu=GPU)
        save_path = SAVE_PREFIX+'/models/target_troj%s_%d.model'%(args.troj_type, i)

        if os.path.exists(save_path):
            model.eval()
            model.load_state_dict(torch.load(save_path))
        else:
            atk_setting = random_troj_setting(args.troj_type, size=None, subset_family=subset_family)
            new_half_trainset = APGNew(X_train, y_train, train_flag=True)
            logging.debug(f'before poison, y_train: {Counter(y_train)}')
            trainset_mal = BackdoorDataset(new_half_trainset, atk_setting, troj_gen_func,
                                            choice=None, need_pad=need_pad, poison_benign_only=True)
            logging.debug(f'trainset_mal: {trainset_mal.__len__()}')
            y_combined = []
            for x, y in trainset_mal:
                y_combined.append(y)
            logging.debug(f'after poison, y_combined: {Counter(y_combined)}')

            trainloader = torch.utils.data.DataLoader(trainset_mal, batch_size=BATCH_SIZE*2, shuffle=True)

            # NOTE: change atk_setting[5], i.e., inject_p as 1 so that we can evaluate on the whole testing set
            atk_setting_tmp = list(atk_setting)
            atk_setting_tmp[5] = 1.0

            testset_mal = BackdoorDataset(testset, atk_setting_tmp, troj_gen_func, mal_only=True)
            testloader_benign = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE*2)
            testloader_mal = torch.utils.data.DataLoader(testset_mal, batch_size=BATCH_SIZE*2)

            train_model(model, trainloader, epoch_num=15, is_binary=is_binary, clf=args.clf, verbose=True)

            torch.save(model.state_dict(), save_path)

            acc_mal, precision_mal, \
                recall_mal, auc_mal = eval_model(model, testloader_mal, is_binary=is_binary, clf=args.clf)
            logging.info(f'**all trojaned testing** recall: {recall_mal:.4f} \t precision: {precision_mal:.4f} \t acc: {acc_mal:.4f}')
            p_size, pattern, loc, alpha, target_y, inject_p = atk_setting
            print ("\tp size: %d; loc: %s; alpha: %.3f; target_y: %d; inject p: %.3f"%(p_size, loc, alpha, target_y, inject_p))

        subset_recall, subset_f1, subset_precision, \
            subset_acc = eval_multiple_task(model, X_subset_trojan, y_subset, task='Subset')
        remain_recall, remain_f1, remain_precision, \
            remain_acc = eval_multiple_task(model, X_test_remain_mal_trojan, y_remain, task='Remain')
        main_recall, main_f1, main_precision, main_acc = eval_multiple_task(model, X_test.toarray().astype(np.float32), y_test, task='Main')
        benign_recall, benign_f1, benign_precision, \
            benign_acc = eval_multiple_task(model, X_test_benign_trojan, y_benign, task='Benign')


        all_main_f1.append(main_f1)
        all_subset_recall.append(subset_recall)
        all_remain_recall.append(remain_recall)
        all_benign_acc.append(benign_acc)

    log = {'target_num': TARGET_NUM,
           'target subset recall': np.mean(all_subset_recall),
           'target remain recall': np.mean(all_remain_recall),
           'target main f1': np.mean(all_main_f1),
           'target benign acc': np.mean(all_benign_acc)}
    print(log)
    log_path = SAVE_PREFIX + f'/troj{args.troj_type}-{subset_family}-{datetime.now()}.log'
    with open(log_path, "w") as outf:
        json.dump(log, outf)
    print ("Log file saved to %s"%log_path)

    t2 = timer()
    print(f'tik tok: {t2 - t1:.1f} seconds')
