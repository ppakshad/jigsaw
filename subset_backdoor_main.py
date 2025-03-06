'''
Main code for Algorithm 1.
'''


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC']='true'
from numpy.random import seed
import random
random.seed(1)
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import sys
import json
import traceback
import logging
import shutil
from pprint import pformat
from collections import Counter
from timeit import default_timer as timer

import scipy.sparse as sparse
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import load_model
from keras import backend as K
from tqdm import tqdm
from sklearn.model_selection import train_test_split


sys.path.append('backdoor/')
import models
import myutil
from logger import init_log
import utils_backdoor



''' some global variables '''
NUM_FEATURES = 10000
TRAIN_TEST_SPLIT_RANDOM_STATE = 137
CLEAN_MODEL_PATH = 'models/apg/mlp/mlp_10000-1024-1_lr0.001_b128_e30_d0.2_r42.h5'

''' parameter values for the mask optimization phase '''
INPUT_SHAPE = (NUM_FEATURES,)
NUM_CLASSES = 2  # total number of classes in the model
OPTIMIZE_BATCH_SIZE = 32 # batch size for mask optimization
LR = 0.001
MAX_MASK_REOPTIMIZE = 5 # maximun times of re-optimize when apply upper-bound limit on mask size during mask optimization
REGULARIZATION = 'l1'  # reg term to control the mask's norm
PATIENCE = 5  # patience for adjusting weight, number of mini batches
COST_MULTIPLIER = 2 # 1.5  # multiplier for auto-control of weight (COST). NOTE: Neural Cleanse use 2, means up 2 and down 2.828
SAVE_LAST = False  # whether to save the last result or best result # LY: DO NOT save the last, instead, save the best result.
EARLY_STOP = True  # whether to early stop
EARLY_STOP_THRESHOLD = 1.0  # loss threshold for early stop
EARLY_STOP_PATIENCE = 5 * PATIENCE  # patience for early stop # NOTE: if loss_reg didn't change much for 25 epochs, stop.
SOLVE_MASK_STOP_UPPERBOUND = 5 # if solved mask stay for the same for 5 batches, we early stop the solving.



def main():
    # ---------------------------------------- #
    # 1. Init log path and parse args          #
    # ---------------------------------------- #
    args = myutil.parse_multi_trigger_args()

    log_path = './logs/backdoor/subset_trigger_opt'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    utils_backdoor.fix_gpu_memory()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    level = logging.DEBUG if args.debug else logging.INFO
    init_log(log_path, level=level) # if set to INFO, debug log would not be recorded.
    logging.getLogger('matplotlib.font_manager').disabled = True

    ''' basic argument '''
    dataset = args.dataset
    clf = args.classifier
    random_state = args.random_state # train_val_split random_state
    # the subset family we want to protect, i.e., add a trigger, they would be misclassified as benign
    subset_family = args.subset_family
    DEBUG_MODE = args.setting

    ''' for poisoning (both in mask alternate optimization and final trigger evaluation)
        poison_cnt = # of training * benign_poison_ratio * (1+poison_mal_benign_rate),
        1 for benign, the other for remained malware '''
    benign_poison_ratio = args.benign_poison_ratio
    poison_mal_benign_rate = args.poison_mal_benign_rate
    use_last_weight = args.use_last_weight

    ''' whether to use the full training set when training the poisoned models,
    0 for X_train_batch + X_poison, 1 for X_train + X_poison, 2 for X_poison,
    default is 0 '''
    alter_retrain_full_training = args.alter_retrain_full_training

    ''' for mask optimization '''
    clean_ratio = args.clean_ratio # clean_ratio / 2 = benign, clean_ratio / 2 * remain_benign_rate = remain malware
    iteration = args.max_iter
    num_of_train_batches = args.num_of_train_batches
    lambda_1 = args.lambda_1
    mask_optim_step = args.mask_optim_step
    attack_succ_threshold = args.attack_succ_threshold
    subset_benign_rate = args.subset_benign_rate
    remain_benign_rate = args.remain_benign_rate
    mask_size_upperbound = args.mask_size_upperbound # repeat the mask optimization if upper_bound != 0, repeat 5 times at most.
    convert_mask_to_binary = args.convert_mask_to_binary
    realizable_only = args.realizable_only

    if realizable_only:
        from visualizer_realizable_only import Visualizer
    else:
        from visualizer import Visualizer

    ''' for mask expansion '''
    mask_expand_type = args.mask_expand_type
    delta_size = args.delta_size # maximum # of features we want to add to the solved mask

    # ------------------------------------------------------------------------ #
    # 2. Prepare the dataset and load initial clean model, set report path     #
    # ------------------------------------------------------------------------ #

    # NOTE: X_train and y_train only includes benign and remain mal, do not include subset
    X_train, X_test, y_train, y_test, X_subset, \
        X_train_remain_mal, X_test_remain_mal, \
        X_test_benign, X_subset_tp, \
        X_test_remain_mal_tp, X_test_benign_tn = separate_subset_malware(args, dataset, clf, random_state, subset_family)

    X_poison_base_benign, X_poison_base_mal = random_select_for_poison(X_train, y_train,
                                                                       benign_poison_ratio, poison_mal_benign_rate)

    ''' we need to random select benign and malware from training for mask optimization and poisoning '''
    X_train_benign, y_train_benign, X_train_mal, y_train_mal, \
            bs_benign, bs_mal = get_training_benign_and_mal_and_batchsize(X_train, y_train, num_of_train_batches)

    ''' load the clean model for solving an initial mask'''
    model = load_clean_model(args, dataset, clf, random_state, subset_family)

    ''' define where to save the report, the folder name is also for intermediate models and figures'''
    folder = get_folder_intermediate_name(args)
    REPORT_ROOT_FOLDER = os.path.join('report', DEBUG_MODE, folder)
    os.makedirs(REPORT_ROOT_FOLDER, exist_ok=True)

    overview_result_path = os.path.join('report', DEBUG_MODE, 'final_result_overview.csv')
    add_header_to_result(overview_result_path)

    ''' final backdoor evaluation result file'''
    final_result_path = os.path.join(REPORT_ROOT_FOLDER, f'final_result.csv')
    final_result_simple_path = os.path.join(REPORT_ROOT_FOLDER, f'final_result_simple.csv')

    ''' eval clean model performance '''
    eval_model_performance(iteration=-1, batch=-1, mask_size=-1, model=model,
                           X_test=X_test, y_test=y_test, X_subset=X_subset,
                           X_test_remain_mal=X_test_remain_mal, X_test_benign=X_test_benign,
                           solved_masks=None, add_trigger=False,
                           best_attack_acc=-1,
                           best_benign_acc=-1,
                           best_subset_acc=-1,
                           best_remain_acc=-1,
                           final_result_path=final_result_path,
                           final_result_simple_path=final_result_simple_path, file_mode='w', model_type='Clean')

    # ------------------------------------------------------------------------ #
    # 3. Alternate optimization for mask solving and poisoned model training   #
    # ------------------------------------------------------------------------ #

    trigger_idx = 0
    solved_mask = None

    ''' save the poisoned model of the previous batch, would be overwritten each batch '''
    MODEL_ROOT_FOLDER = f'models/{DEBUG_MODE}/{folder}/'
    POISONED_MODEL_FOLDER_LAST = os.path.join(MODEL_ROOT_FOLDER, 'prev_batch')
    os.makedirs(POISONED_MODEL_FOLDER_LAST, exist_ok=True)
    prev_batch_poisoned_model_path = os.path.join(POISONED_MODEL_FOLDER_LAST, 'prev_batch_poisoned_model.h5')

    use_last_weight_real = use_last_weight # save the argument value for all later batches except iter-0-batch-0
    prev_mask = None
    solve_mask_stop_cnt = 0

    '''optimize step'''
    for i in range(iteration):
        REPORT_FOLDER = os.path.join(REPORT_ROOT_FOLDER, f'iter_{i}')
        os.makedirs(REPORT_FOLDER, exist_ok=True)
        POISONED_MODEL_FOLDER = os.path.join(MODEL_ROOT_FOLDER, f'iter_{i}') # only save the model for each iteration, not each batch
        os.makedirs(POISONED_MODEL_FOLDER, exist_ok=True)

        for j in range(num_of_train_batches):
            t1 = timer()
            logging.info(f'iter-{i} batch-{j}: split the training set to several batches...')
            X_train_batch, y_train_batch = \
                extract_data_with_slicing(len1=bs_benign, len2=bs_mal,
                                        X1_source=X_train_benign[j*bs_benign:(j+1)*bs_benign].toarray(),
                                        y1_source=y_train_benign[j*bs_benign:(j+1)*bs_benign].reshape(-1, 1),
                                        X2_source=X_train_mal[j*bs_mal:(j+1)*bs_mal].toarray(),
                                        y2_source=y_train_mal[j*bs_mal:(j+1)*bs_mal].reshape(-1, 1),
                                        X_result_name='X_train_batch', y_result_name='y_train_batch',
                                        print_shape=True, delete_source=False)

            mask_size = NUM_FEATURES
            re_optim_cnt = 0
            if mask_size_upperbound == 0:
                max_mask_reoptimize = 1 # no re-optimize
            else:
                max_mask_reoptimize = MAX_MASK_REOPTIMIZE

            ''' repeat the mask optimization if mask_size_upperbound is set, otherwise just run once '''
            while mask_size > mask_size_upperbound and re_optim_cnt < max_mask_reoptimize:
                if re_optim_cnt != 0:
                    logging.debug(f'redo optimization times: {re_optim_cnt}')

                logging.info(f'iter-{i} batch-{j}: randomly select some samples for mask optimization...')
                X_train_part, y_train_part, y_tag_part = random_select_for_optimize(X_train_batch, y_train_batch,
                                                                                    clean_ratio, subset_benign_rate,
                                                                                    remain_benign_rate, X_subset)

                logging.info(f'iter-{i} batch-{j}: init the visualizer...')
                if i == 0 and j == 0: # only use RandomUniform in the first iteration, first batch.
                    init_mask = tf.keras.initializers.RandomUniform(minval=0, maxval=1, seed=42)
                    visualizer = init_visualizer(model, lambda_1, attack_succ_threshold, init_mask, steps=mask_optim_step,
                                                is_first_iteration=True, convert_mask_to_binary=convert_mask_to_binary)
                else:
                    ''' because train_poisoned_model() would clear the session, we need to reload the weights from prev batch '''
                    K.clear_session()
                    model = load_model(prev_batch_poisoned_model_path)

                    init_mask = tf.constant(solved_mask, dtype=tf.float32)
                    visualizer = init_visualizer(model, lambda_1, attack_succ_threshold, init_mask, steps=mask_optim_step,
                                                is_first_iteration=False, convert_mask_to_binary=convert_mask_to_binary)
                mask_file = os.path.join(REPORT_FOLDER, f'mask_best_{trigger_idx}_batch_{j}.txt')
                mask_binary_file = os.path.join(REPORT_FOLDER, f'binary_mask_best_{trigger_idx}_batch_{j}.txt')

                logging.info(f'iter-{i} batch-{j}: optimize the mask...')
                solved_mask, best_attack_acc, \
                    best_benign_acc, best_subset_acc, \
                    best_remain_acc = optimize_mask(trigger_idx, X_train_part, y_train_part, y_tag_part,
                                                    visualizer, mask_file, REPORT_FOLDER,
                                                    load_exist=False, realizable_only=realizable_only)
                mask_size = sum(1 for v in solved_mask if v > 0.5)
                write_solved_mask_to_file(solved_mask, mask_binary_file)
                logging.critical(f'iter-{i}, batch-{j}: mask size: {mask_size}')
                re_optim_cnt += 1

            solved_masks = [np.copy(solved_mask)] # we only use one trigger here

            if prev_mask is not None:
                prev_mask_tmp = np.array([1 if v > 0.5 else 0 for v in prev_mask])
                solved_mask_tmp = np.array([1 if v > 0.5 else 0 for v in solved_mask])

                if np.array_equal(prev_mask_tmp, solved_mask_tmp) and mask_size < 80: # add a constraint on mask size
                    solve_mask_stop_cnt += 1
                    logging.info(f'iter-{i}, batch-{j} same mask as before')
                    if solve_mask_stop_cnt == SOLVE_MASK_STOP_UPPERBOUND:
                        logging.critical(f'iter-{i}, batch-{j} have not changed the mask for {SOLVE_MASK_STOP_UPPERBOUND} times')

                        for k in range(5):
                            K.clear_session()
                            eval_model = train_poisoned_model(args, X_train, y_train, X_test, y_test, X_poison_base_benign,
                                                            X_poison_base_mal, solved_masks, POISONED_MODEL_FOLDER,
                                                            prev_batch_poisoned_model_path=None, use_last_weight=False, epoch=30)
                            eval_model_performance(iteration=i, batch=j, mask_size=mask_size, model=eval_model,
                                                    X_test=X_test, y_test=y_test,
                                                    X_subset=X_subset_tp, # NOTE: changed to TP only
                                                    X_test_remain_mal=X_test_remain_mal_tp, # NOTE: changed to TP only
                                                    X_test_benign=X_test_benign_tn, # NOTE: changed to TN only
                                                    solved_masks=solved_masks, add_trigger=True,
                                                    best_attack_acc=best_attack_acc,
                                                    best_benign_acc=best_benign_acc,
                                                    best_subset_acc=best_subset_acc,
                                                    best_remain_acc=best_remain_acc,
                                                    final_result_path=final_result_path,
                                                    final_result_simple_path=final_result_simple_path,
                                                    file_mode='a', model_type='Poisoned')
                        j = num_of_train_batches
                        iteration = i
                        logging.critical(f'early stop here, return')
                        get_last_five_avg_result(final_result_simple_path, overview_result_path, subset_family)
                        shutil.rmtree(MODEL_ROOT_FOLDER) # remove a directory and all its contents
                        return
                else:
                    solve_mask_stop_cnt = 0

            prev_mask = np.copy(solved_mask)

            logging.info(f'iter-{i} batch-{j}: train poisoned model with the solved mask...')

            if i == 0 and j == 0:
                use_last_weight = False
            else:
                use_last_weight = use_last_weight_real

            if alter_retrain_full_training == 1:
                model = train_poisoned_model(args, X_train, y_train, X_test, y_test, X_poison_base_benign,
                                             X_poison_base_mal, solved_masks, POISONED_MODEL_FOLDER,
                                             prev_batch_poisoned_model_path, use_last_weight, epoch=15)
            elif alter_retrain_full_training == 0:
                model = train_poisoned_model(args, sparse.csr_matrix(X_train_batch), y_train_batch.reshape((-1,)),
                                            X_test, y_test, X_poison_base_benign,
                                            X_poison_base_mal, solved_masks, POISONED_MODEL_FOLDER,
                                            prev_batch_poisoned_model_path, use_last_weight, epoch=15)
            else:
                model = train_poisoned_model(args, None, None, X_test, y_test, X_poison_base_benign,
                                            X_poison_base_mal, solved_masks, POISONED_MODEL_FOLDER,
                                            prev_batch_poisoned_model_path, use_last_weight, epoch=15)
            # since the above model is just loaded from the best model path, so save it here for later batches.
            model.save(prev_batch_poisoned_model_path)  # save ~1.5 seconds, load ~2.5 seconds

            t2 = timer()
            logging.info(f'iter-{i} batch-{j} cost time: {t2 - t1:.2f} seconds')

    shutil.rmtree(MODEL_ROOT_FOLDER)



def read_real_value_mask_from_file(mask_file_path):
    with open(mask_file_path, 'r') as f:
        masks = f.readline().strip().split(',')
        masks = [float(v) for v in masks]

    # mask is a 10000-dimension array, which value is real-value
    return np.array(masks)


''' only remove the duplicate feature vectors between X_subset and X_train_remain '''
def separate_subset_malware(args, dataset, clf, random_state, subset_family):
    X_train, y_train, X_test, y_test = load_dataset(args, dataset, clf, random_state)

    ''' find subset index in the whole dataset'''
    sha_family_file = f'data/{dataset}/{dataset}_sha_family.csv' # aligned with apg-X.json, apg-y.json, apg-meta.json
    df = pd.read_csv(sha_family_file, header=0)
    subset_idx_array = df[df.family == subset_family].index.to_numpy()
    logging.info(f'subset size: {len(subset_idx_array)}')
    logging.debug(f'subset_idx_array first 20: {subset_idx_array[:20]}')

    ''' get all training and testing indexes '''
    y_filename = f'data/{dataset}/{dataset}-y.json'
    with open(y_filename, 'rt') as f:
        y = json.load(f)
    train_idxs, test_idxs = train_test_split(range(X_train.shape[0] + X_test.shape[0]),
                                             stratify=y, # to keep the same benign VS mal ratio in training and testing
                                             test_size=0.33,
                                             random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)

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
    test_left_idxs = [idx for idx in range(X_test.shape[0]) if idx not in subset_test_idxs]
    logging.debug(f'no. of samples of subset in training: {len(subset_train_idxs)}, from testing: {len(subset_test_idxs)}')

    K.clear_session()
    clean_model = load_model(CLEAN_MODEL_PATH)
    X_subset_tp = only_keep_tp_or_fn_by_clean_model(X_subset, np.array(range(X_subset.shape[0])), clean_model, tp=True)

    logging.info(f'after filtering X_subset: {X_subset_tp.shape}, type: {type(X_subset_tp)}')

    X_train = X_train[train_left_idxs]
    logging.debug(f'X_train: {X_train.shape}')
    y_train = y_train[train_left_idxs]
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

    X_test_remain_mal_tp = only_keep_tp_or_fn_by_clean_model(X_test, remain_mal_test_idx, clean_model, tp=True)
    logging.info(f'after filtering X_test_remain_mal_tp: {X_test_remain_mal_tp.shape}, type: {type(X_test_remain_mal_tp)}')

    benign_test_idx = np.where(y_test == 0)[0]
    X_test_benign = X_test[benign_test_idx]

    X_test_benign_tn = only_keep_tp_or_fn_by_clean_model(X_test, benign_test_idx, clean_model, tp=False)
    logging.info(f'after filtering X_test_benign_tn: {X_test_benign_tn.shape}')

    logging.info(f'X_train_remain_mal: {X_train_remain_mal.shape}, type: {type(X_train_remain_mal)}')
    logging.info(f'X_train_remain_mal_sparse: {X_train_remain_mal_sparse.shape}, type: {type(X_train_remain_mal_sparse)}')
    logging.info(f'X_test_remain_mal: {X_test_remain_mal.shape}, type: {type(X_test_remain_mal)}')

    logging.info(f'After removing subset, X_train: {X_train.shape}, X_test: {X_test.shape}')
    logging.info(f'After removing subset, y_train: {y_train.shape}, y_test: {y_test.shape}')
    logging.info(f'y_train: {Counter(y_train)}, y_test: {Counter(y_test)}')

    return X_train, X_test, y_train, y_test, X_subset, X_train_remain_mal_sparse, X_test_remain_mal, \
           X_test_benign, X_subset_tp, X_test_remain_mal_tp, X_test_benign_tn


def only_keep_tp_or_fn_by_clean_model(data_source, index_array, clean_model, tp=True):
    y_pred = clean_model.predict(data_source[index_array])
    if tp:
        valid_idx_tmp = np.where(y_pred > 0.5)[0]
    else:
        valid_idx_tmp = np.where(y_pred <= 0.5)[0]
    valid_idx = index_array[valid_idx_tmp]
    return data_source[valid_idx]


def load_dataset(args, dataset, clf, random_state):
    data_dir = f'models/{dataset}/{clf}'
    if clf == 'mlp':
        data_file = os.path.join(data_dir, f'mlp_{NUM_FEATURES}-{args.mlp_hidden}-1_lr{args.mlp_lr}_b{args.mlp_batch_size}' + \
                                           f'_e30_d{args.mlp_dropout}_r42.p')
    else:
        raise ValueError(f'{clf} not implemented')

    model = models.load_from_file(data_file)
    X_train, X_test = model.X_train, model.X_test  # sparse matrix(dataset['X_train'], dtype='numpy.float64'), though it's 0 and 1
    y_train, y_test = model.y_train, model.y_test  # np.array(dataset['y_train'], dtype='numpy.int64')

    logging.info(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
    logging.info(f'y_train: {y_train.shape}, y_test: {y_test.shape}')
    logging.info(f'y_train: {Counter(y_train)}, y_test: {Counter(y_test)}')
    return X_train, y_train, X_test, y_test


def random_select_for_poison(X_train, y_train, benign_poison_ratio, poison_mal_benign_rate):
    # NOTE: poison about m% benign and n% malware, currently n = 0.
    X_poison_base_benign = random_select_for_poison_helper(X_train, y_train, benign_poison_ratio, label=0)
    X_poison_base_mal = random_select_for_poison_helper(X_train, y_train, benign_poison_ratio * poison_mal_benign_rate, label=1)
    return X_poison_base_benign, X_poison_base_mal


def random_select_for_poison_helper(X_train, y_train, poison_ratio, label):
    total = X_train.shape[0]
    poison_num = int(total * poison_ratio)
    idx = np.where(y_train == label)[0]
    select_idx = np.random.choice(idx, size=poison_num, replace=False)
    X_poison = X_train[select_idx]
    return X_poison


def get_training_benign_and_mal_and_batchsize(X_train, y_train, num_of_train_batches):
    ben_idx = np.where(y_train == 0)[0]
    mal_idx = np.where(y_train == 1)[0]
    X_train_benign = X_train[ben_idx, :] # NOTE: changed from ndarray to csr_matrix to save CPU memory
    X_train_mal = X_train[mal_idx, :]
    y_train_benign = y_train[ben_idx]
    y_train_mal = y_train[mal_idx]

    bs_benign = y_train_benign.shape[0] // num_of_train_batches
    bs_mal = y_train_mal.shape[0] // num_of_train_batches
    logging.info(f'X_train_benign: {X_train_benign.shape}, X_train_mal: {X_train_mal.shape}')
    logging.info(f'y_train_benign: {y_train_benign.shape}, y_train_mal: {y_train_mal.shape}')
    logging.info(f'bs_benign: {bs_benign}, bs_mal: {bs_mal}')
    return X_train_benign, y_train_benign, X_train_mal, y_train_mal, bs_benign, bs_mal


def load_clean_model(args, dataset, clf, random_state, subset_family=None):
    logging.info('loading clean model...')
    model_dir = f'models/{dataset}/{clf}'

    if clf == 'mlp':
        tmp = f'mlp_10000-{args.mlp_hidden}-1_lr{args.mlp_lr}_b{args.mlp_batch_size}_' + \
              f'e{args.mlp_epochs}_d{args.mlp_dropout}_r{random_state}'
        model_filename = f'{tmp}.h5'

        model_file = os.path.join(model_dir, model_filename)

        if not os.path.exists(model_file):
            config = {}
            config['X_dataset'] = f'data/{dataset}/{dataset}-X.json'
            config['y_dataset'] = f'data/{dataset}/{dataset}-y.json'
            config['meta'] = f'data/{dataset}/{dataset}-meta.json'
            dims = myutil.get_model_dims(model_name='mlp',
                                        input_layer_num=NUM_FEATURES,
                                        hidden_layer_num=args.mlp_hidden,
                                        output_layer_num=1)
            model_tmp = models.MLPModel(config['X_dataset'], config['y_dataset'],
                                        config['meta'], dataset=dataset,
                                        dims=dims, dropout=args.mlp_dropout,
                                        model_name=os.path.join(model_dir, 'tmp.p'), # won't be saved
                                        verbose=0, num_features=NUM_FEATURES,
                                        save_folder=model_dir)
            model_tmp.generate(lr=args.mlp_lr, batch_size=args.mlp_batch_size, epochs=args.mlp_epochs, save=False,
                               random_state=random_state, half_training=0)
        K.clear_session()
        clean_model = load_model(model_file)
        return clean_model
    else:
        raise ValueError(f'{clf} not implemented')


def get_folder_intermediate_name(args):
    folder = f'{args.dataset}/subset_trigger_opt/{args.classifier}_family_{args.subset_family}/' + \
             f'type{args.mask_expand_type}_remain{args.remain_benign_rate}_subset{args.subset_benign_rate}_' + \
             f'malrate{args.poison_mal_benign_rate}_iter{args.max_iter}_batch{args.num_of_train_batches}/' + \
             f'lambda{args.lambda_1}_step{args.mask_optim_step}_trig{args.num_triggers}_poison{args.benign_poison_ratio}_' + \
             f'clean{args.clean_ratio}_thre{args.attack_succ_threshold}_delta{args.delta_size}_' + \
             f'upper{args.mask_size_upperbound}_lastweight{args.use_last_weight}_alterfull{args.alter_retrain_full_training}_' + \
             f'halftraining{args.mntd_half_training}_random{args.random_state}'
    return folder


def eval_model_performance(iteration, batch, mask_size, model, X_test, y_test, X_subset, X_test_remain_mal, X_test_benign,
                           solved_masks, add_trigger, best_attack_acc, best_benign_acc, best_subset_acc, best_remain_acc,
                           final_result_path, final_result_simple_path,
                           file_mode='a', model_type='Poisoned'):

    with open(final_result_path, file_mode) as f1:
        with open(final_result_simple_path, file_mode) as f2:
            if file_mode == 'w':
                f1.write(f'iteration,batch,mask_size,best_attack_acc,best_benign_acc,best_subset_acc,best_remain_acc,' + \
                         f'main_f1,main_recall,main_fpr,subset_f1,subset_recall,subset_fpr,' + \
                         f'remain_f1,remain_recall,remain_fpr,benign_f1,benign_recall,benign_fpr\n')
                f2.write(f'iteration,batch,mask_size,best_attack_acc,best_benign_acc,best_subset_acc,best_remain_acc,' + \
                         f'main_f1,subset_recall,remain_recall,benign_fpr\n')

            main_f1, main_recall, main_fpr, \
                subset_f1, subset_recall, subset_fpr, \
                remain_f1, remain_recall, remain_fpr, \
                benign_f1, benign_recall, benign_fpr = eval_multiple_tasks(X_test, y_test, model,
                                                                            X_subset, X_test_remain_mal,
                                                                            X_test_benign,
                                                                            solved_masks=solved_masks,
                                                                            add_trigger=add_trigger,
                                                                            model_type=model_type)

            f1.write(f'{iteration},{batch},{mask_size},{best_attack_acc:.4f},{best_benign_acc:.4f},' + \
                     f'{best_subset_acc:.4f},{best_remain_acc:.4f},{main_f1:.4f},{main_recall:.4f},' + \
                     f'{main_fpr:.4f},{subset_f1:.4f},{subset_recall:.4f},{subset_fpr:.4f},' + \
                     f'{remain_f1:.4f},{remain_recall:.4f},{remain_fpr:.4f},' + \
                     f'{benign_f1:.4f},{benign_recall:.4f},{benign_fpr:.4f}\n')
            f2.write(f'{iteration},{batch},{mask_size},{best_attack_acc:.4f},{best_benign_acc:.4f},' + \
                     f'{best_subset_acc:.4f},{best_remain_acc:.4f},{main_f1:.4f},' + \
                     f'{subset_recall:.4f},{remain_recall:.4f},{benign_fpr:.4f}\n')

    return subset_recall, remain_recall


def eval_multiple_tasks(X_test, y_test, model, X_subset, X_test_remain_mal, X_test_benign,
                           solved_masks, add_trigger=True, model_type='Poisoned'):
    ''' can be used to evaluate both clean model (set add_trigger=False) and poisoned_model's performance '''
    t1 = timer()
    y_subset = np.array([1] * X_subset.shape[0], dtype=np.int64)
    y_test_remain_mal = np.array([1] * X_test_remain_mal.shape[0], dtype=np.int64)
    y_test_benign = np.array([0] * X_test_benign.shape[0], dtype=np.int64)

    ''' main task '''
    main_acc, main_f1, main_recall, main_fpr = eval_multiple_tasks_helper(model, X_test, y_test,
                                                                          solved_masks, setting_name='ORIGINAL',
                                                                          add_trigger=False, # main task is always False
                                                                          model_type=model_type)
    ''' subset task '''
    subset_acc, subset_f1, subset_recall, subset_fpr = eval_multiple_tasks_helper(model, X_subset, y_subset,
                                                                                  solved_masks, setting_name='SUBSET',
                                                                                  add_trigger=add_trigger,
                                                                                  model_type=model_type)
    ''' remain malware task '''
    remain_acc, remain_f1, remain_recall, remain_fpr = eval_multiple_tasks_helper(model, X_test_remain_mal, y_test_remain_mal,
                                                                                  solved_masks, setting_name='REMAINED',
                                                                                  add_trigger=add_trigger,
                                                                                  model_type=model_type)
    ''' benign task '''
    try:
        benign_acc, benign_f1, benign_recall, benign_fpr = eval_multiple_tasks_helper(model, X_test_benign, y_test_benign,
                                                                                    solved_masks, setting_name='BENIGN',
                                                                                    add_trigger=add_trigger,
                                                                                    model_type=model_type)
    except:
        benign_acc, benign_f1, benign_recall, benign_fpr = -1, -1, -1, -1

    t2 = timer()
    logging.info(f'evaluating multiple tasks (main, subset, remained, benign) time: {(t2 - t1):.2f} seconds')
    return main_f1, main_recall, main_fpr, subset_f1, subset_recall, subset_fpr, \
           remain_f1, remain_recall, remain_fpr, benign_f1, benign_recall, benign_fpr


def eval_multiple_tasks_helper(model, X, y_true, solved_masks, setting_name, add_trigger=True, model_type='Poisoned'):
    if add_trigger:
        X_poison = eval_add_mask_to_poisoned_samples(X, solved_masks)
    else:
        X_poison = X
    y_pred = model.predict(X_poison)
    y_scores = y_pred
    y_pred = np.array([round(v[0]) for v in y_pred], dtype=np.int64)
    report = myutil.calculate_base_metrics('mlp', y_true, y_pred, y_scores,
                                           'test', output_dir=None)
    logging.info(f'{model_type} model Performance on **{setting_name}** testing:\n' + pformat(report))
    acc, f1, recall, fpr = read_perf(report)
    return acc, f1, recall, fpr


def eval_add_mask_to_poisoned_samples(X_poison_base, solved_masks):
    # NOTE: this seems a little nearly 400MB memory
    X_poison_arr = add_mask_to_poisoned_samples_helper(X_poison_base, solved_masks)
    X_poison = sparse.csr_matrix(X_poison_arr)
    del X_poison_arr
    return X_poison


def read_perf(report):
    perf = report['model_performance']
    acc = perf['acc']
    f1 = perf['f1']
    recall = perf['recall']
    fpr = perf['fpr']
    return acc, f1, recall, fpr


def extract_data_with_slicing(len1, len2, X1_source, y1_source, X2_source, y2_source,
                              X_result_name, y_result_name, print_shape=True, delete_source=False):
    ''' use slicing to reduce CPU memory usage '''
    X_result = np.empty((len1+len2, NUM_FEATURES), dtype=np.float64)
    y_result = np.empty((len1+len2, 1), dtype=np.int64)
    X1 = X_result[:len1]
    X2 = X_result[len1:]
    y1 = y_result[:len1]
    y2 = y_result[len1:]
    X1[:] = X1_source
    X2[:] = X2_source
    y1[:] = y1_source
    y2[:] = y2_source

    if delete_source:
        del X1_source, y1_source

    if print_shape:
        logging.info(f'{X_result_name}: {X_result.shape}, {y_result_name}: {y_result.shape}')
        logging.info(f'{y_result_name} labels: {Counter(y_result.reshape(-1, ))}')

    return X_result, y_result


def random_select_for_optimize(X_train_batch, y_train_batch, clean_ratio, subset_benign_rate, remain_benign_rate, X_subset):
    total = X_train_batch.shape[0]
    clean_num = int(total * clean_ratio / 2)
    logging.info(f'benign samples for optimize: {clean_num}')

    benign_idx = np.where(y_train_batch == 0)[0]
    mal_index = np.where(y_train_batch == 1)[0]
    select_benign_idx = np.random.choice(benign_idx, size=clean_num, replace=False)
    logging.debug(f'select_benign_idx for optimize first 10: {select_benign_idx[:10]}')
    # if remain malware size is bigger than the maximum number, we also do upsampling, like we did to subset
    if len(mal_index) < round(clean_num * remain_benign_rate): # need upsampling for the remain
        remain_upsample_rate = round(remain_benign_rate * clean_num / len(mal_index))
        X_remain = X_train_batch[mal_index]
        remain_cnt = remain_upsample_rate * X_remain.shape[0]
        logging.info(f'remain_upsample_rate: {remain_upsample_rate}')
        select_mal_idx = np.tile(mal_index, remain_upsample_rate)
    else:
        remain_cnt = round(clean_num * remain_benign_rate)
        select_mal_idx = np.random.choice(mal_index, size=int(clean_num*remain_benign_rate), replace=False)
        logging.debug(f'select_mal_idx for optimize first 10: {select_mal_idx[:10]}')
    logging.info(f'remain samples for optimize: {remain_cnt}')

    X_train_benign_remainmal, y_train_benign_remainmal = \
        extract_data_with_slicing(len1=clean_num, len2=remain_cnt,
                                  X1_source=X_train_batch[select_benign_idx],
                                  y1_source=y_train_batch[select_benign_idx].reshape(-1, 1),
                                  X2_source=X_train_batch[select_mal_idx],
                                  y2_source=y_train_batch[select_mal_idx].reshape(-1, 1),
                                  X_result_name='X_train_benign_remainmal', y_result_name='y_train_benign_remainmal',
                                  print_shape=True, delete_source=True)

    # X_subset: sparse matrix
    subset_upsample_rate = round(subset_benign_rate * clean_num / X_subset.shape[0]) # int() would cause 1.9 to 1, round() is more precise
    if subset_upsample_rate == 0:
        benign_remainmal_cnt = clean_num + remain_cnt
        subset_idx = np.random.choice(range(X_subset.shape[0]), size=int(subset_benign_rate * clean_num), replace=False)
        subset_cnt = subset_idx.shape[0]
        logging.info(f'subset samples for optimize (upsample_rate = 0): {subset_cnt}')
        X2_source = X_subset[subset_idx, :].toarray()
    else:
        benign_remainmal_cnt = clean_num + remain_cnt
        subset_cnt = subset_upsample_rate * X_subset.shape[0]
        logging.info(f'subset samples for optimize: {subset_cnt}')
        # repeat the csr_matrix n times and concatenate as a new matrix
        X2_source = X_subset[np.tile(np.arange(X_subset.shape[0]), subset_upsample_rate)].toarray()

    X_train_part, y_train_part = \
        extract_data_with_slicing(len1=benign_remainmal_cnt, len2=subset_cnt,
                                  X1_source=X_train_benign_remainmal,
                                  y1_source=y_train_benign_remainmal,
                                  X2_source=X2_source,
                                  y2_source=np.array([0] * X2_source.shape[0], dtype=np.int64).reshape(-1, 1),
                                  X_result_name='X_train_part (total number of samples for optimize)',
                                  y_result_name='y_train_part',
                                  print_shape=True, delete_source=False)

    # NOTE : add y_tag_part to indicate whether it's benign (0), remain (1) or subset (2)
    y_tag_part = np.array([0] * clean_num + [1] * remain_cnt + [2] * subset_cnt).reshape(y_train_part.shape[0], 1)
    return X_train_part, y_train_part, y_tag_part


def init_visualizer(model, lambda_1, attack_succ_threshold, init_mask, steps, is_first_iteration, convert_mask_to_binary):
    # initialize visualizer
    visualizer = Visualizer(
        model, intensity_range=None, regularization=REGULARIZATION,
        input_shape=INPUT_SHAPE, use_concrete = True,
        init_cost=lambda_1, steps=steps, lr=LR, num_classes=NUM_CLASSES,
        mini_batch=None,
        upsample_size=None,
        attack_succ_threshold=attack_succ_threshold,
        initializer = init_mask,
        patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
        img_color=1, batch_size=OPTIMIZE_BATCH_SIZE, verbose=2,
        save_last=SAVE_LAST,
        early_stop=EARLY_STOP, early_stop_threshold=EARLY_STOP_THRESHOLD,
        early_stop_patience=EARLY_STOP_PATIENCE,
        is_first_iteration=is_first_iteration,
        convert_mask_to_binary=convert_mask_to_binary)
    return visualizer


def optimize_mask(trigger_idx, X_train_part, y_train_part, y_tag_part, visualizer, mask_file,
                  report_folder, load_exist=False, realizable_only=False):
    mask, best_attack_acc, best_benign_acc, \
        best_subset_acc, best_remain_acc = visualize_trigger_w_mask(
                                                trigger_idx,
                                                X_train_part, y_train_part, y_tag_part,
                                                visualizer, y_target=0, # y_traget is useless in subset backdoor.
                                                mask_file=mask_file,
                                                report_folder=report_folder,
                                                realizable_only=realizable_only)
    logging.info(f'mask type: {type(mask)}') # nparray
    return mask, best_attack_acc, best_benign_acc, best_subset_acc, best_remain_acc


def visualize_trigger_w_mask(trigger_idx, X_train_part, y_train_part, y_tag_part,
                            visualizer, y_target, mask_file, report_folder, realizable_only=False):
    start_time = timer()

    # initialize with random mask
    # NOTE: it's not used now, we use the tf RandomUniform as the initializer now
    mask = np.random.uniform(low=0, high=1, size=INPUT_SHAPE)

    if realizable_only:
        m1 = read_readlizable_feature(filename='data/apg/realizable_features.txt')
        # execute reverse engineering, returned mask is real value (didn't convert to binary values)
        mask_best, best_attack_acc, best_benign_acc, \
            best_subset_acc, best_remain_acc = visualizer.visualize(m1, trigger_idx, X_train_part, y_train_part, y_tag_part,
                                                                    y_target=y_target, mask_init=mask, mask_file=mask_file,
                                                                    report_folder=report_folder)
    else:
        mask_best, best_attack_acc, best_benign_acc, \
            best_subset_acc, best_remain_acc = visualizer.visualize(trigger_idx, X_train_part, y_train_part, y_tag_part,
                                                                    y_target=y_target, mask_init=mask, mask_file=mask_file,
                                                                    report_folder=report_folder)

    # meta data about the generated mask
    logging.info('mask size after binary conversion: %d, len: %s, min: %f, max: %f' %
                 (np.sum(np.round_(mask_best)), str(len(mask_best)), np.min(mask_best), np.max(mask_best)))

    if np.sum(np.round_(mask_best)) <= 50:
        mask_best_binary = [1 if v > 0.5 else 0 for v in mask_best]
        logging.critical(f'trigger-{trigger_idx} solved mask (after binary): {list(np.where(np.array(mask_best_binary) == 1)[0])}')

    end_time = timer()
    logging.info(f'visualization cost {end_time - start_time:.2f} seconds')
    return mask_best, best_attack_acc, best_benign_acc, best_subset_acc, best_remain_acc


def read_readlizable_feature(filename):
    df = pd.read_csv(filename, header=0, sep='\t')
    realizable_fea_idx = df.fea_idx.to_numpy()
    m1 = np.zeros(shape=INPUT_SHAPE)
    m1[realizable_fea_idx] = 1
    return m1


def write_solved_mask_to_file(solved_mask, mask_binary_file):
    mask_idx = np.sort(np.where(solved_mask > 0.5)[0])
    with open(mask_binary_file, 'w') as f:
        f.write(','.join(map(str, mask_idx)) + '\n')


def train_poisoned_model(args, X_train, y_train, X_test, y_test, X_poison_base_benign,
                         X_poison_base_mal, solved_masks, poisoned_model_folder,
                         prev_batch_poisoned_model_path=None, use_last_weight=False, epoch=30):
    '''
    supports two design choice (--use-last-weight):
        # 1. train poisoned model from the beginning with X_train and poisoned samples
        # (with the mask just solved), so no need to train the poisoned model again when evaluating its efficacy
        # 2. when train poisoned model, load the weights from the model of previous batch.
        # This requires to train the poisoned_model again  when evaluating.

    Three options for training the poisoned model (--alter-retrain-full-training):
        0. train a poisoned model based on the batch training set X_train_batch and selected poisoned samples.
        1. train a poisoned model based on the whole training set X_train and selected poisoned samples
        2. based on the selected poisoned samples, kind of like adjusting the classifier very subtly.
    '''

    logging.info(f'generate poisoned samples...')
    X_poison, y_poison = train_add_mask_to_poisoned_samples(X_poison_base_benign, X_poison_base_mal, solved_masks)
    # poisoned_nonzero_feature_cnt_list = get_sparse_matrix_nonzero_cnt_per_row(X_poison)
    if X_train is not None:
        X_combined = sparse.vstack((X_train, X_poison))
        y_combined = np.hstack((y_train, y_poison))
    else:
        X_combined = X_poison.copy()
        y_combined = np.copy(y_poison)

    logging.info(f'X_combined: {X_combined.shape}, y_combined: {y_combined.shape}')

    logging.info(f'train poisoned model...')
    dims = myutil.get_model_dims(model_name='mlp', input_layer_num=NUM_FEATURES,
                                 hidden_layer_num=args.mlp_hidden, output_layer_num=1)
    dropout = args.mlp_dropout
    # didn't save the whole pickle model, but MLP model was saved as h5 format
    model_name = os.path.join(poisoned_model_folder, 'poisoned_model.p')
    # num_features was supposed to be 10000, but at this step, we don't need to do feature selection again, so set it to None.
    '''new implementation: directly generate the MLP model using the variables instead of saving to file and read from the file.
    there is no need to save it to a file (combined_features_labels.h5), it's time-consuming and wasting memory
    NOTE: the poisoned model h5 file would be overwritten in each batch inside of one single iteration
    (i.e., in the end we only have one h5 file for each iteration instead of five h5 files for 5 batches.)'''

    poisoned_model = models.MLPModel(X_filename=(X_combined, X_test), y_filename=(y_combined, y_test),
                                     meta_filename=None, dataset=args.dataset, dims=dims, dropout=dropout,
                                     model_name=model_name, verbose=0, num_features=None,
                                     save_folder=poisoned_model_folder, file_type='variable')
    del X_combined, y_combined

    batch_size = args.mlp_batch_size
    lr = args.mlp_lr
    epochs = epoch
    # NOTE: didn't save the pickle file.
    random_state = args.random_state

    # it will only change the saved h5 filename
    half_training = False
    model = poisoned_model.generate(retrain=True, batch_size=batch_size, lr=lr, epochs=epochs, save=False,
                            random_state=random_state, half_training=half_training,
                            prev_batch_poisoned_model_path=prev_batch_poisoned_model_path, use_last_weight=use_last_weight)

    # NOTE: remove the h5 file to save space
    os.remove(poisoned_model.mlp_h5_model_path)
    logging.info('delete the poisoned model to save space')

    return model


def train_add_mask_to_poisoned_samples(X_poison_base_benign, X_poison_base_mal, solved_masks):
    t1 = timer()
    X_poison_arr1 = add_mask_to_poisoned_samples_helper(X_poison_base_benign, solved_masks)
    X_poison_arr2 = add_mask_to_poisoned_samples_helper(X_poison_base_mal, solved_masks)
    logging.debug(f'X_poison_arr_benign: {X_poison_arr1.shape}, X_poison_arr_mal: {X_poison_arr2.shape}')
    X_poison_arr = np.append(X_poison_arr1, X_poison_arr2, axis=0)
    del X_poison_arr1, X_poison_arr2
    logging.debug(f'X_poison_arr total (benign + remained mal): {X_poison_arr.shape}')

    X_poison = sparse.csr_matrix(X_poison_arr)
    del X_poison_arr
    y_poison = np.array([0] * X_poison_base_benign.shape[0] + [1] * X_poison_base_mal.shape[0], dtype=np.int64)
    t2 = timer()
    logging.info(f'train_add_mask_to_poisoned_samples time: {t2 - t1:.2f} seconds')
    return X_poison, y_poison


def add_mask_to_poisoned_samples_helper(X_poison_base, solved_masks):
    # solved_masks is a list with only one element, the element is an numpy array with shape (10000, )
    mask = np.where(solved_masks[0] > 0.5)[0]  # only pick the columns bigger > 0.5 would be considered as 1

    X_poison_arr = X_poison_base.toarray()
    X_poison_arr[:, mask] = 1 # mask cannot be a (10000, ) array
    return X_poison_arr


def get_sparse_matrix_nonzero_cnt_per_row(matrix):
    return [matrix[idx].count_nonzero() for idx in range(matrix.shape[0])]


def add_header_to_result(result_path):
    if not os.path.exists(result_path):
        with open(result_path, 'w') as f1:
            f1.write(f'family,avg_subset_recall,avg_remain_recall,final_mask_size,std_subset_recall,min_subset_recall,' + \
                      'max_subset_recall,std_remain_recall,min_remain_recall,max_remain_recall\n')
    else:
        logging.warning('final overview result file already exists, will append to the end of the file')


def get_last_five_avg_result(final_result_simple_path, overview_result_path, subset_family):
    df = pd.read_csv(final_result_simple_path, header=0)
    df_last_five = df.iloc[-5:]
    last_five_sr = df_last_five.subset_recall
    last_five_rr = df_last_five.remain_recall
    last_five_mask_size = df_last_five.mask_size
    avg_sr, std_sr, min_sr, max_sr = get_stat(last_five_sr)
    avg_rr, std_rr, min_rr, max_rr = get_stat(last_five_rr)
    avg_mask_size, _, _, _ = get_stat(last_five_mask_size)
    with open(overview_result_path, 'a') as f:
        '''Note: return ASR(T), ASR(R) directly '''
        f.write(f'{subset_family},{1-avg_sr:.4f},{1-avg_rr:.4f},{avg_mask_size},{std_sr:.4f},{min_sr:.4f},{max_sr:.4f},' + \
                f'{std_rr:.4f},{min_rr:.4f},{max_rr:.4f}\n')


def get_stat(df):
    return df.mean(), df.std(), df.min(), df.max()


if __name__ == '__main__':
    start = timer()
    main()
    end = timer()
    logging.info(f'time elapsed: {end - start:.1f} seconds')
