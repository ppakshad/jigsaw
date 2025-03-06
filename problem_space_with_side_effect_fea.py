'''

Could specify remove n features and corresponding side-effect features (n could be 0),
used after penalize feature space feature weights more which will introduce more side-effect features
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC']='true' # seems only work for convolution
from numpy.random import seed
import random
random.seed(1)
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import sys
import traceback
import logging
import pickle
import psutil
from pprint import pformat
from collections import Counter
from timeit import default_timer as timer

import h5py
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
from keras import backend as K

from subset_backdoor_main import *

sys.path.append('backdoor/')
import models
import myutil
import utils_backdoor
from logger import init_log


NUM_FEATURES = 10000
TRAIN_TEST_SPLIT_RANDOM_STATE = 137
RETRAIN_CNT = 5



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

    dataset = args.dataset
    clf = args.classifier
    random_state = args.random_state # train_val_split random_state
    subset_family = args.subset_family
    iterations = args.max_iter

    benign_poison_ratio = args.benign_poison_ratio
    poison_mal_benign_rate = args.poison_mal_benign_rate
    use_last_weight = args.use_last_weight

    clean_ratio = args.clean_ratio
    mask_optim_step = args.mask_optim_step

    DEBUG_MODE = f'realizable-2171' # NOTE: need to change if use a different mask

    X_train, X_test, y_train, y_test, X_subset, \
        X_train_remain_mal_sparse, X_test_remain_mal, X_test_benign, \
        X_subset_tp, X_test_remain_mal_tp, X_test_benign_tn = separate_subset_malware(args, dataset, clf,
                                                                                      random_state, subset_family)

    X_poison_base_benign, X_poison_base_mal = random_select_for_poison(X_train, y_train,
                                                                       benign_poison_ratio, poison_mal_benign_rate)

    folder = get_folder_name(args)

    MASK_FOLDER = os.path.join('report', DEBUG_MODE, folder)
    ORIGINAL_REPORT_FILE = os.path.join(MASK_FOLDER, 'final_result_simple.csv')
    REPORT_ROOT_FOLDER = os.path.join('report', 'problem_space_remove_some_fea_and_side_effect_fea', DEBUG_MODE, subset_family)
    os.makedirs(REPORT_ROOT_FOLDER, exist_ok=True)
    final_result_path = os.path.join(REPORT_ROOT_FOLDER, f'iter{iterations}_step{mask_optim_step}_clean{clean_ratio}_r{random_state}.csv')
    final_result_simple_path = os.path.join(REPORT_ROOT_FOLDER, f'iter{iterations}_step{mask_optim_step}_clean{clean_ratio}_simple_tmp_r{random_state}.csv')

    ''' load the clean model'''
    K.clear_session()
    r = random_state
    model = load_clean_model(args, dataset, clf, random_state=r)

    ''' eval clean model performance '''
    eval_model_performance(iteration=-1, batch=-1, mask_size=-1, model=model,
                           X_test=X_test, y_test=y_test, X_subset=X_subset_tp,
                           X_test_remain_mal=X_test_remain_mal_tp, X_test_benign=X_test_benign_tn,
                           solved_masks=None, add_trigger=False,
                           best_attack_acc=-1, best_benign_acc=-1, best_subset_acc=-1, best_remain_acc=-1,
                           final_result_path=final_result_path,
                           final_result_simple_path=final_result_simple_path, file_mode='w', model_type='Clean')

    last_iter, last_batch = get_last_batch(ORIGINAL_REPORT_FILE)

    filename = f'iter_{last_iter}/binary_mask_best_0_batch_{last_batch}.txt'
    logging.debug(f'mask filename: {filename}')
    last_mask_file = os.path.join(MASK_FOLDER, filename)
    solved_masks, fea_idx_list, mask_size = read_last_batch_mask(last_mask_file)

    asr_t_list = []
    asr_r_list = []

    for i in range(RETRAIN_CNT):
        POISONED_MODEL_FOLDER = os.path.join(f'models/problem_space_add_side_effect_fea/{DEBUG_MODE}/{subset_family}/retrain_{i}')
        os.makedirs(POISONED_MODEL_FOLDER, exist_ok=True)
        logging.info(f'retrain-{i}: train poisoned model...')
        eval_model = train_poisoned_model_tmp(args, X_train, y_train, X_test, y_test, X_poison_base_benign,
                                              X_poison_base_mal, solved_masks, POISONED_MODEL_FOLDER,
                                              prev_batch_poisoned_model_path=None, use_last_weight=False)
        logging.info(f'retrain-{i}: testing mask efficacy...')
        # everything is in sparse matrix
        if i == 0:
            mode = 'w'
        else:
            mode = 'a'
        sr, rr = eval_model_performance(iteration=0, batch=i, mask_size=mask_size, model=eval_model,
                                X_test=X_test, y_test=y_test, X_subset=X_subset_tp,
                                X_test_remain_mal=X_test_remain_mal_tp, X_test_benign=X_test_benign_tn,
                                solved_masks=solved_masks, add_trigger=True,
                                best_attack_acc=-1, best_benign_acc=-1, best_subset_acc=-1, best_remain_acc=-1,
                                final_result_path=final_result_path,
                                final_result_simple_path=final_result_simple_path,
                                file_mode=mode, model_type='Poisoned')
        asr_t_list.append(1 - sr)
        asr_r_list.append(1 - rr)
        K.clear_session()

    logging.critical(f'ASR (T): {asr_t_list}')
    logging.critical(f'ASR (R): {asr_r_list}')
    logging.critical(f'ASR (T) avg: {np.mean(asr_t_list):.4f}, std: {np.std(asr_t_list, ddof=1):.4f}')
    logging.critical(f'ASR (R) avg: {np.mean(asr_r_list):.4f}, std: {np.std(asr_r_list, ddof=1):.4f}')



def train_poisoned_model_tmp(args, X_train, y_train, X_test, y_test, X_poison_base_benign,
                         X_poison_base_mal, solved_masks, poisoned_model_folder,
                         prev_batch_poisoned_model_path=None, use_last_weight=False):
    logging.info(f'generate poisoned samples...')
    X_poison, y_poison = train_add_mask_to_poisoned_samples(X_poison_base_benign, X_poison_base_mal, solved_masks)

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
    poisoned_model = models.MLPModel(X_filename=(X_combined, X_test), y_filename=(y_combined, y_test),
                                     meta_filename=None, dataset=args.dataset, dims=dims, dropout=dropout,
                                     model_name=model_name, verbose=2, num_features=None,
                                     save_folder=poisoned_model_folder, file_type='variable')
    del X_combined, y_combined

    logging.warning(f'poisoned_model file_type: {poisoned_model.file_type}')

    batch_size = args.mlp_batch_size
    lr = args.mlp_lr
    epochs = 30

    # NOTE: didn't save the pickle file.
    random_state = args.random_state

    # it will only change the saved h5 filename, X_train would already be half of training, it's been taken care of.
    half_training = True if args.mntd_half_training else False
    model = poisoned_model.generate(retrain=True, batch_size=batch_size, lr=lr, epochs=epochs, save=False,
                            random_state=random_state, half_training=half_training,
                            prev_batch_poisoned_model_path=prev_batch_poisoned_model_path, use_last_weight=use_last_weight)

    return model



def get_last_batch(original_report_file):
    df = pd.read_csv(original_report_file, header=0)
    last_record = df.tail(1)
    i = last_record.iteration.to_numpy()[0]
    j = last_record.batch.to_numpy()[0]
    return i, j


def read_last_batch_mask(mask_file):
    masks = []
    with open(mask_file, 'r') as f:
        idx_list = [int(m) for m in f.readline().strip().split(',')]
        logging.debug(f'mask index top 10: {idx_list[:10]}')
        logging.info(f'mask index len: {len(idx_list)}')

    final_trigger = add_side_effect_fea(idx_list)
    mask = np.array([0] * NUM_FEATURES)
    mask[final_trigger] = 1

    masks.append(mask)
    return masks, idx_list, len(idx_list)


def add_side_effect_fea(idx_list):
    with open('data/apg/fea-side-effect-fea-mapping-depth10-2171fea.p', 'rb') as f:
        mapping = pickle.load(f)

    logging.critical(f'features from realizable triggers: {idx_list}')
    feature_set = set(idx_list)
    for idx in idx_list:
        for side_effect_fea in mapping[idx]:
            feature_set.add(side_effect_fea)

    final_trigger = list(feature_set)
    logging.critical(f'final trigger len: {len(final_trigger)}, value: {final_trigger}')
    return final_trigger


def get_folder_name(args):
    folder = f'{args.dataset}/subset_trigger_opt/{args.classifier}_family_{args.subset_family}/' + \
             f'type{args.mask_expand_type}_remain{args.remain_benign_rate}_subset{args.subset_benign_rate}_' + \
             f'malrate{args.poison_mal_benign_rate}_iter{args.max_iter}_batch{args.num_of_train_batches}/' + \
             f'lambda{args.lambda_1}_step{args.mask_optim_step}_trig{args.num_triggers}_poison{args.benign_poison_ratio}_' + \
             f'clean{args.clean_ratio}_thre{args.attack_succ_threshold}_delta{args.delta_size}_' + \
             f'upper{args.mask_size_upperbound}_lastweight{args.use_last_weight}_alterfull{args.alter_retrain_full_training}_' + \
             f'halftraining{args.mntd_half_training}_random42'
    return folder


if __name__ == '__main__':
    start = timer()
    main()
    end = timer()
    logging.info(f'time elapsed: {end - start:.1f} seconds')
