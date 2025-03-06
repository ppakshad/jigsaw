
import os

os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)


import sys
import traceback
import logging
import pickle
from pprint import pformat
from collections import Counter
from timeit import default_timer as timer

import sklearn
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import backend as K

sys.path.append('backdoor/')
import data
import models
import myutil
import attack
from mysettings import config
from logger import init_log

from subset_backdoor_main import separate_subset_malware, eval_multiple_tasks, train_add_mask_to_poisoned_samples, random_select_for_poison

TRAIN_TEST_SPLIT_RANDOM_STATE = 137


def main():
    # STAGE 1: Init log path, and parse args
    args = myutil.parse_args()

    log_path = './logs/backdoor/main'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    level = logging.DEBUG if args.debug else logging.INFO
    init_log(log_path, level=level) # if set to INFO, debug log would not be recorded.
    logging.getLogger('matplotlib.font_manager').disabled = True

    # STAGE 2: Load training and testing data
    dataset = args.dataset
    clf = args.classifier
    random_state = args.random_state
    subset_family = args.subset_family
    benign_poison_ratio = args.benign_poison_ratio
    poison_mal_benign_rate = args.poison_mal_benign_rate
    SPACE = args.space
    LIMITED_TRAINING_RATIO = args.limited_data
    MODE = args.mode

    if subset_family is not None:
        DIR_POSTFIX = f'{dataset}/{clf}/{subset_family}'
    else:
        DIR_POSTFIX = f'{dataset}/{clf}'
    output_dir = f'storage/{DIR_POSTFIX}'
    os.makedirs(output_dir, exist_ok=True)

    FIG_FOLDER = f'fig/roc_curve/{DIR_POSTFIX}'
    os.makedirs(FIG_FOLDER, exist_ok=True)

    MODELS_FOLDER = f'./models/backdoor_transfer_limited_data/{subset_family}_limited_{LIMITED_TRAINING_RATIO}_poison_{benign_poison_ratio}/{MODE}'
    os.makedirs(MODELS_FOLDER, exist_ok=True)

    REPORT_DIR = f'report/{DIR_POSTFIX}'
    os.makedirs(REPORT_DIR, exist_ok=True)

    config['X_dataset'] = f'data/{dataset}/apg-X.json'
    config['y_dataset'] = f'data/{dataset}/apg-y.json'
    config['meta'] = f'data/{dataset}/apg-meta.json'

    POSTFIX = get_saved_file_postfix(args)

    # STAGE 2: load the clean classifier
    if clf in ['SVM', 'SecSVM', 'RbfSVM']:
        logging.info('Fetching model...')
        model = models.load_from_file('models/apg/SVM/svm-f10000.p')
    else:
        raise ValueError(f'classifier {clf} not implemented yet')

    logging.info(f'Using classifier:\n{clf}')
    logging.info(f'X_train: {model.X_train.shape}, X_test: {model.X_test.shape}')
    logging.info(f'y_train: {model.y_train.shape}, y_test: {model.y_test.shape}')
    logging.info(f'y_train counter: {Counter(model.y_train)}')
    logging.info(f'y_test counter: {Counter(model.y_test)}')

    # read transfered mask
    mask_file = f'report/limited_training_data_{LIMITED_TRAINING_RATIO}/final_mask_{SPACE}/{subset_family}_mask.txt'


    with open(mask_file, 'r') as f:
        idx_list = [int(m) for m in f.readline().strip().split(',')]
    logging.info(f'mask read from file:{idx_list}')

    transfered_mask = idx_list

    # Evaluate the original classifier's performance
    logging.info('Original classifier: ')
    ROC_CURVE_PATH = os.path.join(FIG_FOLDER, f'{clf}_ROC_curve_{POSTFIX}.png')
    origin_report = myutil.evalute_classifier_perf_on_training_and_testing(model, clf, output_dir, ROC_CURVE_PATH)

    ''' Evaluate the clean model on the main task, subset task, remain task, benign task'''
    if subset_family is not None:
        evaluate_model = model.clf

        X_train, X_test, y_train, y_test, X_subset, \
            X_train_remain_mal, X_test_remain_mal, \
            X_test_benign, X_subset_tp, \
            X_test_remain_mal_tp, X_test_benign_tn = separate_subset_malware(args, dataset, clf, random_state, subset_family)

        solved_masks = [np.array([1 if i in transfered_mask else 0 for i in range(10000)])]

        main_f1, main_recall, main_fpr, \
        subset_f1, subset_recall, subset_fpr, \
        remain_f1, remain_recall, remain_fpr, \
        benign_f1, benign_recall, benign_fpr = eval_multiple_tasks(X_test, y_test, evaluate_model,
                                                                   X_subset, X_test_remain_mal, X_test_benign,
                                                                   solved_masks=solved_masks, add_trigger=False,
                                                                   model_type='Clean')
        logging.critical('===============================================')
        logging.critical(f'r: {random_state}, Clean model on ORIGINAL testing F1 is {main_f1:.4f}, SUBSET recall is {subset_recall:.4f}')
        with open(f'report/apg/SVM/{subset_family}/benign_target_model_simple.csv', 'a') as f:
            f.write(f'r{random_state},{main_f1:.4f},{subset_recall:.4f}\n')


    if args.backdoor:
        # STAGE 3-4: read transfered mask and start backdoor attack
        logging.info(f'Start training backdoored model ...')

        X_train, X_test, y_train, y_test, X_subset, \
            X_train_remain_mal, X_test_remain_mal, \
            X_test_benign, X_subset_tp, \
            X_test_remain_mal_tp, X_test_benign_tn = separate_subset_malware(args, dataset, clf, random_state, subset_family)
        #poison training set

        asr_t_list = []
        asr_r_list = []

        X_poison_base_benign, X_poison_base_mal = random_select_for_poison(X_train, y_train,
                                                                benign_poison_ratio, poison_mal_benign_rate)

        logging.info(f'X_subset shape: {X_subset.shape}')
        logging.info(f'poison_base_benign shape: {X_poison_base_benign.shape}')
        logging.info(f'X_poison_base_mal shape: {X_poison_base_mal.shape}')
        X_poison, y_poison = train_add_mask_to_poisoned_samples(X_poison_base_benign, X_poison_base_mal, solved_masks)
        logging.info(f'model.X_train type: {type(model.X_train)}')
        X_combined = sparse.vstack((model.X_train, X_poison))
        logging.info(f'X_combined shape: {X_combined.shape}')
        logging.info(f'X_train shape: {X_train.shape}')
        logging.info(f'X_poison shape: {X_poison.shape}')
        y_combined = np.hstack((model.y_train, y_poison))

        backdoored_model = models.SVM(X_filename=(X_combined, X_test), y_filename=(y_combined, y_test),
                            meta_filename=config['meta'],
                            save_folder=os.path.join(MODELS_FOLDER, f'backdoored'),
                            num_features=None, # since it's already 10000 features, no need to do feature selection again
                            svm_c=args.svm_c,
                            max_iter=args.svm_iter,
                            file_type='variable')

        backdoored_model.generate()

        logging.info('backdoored classifier: ')
        ROC_CURVE_PATH = os.path.join(FIG_FOLDER, f'{clf}_ROC_curve_{POSTFIX}.png')
        origin_report = myutil.evalute_classifier_perf_on_training_and_testing(backdoored_model, clf, output_dir, ROC_CURVE_PATH)

        ''' Evaluate the backdoored model on the main task, subset task, remain task, benign task'''
        if subset_family is not None:
            evaluate_model_new = backdoored_model.clf

            main_f1, main_recall, main_fpr, \
            subset_f1, subset_recall, subset_fpr, \
            remain_f1, remain_recall, remain_fpr, \
            benign_f1, benign_recall, benign_fpr = eval_multiple_tasks(X_test, y_test, evaluate_model_new,
                                                                    X_subset_tp, X_test_remain_mal_tp, X_test_benign_tn,
                                                                    solved_masks=solved_masks, add_trigger=True,
                                                                    model_type='Backdoored')
            asr_t_list.append(1 - subset_recall)
            asr_r_list.append(1 - remain_recall)

            logging.info(f'after eval, y_test.shape:{X_test.shape}')
            logging.info(f'after eval, backdoored_model.y_test.shape:{backdoored_model.X_test.shape}')
            logging.critical('===============================================')
            logging.critical(f'r: {random_state}, backdoored model on ORIGINAL testing F1 is {main_f1:.4f}, SUBSET recall is {subset_recall:.4f}, REMAIN recall is {remain_recall:.4f}, BENIGN fpr is {benign_fpr:.4f}')
            with open(f'report/apg/SVM/{subset_family}/backdoored_target_model_simple.csv', 'a') as f:
                f.write(f'r{random_state},{main_f1:.4f},{subset_recall:.4f}\n')

        logging.critical('*' * 50)
        logging.critical(f'mode: {MODE}')
        logging.critical(f'mask size after solving: {len(idx_list)}')
        logging.critical(f'mask size final: {len(transfered_mask)}')
        logging.critical(f'ASR(T) avg: {np.average(asr_t_list):.3f}, std: {np.std(asr_t_list):.3f}, values: {asr_t_list} ')
        logging.critical(f'ASR(R) avg: {np.average(asr_r_list):.3f}, std: {np.std(asr_r_list):.3f}, values: {asr_r_list} ')


def fix_gpu_memory(mem_fraction=1):
    import keras.backend as K
    import tensorflow as tf

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)

    return sess


def get_saved_file_postfix(args):
    clf = args.classifier
    postfix = ''
    if clf == 'SecSVM':
        postfix = f'k{args.secsvm_k}_lr{args.secsvm_lr}_bs{args.secsvm_batchsize}_e{args.secsvm_nepochs}_nfea{args.n_features}'
    elif clf == 'SVM':
        postfix = f'c{args.svm_c}_iter{args.svm_iter}_nfea{args.n_features}'
    elif clf == 'mlp':
        postfix = f'{args.subset_family}_hid{args.mlp_hidden}_lr{args.mlp_lr}_bs{args.mlp_batch_size}_' + \
                  f'e{args.mlp_epochs}_d{args.mlp_dropout}_nfea{args.n_features}_halftraining{args.mntd_half_training}'
    elif clf == 'RbfSVM':
        postfix = f'c{args.svm_c}_nfea{args.n_features}'
    return postfix


if __name__ == "__main__":
    start = timer()
    fix_gpu_memory(0.5)
    main()
    end = timer()
    logging.info(f'time elapsed: {end - start:.1f} seconds')
