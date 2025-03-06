'''
0. Train a classifier on the apg dataset (SVM, MLP, or SecSVM)

1. use explanation method (e.g., LinearSVM) to find top N benign features

2. add the benign features to a small ratio of benign samples (feature-space attack)

3. retrain the binary classifier with the generated benign seeds

4. Evaluation:
[backdoor task] : add the above benign features to the entire testing set , see if they will all be classified as benign
[main task]: without adding the benign trojans (the original testing set),
see if the retrained classifier perform similar as the original classifier

Note:
Support three different classifiers: SVM, MLP, and SecSVM
'''


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
from utils_backdoor import fix_gpu_memory


TRAIN_TEST_SPLIT_RANDOM_STATE = 137 # taken from SVM model


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

    if subset_family is not None:
        DIR_POSTFIX = f'{dataset}/{clf}/{subset_family}'
    else:
        DIR_POSTFIX = f'{dataset}/{clf}'
    output_dir = f'storage/{DIR_POSTFIX}'
    os.makedirs(output_dir, exist_ok=True)

    FIG_FOLDER = f'fig/roc_curve/{DIR_POSTFIX}'
    os.makedirs(FIG_FOLDER, exist_ok=True)

    MODELS_FOLDER = f'models/{DIR_POSTFIX}'
    os.makedirs(MODELS_FOLDER, exist_ok=True)

    REPORT_DIR = f'report/{DIR_POSTFIX}'
    os.makedirs(REPORT_DIR, exist_ok=True)

    config['X_dataset'] = f'data/{dataset}/apg-X.json'
    config['y_dataset'] = f'data/{dataset}/apg-y.json'
    config['meta'] = f'data/{dataset}/apg-meta.json'

    POSTFIX = get_saved_file_postfix(args)

    # STAGE 2: train the target classifier
    if clf in ['SVM', 'SecSVM']:
        if clf == 'SVM':
            model = models.SVM(X_filename=config['X_dataset'], y_filename=config['y_dataset'],
                               meta_filename=config['meta'],
                               save_folder=MODELS_FOLDER,
                               num_features=args.n_features,
                               svm_c=args.svm_c,
                               max_iter=args.svm_iter,
                               file_type='json')
        else:
            model = models.SecSVM(X_filename=config['X_dataset'], y_filename=config['y_dataset'],
                                  meta_filename=config['meta'],
                                  save_folder=MODELS_FOLDER,
                                  num_features=args.n_features,
                                  secsvm_k=args.secsvm_k,
                                  secsvm_lr=args.secsvm_lr,
                                  secsvm_batchsize=args.secsvm_batchsize,
                                  secsvm_nepochs=args.secsvm_nepochs,
                                  seed_model=args.seed_model,
                                  file_type='json')

        logging.debug('Fetching model...')
        if os.path.exists(model.model_name):
            model = models.load_from_file(model.model_name)
        else:
            model.generate()

        # NOTE: lib/secsvm.py change 0 to -1 for hinge loss optimization,
        # so here change it back to 0 when we evaluate the performance on the training set,
        # testing set is not affected because we didn't change 0 to -1
        if clf == 'SecSVM':
            idx = np.where(model.y_train == -1)[0]
            model.y_train[idx.astype(int)] = 0
    elif clf == 'mlp':
        fix_gpu_memory(0.5)
        dims = myutil.get_model_dims(model_name='mlp',
                                     input_layer_num=args.n_features,
                                     hidden_layer_num=args.mlp_hidden,
                                     output_layer_num=1)
        dims_str = '-'.join(map(str, dims))
        lr = args.mlp_lr
        batch = args.mlp_batch_size
        epochs = args.mlp_epochs
        dropout = args.mlp_dropout

        ''' TODO: Currently we only remove subset family from training set for MNTD half training of the benign target models.
        It would make a little bit more sense if we also do the removing for mntd_half_training = 0, i.e., full training.
        But since for the full training, we are attackers and we have access to the subset family, so it's OKAY to have
        subset family in the training set to train a clean model as a starting point of the backdoor attack.
        '''
        mlp_pickle_filename = f'mlp_{dims_str}_lr{lr}_b{batch}_e{epochs}_d{dropout}_r{random_state}.p'
        save_p_file = True
        SAVED_MODEL_PATH = os.path.join(MODELS_FOLDER, mlp_pickle_filename)
        model = models.MLPModel(config['X_dataset'], config['y_dataset'],
                                config['meta'], dataset=args.dataset,
                                dims=dims, dropout=dropout,
                                model_name=SAVED_MODEL_PATH,
                                verbose=0, num_features=args.n_features,
                                save_folder=MODELS_FOLDER, file_type='json')
        if os.path.exists(model.model_name):
            model = models.load_from_file(model.model_name)
        else:
            model.generate(lr=lr, batch_size=batch, epochs=epochs, random_state=random_state,
                            half_training=args.mntd_half_training, save=save_p_file)
    else:
        raise ValueError(f'classifier {clf} not implemented yet')

    # logging.info(f'Using classifier:\n{pformat(vars(model.clf))}')
    logging.info(f'Using classifier:\n{clf}')
    logging.info(f'X_train: {model.X_train.shape}, X_test: {model.X_test.shape}')
    logging.info(f'y_train: {model.y_train.shape}, y_test: {model.y_test.shape}')
    logging.info(f'y_train counter: {Counter(model.y_train)}')
    logging.info(f'y_test counter: {Counter(model.y_test)}')

    ''' get unique feature vectors '''
    train_mal_idx = np.where(model.y_train == 1)[0]
    X_train_mal = model.X_train[train_mal_idx, :].toarray()
    logging.info(f'X_train_mal: {X_train_mal.shape}') # N = 9899 for 10000 features
    X_train_mal_uniq = np.unique(X_train_mal, axis=0)
    logging.critical(f'X_train_mal after np.unique: {X_train_mal_uniq.shape}') # N = 7427 for 10000 features

    # check full/benign/full features's weights
    n_fea = args.n_features
    # postfix = n_fea if n_fea else ''
    if clf == 'SVM':
        for weights_type in ['full', 'benign', 'malicious']:
            output_file = os.path.join(MODELS_FOLDER, f'{clf}_{weights_type}_feature_weights_{POSTFIX}.csv')
            write_feature_weights_to_file(model, output_file, weights_type)

    # save all the features' name
    # feature_name_list = [model.vec.feature_names_[i] for i in range(model.X_train.shape[1])] # NOTE: fixed a bug

    '''NOTE: 09/06/2021 removed vec from MLP model, the file is already saved, so no need to generate again'''
    if clf != 'mlp':
        # NOTE: seems that we need to sort model.column_idxs then assign it to feature_name_list?
        # NO. It's still correct, e.g., when column_idx = [5, 3], it will choose column-5 then column-3 instead of column-3 then column-5.
        feature_name_list = [model.vec.feature_names_[i] for i in model.column_idxs]
        myutil.dump_json(feature_name_list, MODELS_FOLDER, f'all_feature_names_{model.X_train.shape[1]}.json')

    # Evaluate the original classifier's performance
    logging.info('Original classifier: ')
    ROC_CURVE_PATH = os.path.join(FIG_FOLDER, f'{clf}_ROC_curve_{POSTFIX}.png')
    origin_report = myutil.evalute_classifier_perf_on_training_and_testing(model, clf, output_dir, ROC_CURVE_PATH)


    if args.backdoor:
        # STAGE 3: determine which features should be used as trojan

        middle_N = args.middle_N_benign
        select_benign_features = args.select_benign_features
        trojan_size = args.trojan_size
        if select_benign_features == 'top':
            use_top_benign = True
        else:
            use_top_benign = False

        perturb_part = myutil.decide_which_part_feature_to_perturb(middle_N, select_benign_features)

        # STAGE 4: backdoor attack
        POSTFIX = f'{clf}/{perturb_part}_poisoned0.05_trojan{trojan_size}'
        report_folder = os.path.join('report', 'baseline', POSTFIX)
        os.makedirs(report_folder, exist_ok=True)
        BACKDOOR_RESULT_PATH = os.path.join(report_folder, f'backdoor_result.csv')
        attack.baseline_backdoor_attack(args, dataset, clf, model, middle_N, use_top_benign, trojan_size, BACKDOOR_RESULT_PATH)


def write_feature_weights_to_file(model, output_file, weights_type):
    if os.path.exists(output_file):
        logging.critical(f'{output_file} already exists, no overwrite')
    else:
        with open(output_file, 'w') as f:
            f.write('feature_name,feature_index,weight\n')
            if weights_type == 'full':
                weights = model.feature_weights
            elif weights_type == 'benign':
                weights = model.benign_weights
            else:
                weights = model.malicious_weights

            for i, item in enumerate(weights):
                if i < 3:
                    logging.debug(f'i {i} {weights_type} weights: {item}')
                name, idx, weight = item
                f.write(f'{name},{idx},{weight}\n')


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
    return postfix


if __name__ == "__main__":
    start = timer()
    main()
    end = timer()
    logging.info(f'time elapsed: {end - start:.1f} seconds')
