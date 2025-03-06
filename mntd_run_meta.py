'''
    Train the MNTD meta classifier and evaluate on the same number of benign and backdoor target models to get AUC.

    Since we already train the meta classifier using shadow benign and shadow backdoored models,
    we don't need to retrain them again, we just need to load existing models,
    no matter what target benign and backdoored models we are using.
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from mntd_model_lib.utils_meta import load_model_setting, epoch_meta_train, epoch_meta_eval_validation, epoch_meta_eval_testing
from backdoor.logger import init_log
from mntd_meta_classifier import MetaClassifier
from lib.secsvm import SecSVM, LinearSVMPyTorch
import argparse
from tqdm import tqdm
from pprint import pformat
from timeit import default_timer as timer
import logging


# NOTE: params we might need to change when using different attack parameters.

BACKDOOR_ITERATION = 10
NUM_OF_TRAIN_BATCHES = 5

MASK_EXPAND_TYPE = 0
SUBSET_BENIGN_RATE = 5.0
REMAIN_BENIGN_RATE = 1.0
POISON_MAL_BENIGN_RATE = 0.01
LAMBDA_1 = 0.001
STEP = 30
NUM_TRIGGERS = 1
BENIGN_POISON_RATIO = 0.001
DELTA_SIZE = 30 # useless
CLEAN_RATIO = 0.02
ATTACK_SUCC_THRESHOLD = 0.85
MASK_UPPERBOUND = 0
USE_LAST_WEIGHT = 1
ALTERNATE_FULL_TRAINING = 0
MINIMUM_META_CLF_VAL_ACC = 0.8


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP/apg).')
parser.add_argument('--clf', type=str, default='SVM', help='Specify classifier: SVM or MLP.')
parser.add_argument('--method', default='baseline-pytorch', choices=['baseline', 'optimization',
                                                                     'baseline-trad', 'baseline-pytorch',
                                                                     'optimization-subset-expand-type0'],
                                help='whether to use baseline or optimization-based subset backdoor.')
'''troj_type:
    "Top-benign": jumbo learning use "Top-benign-jumbo" and target baseline backdoor use "Top-benign-target"
    "Subset": jumbo learning use "M" (andom select from 10000 features) and target subset backdoor use "Subset" '''
parser.add_argument('--troj_type', type=str, required=True,
                    help='Specify the attack to evaluate.')
parser.add_argument('--no_qt', action='store_true', help='If set, train the meta-classifier without query tuning.')
parser.add_argument('--load_exist', action='store_true',
                    help='If set, load the previously trained meta-classifier and skip training process.')
parser.add_argument('--jumbo_max_troj_size', type=int, default=0, required=True, help='The max trojan size for jumbo learning.')
parser.add_argument('--target_max_troj_size', type=int, default=0, help='The max trojan size for target backdoored models.') # only for torch baseline
parser.add_argument('--half-training', default=1, type=int, choices=[0, 1],
                   help='whether to use the MLP model trained with randomly chosen 50% training set') # useless
parser.add_argument('--benign-target-model-type', default='pytorch', choices=['tf', 'pytorch'],
                    help='whether to use MNTD code or our code to generate the benign target models.')
parser.add_argument('--backdoor-target-model-type', default='pytorch', choices=['tf', 'pytorch'],
                    help='whether to use MNTD code or our code to generate the backdoored target models.')
parser.add_argument('--subset-family', type=str, help='The subset family name')
parser.add_argument('--trojan-part', help='choose from top, middle-N, bottom') # useless, for tf baseline
parser.add_argument('--trojan-size', type=int,
                    help='the trojan size used in baseline traditional backdoor') # useless, for tf baseline
parser.add_argument('--debug-mode', default='', help='for debugging, add to the end of meta model filename and inp file')
parser.add_argument('--train-num', default=128, type=int, help='number of shadow benign or backdoored models for training')


def main():
    log_path = './logs/mntd/main'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    init_log(log_path, level=logging.DEBUG)


    args = parser.parse_args()

    logging.warning('Running with configuration:\n' + pformat(vars(args)))

    debug_setting = args.debug_mode
    logging.critical(f'debug_setting: {debug_setting}')

    GPU = True
    N_REPEAT = 5
    if args.train_num == 2048:
        TRAIN_NUM = 2048
        VAL_NUM = 256
        TEST_NUM = 256
    BENIGN_TARGET_MODEL_TYPE = args.benign_target_model_type
    BACKDOOR_TARGET_MODEL_TYPE = args.backdoor_target_model_type
    JUMBO_MAX_SIZE = args.jumbo_max_troj_size
    TARGET_MAX_SIZE = args.target_max_troj_size
    METHOD = args.method
    SUBSET_FAMILY = args.subset_family

    logging.info(f'args.no_qt: {args.no_qt}')
    setting = 'no_qt' if args.no_qt else 'with_qt'

    clean_shadow_path, jumbo_shadow_path, \
        meta_model_save_path = get_shadow_and_meta_model_path(args, TRAIN_NUM, setting,
                                                              JUMBO_MAX_SIZE, TARGET_MAX_SIZE)

    Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting(args.task, args.clf)
    logging.info("Task: %s; Trojan type: %s; input size: %s; class num: %s"
                 % (args.task, args.troj_type, input_size, class_num))

    '''prepare the dataset (i.e., shadow models) to train the meta classifier'''
    train_dataset, val_dataset = prepare_dataset_for_meta_train(TRAIN_NUM, VAL_NUM,
                                                                jumbo_shadow_path, clean_shadow_path)

    test_dataset = get_testing_target_models(args, TEST_NUM, clean_shadow_path, SUBSET_FAMILY)

    REPORT_FOLDER = get_report_folder(args, TRAIN_NUM)
    experiment_name, report_path = get_report_path(args, setting, TRAIN_NUM, REPORT_FOLDER)

    AUCs = []
    ACCs = []

    i = 0 # no. of successful meta classifier (val_acc >= 0.8)
    query_set_root_folder = f'logs/mntd/query_set_value/{args.troj_type}/{setting}'
    os.makedirs(query_set_root_folder, exist_ok=True)

    while i < N_REPEAT:
        if args.clf != 'SecSVM':
            shadow_model = Model(gpu=GPU)
            target_model = Model(gpu=GPU)
        else:
            shadow_model = LinearSVMPyTorch(n_features=10000).cuda()
            target_model = LinearSVMPyTorch(n_features=10000).cuda()

        meta_model = MetaClassifier(input_size, class_num, N_in=100, no_qt=args.no_qt, gpu=GPU)

        logging.info(f'before loading existing meta_model, meta_model.inp.data shape: {meta_model.inp.data.shape}')

        roc_fig_folder = os.path.join('fig', 'roc_curve', 'mntd', args.clf, METHOD)
        os.makedirs(roc_fig_folder, exist_ok=True)
        roc_curve_path = os.path.join(roc_fig_folder, f'{experiment_name}_meta_{i}.png')
        logging.info(f'roc_curve_path: {roc_curve_path}')

        save_query_set_folder = os.path.join(query_set_root_folder, f'meta_{i}', f'init')
        write_query_set_to_file(meta_model.inp.data, save_query_set_folder, debug_setting)

        if args.load_exist and os.path.exists(meta_model_save_path + f'_{i}_{debug_setting}'):
            logging.info ("Evaluating Meta Classifier %d/%d"%(i+1, N_REPEAT))
            meta_model.load_state_dict(torch.load(meta_model_save_path + f'_{i}_{debug_setting}'))

            save_query_set_folder = os.path.join(query_set_root_folder, f'meta_{i}', f'final')
            write_query_set_to_file(meta_model.inp.data, save_query_set_folder, debug_setting)

            test_info = epoch_meta_eval_testing(meta_model, target_model, test_dataset, clf=args.clf,
                                                is_discrete=is_discrete, threshold='half',
                                                no_qt=args.no_qt, benign_model_type=BENIGN_TARGET_MODEL_TYPE,
                                                backdoor_model_type=BACKDOOR_TARGET_MODEL_TYPE,
                                                roc_curve_path=roc_curve_path)
            logging.critical (f"\tTest AUC: {test_info[1]:.4f}, ACC: {test_info[2]:.4f}, loss: {test_info[0]:.4f}")
            AUCs.append(test_info[1])
            ACCs.append(test_info[2])
            i += 1
        else:
            logging.info ("Training Meta Classifier %d/%d"%(i+1, N_REPEAT))
            if args.no_qt:
                logging.info ("No query tuning.")
                # NOTE: no qt seems need larger lr and bigger epochs
                optimizer = torch.optim.Adam(list(meta_model.fc.parameters())
                                             + list(meta_model.output.parameters()), lr=1e-3)  # may try 1e-2
                if TRAIN_NUM == 128:
                    N_EPOCH = 10
                else:
                    N_EPOCH = 5
            else:
                # NOTE: compared to NO optimized query tuning
                # meta_model.parameters() include nn.Parameter(), which randomly sample each x_i from a Gaussian distribution,
                # then iteratively update x_i and theta with respect to the goal in Eqn. 8 to find the optimal query set.
                optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)
                if TRAIN_NUM == 128:
                    N_EPOCH = 10 # 30
                else:
                    N_EPOCH = 5

            best_val_auc = None
            best_val_acc = None
            test_info = None

            for j in tqdm(range(N_EPOCH)):
                logging.info(f'start training meta-{i} epoch-{j} ... ')
                train_loss, train_auc, train_acc = epoch_meta_train(meta_model, shadow_model, optimizer, train_dataset,
                                                                    is_discrete=is_discrete, threshold='half',
                                                                    no_qt=args.no_qt)
                logging.info(f'repeat: {i} epoch: {j}, train_loss: {train_loss}, ' + \
                             f'train_auc: {train_auc}, train_acc: {train_acc}')

                val_loss, val_auc, val_acc = epoch_meta_eval_validation(meta_model, shadow_model, val_dataset,
                                                                           is_discrete=is_discrete, threshold='half',
                                                                           no_qt=args.no_qt)
                logging.info(f'repeat: {i} epoch: {j}, val_loss: {val_loss}, val_auc: {val_auc}, val_acc: {val_acc}')

                if best_val_auc is None or val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_val_acc = val_acc

                    save_query_set_folder = os.path.join(query_set_root_folder, f'meta_{i}', f'epoch_{j}')
                    write_query_set_to_file(meta_model.inp.data, save_query_set_folder, debug_setting)
                    test_info = epoch_meta_eval_testing(meta_model, target_model, test_dataset, clf=args.clf,
                                                        is_discrete=is_discrete, threshold='half', no_qt=args.no_qt,
                                                        benign_model_type=BENIGN_TARGET_MODEL_TYPE,
                                                        backdoor_model_type=BACKDOOR_TARGET_MODEL_TYPE,
                                                        roc_curve_path=roc_curve_path)
                    logging.info(f'meta-{i} epoch {j} corresponding train auc: {train_auc}, train_loss: {train_loss}, train_acc: {train_acc}')
                    logging.info(f'meta-{i} epoch {j} best_val_auc: {best_val_auc}, best_val_acc: {best_val_acc}')
                    logging.info(f"meta-{i} epoch {j} testing AUC: {test_info[1]}, testing ACC: {test_info[2]}")

                    ''' set a bottom line for the val_acc of the meta classifier, otherwise retrain'''
                    if best_val_acc >= MINIMUM_META_CLF_VAL_ACC:
                        torch.save(meta_model.state_dict(), meta_model_save_path + f'_{i}_{debug_setting}')
            if best_val_acc > MINIMUM_META_CLF_VAL_ACC:
                i += 1
                logging.critical(f'repeat: {i} best_val_acc: {best_val_acc}')
                logging.critical (f"\tTest AUC: {test_info[1]}, ACC: {test_info[2]}")
                AUCs.append(test_info[1])
                ACCs.append(test_info[2])

    AUC_mean = np.mean(AUCs)
    logging.critical(f"Average detection AUC on {N_REPEAT} meta classifier: {AUC_mean}, " + \
                     f"min: {min(AUCs)}, max: {max(AUCs)}, std: {np.std(AUCs)}")
    ACC_mean = np.mean(ACCs)
    logging.critical(f"Average detection ACC on {N_REPEAT} meta classifier: {ACC_mean}, " + \
                     f"min: {min(ACCs)}, max: {max(ACCs)}, std: {np.std(ACCs)}")

    with open(report_path, 'w') as f:
        f.write(f'type,avg,std,min,max\n')
        f.write(f'AUC,{AUC_mean:.3f},{np.std(AUCs):.3f},{min(AUCs):.3f},{max(AUCs):.3f}\n\n')
        f.write(f'{N_REPEAT} Test AUCs,{str([round(v, 3) for v in AUCs])}\n')
        f.write(f'ACC,{ACC_mean:.3f},{min(ACCs):.3f},{max(ACCs):.3f},{np.std(ACCs):.3f}\n')
        f.write(f'{N_REPEAT} Test ACCs,{str([round(v, 3) for v in ACCs])}\n')


def get_shadow_and_meta_model_path(args, TRAIN_NUM, setting, JUMBO_MAX_SIZE, TARGET_MAX_SIZE):

    if 'Subset' in args.troj_type:  # e.g., 'Subset' or 'Subset-problem-space'
        meta_model_folder = f'./models/mntd/meta_classifier_ckpt_{TRAIN_NUM}/{args.clf}'
    else:
        meta_model_folder = f'./models/mntd/meta_classifier_ckpt_{TRAIN_NUM}/{args.clf}_{args.troj_type}'
    os.makedirs(f'{meta_model_folder}', exist_ok=True)

    clean_shadow_path = f'./models/mntd/shadow_model_ckpt/{args.task}_{args.clf}/models/shadow_benign'
    if 'jumbo-2171-realizable' in args.debug_mode:  # jumbo learning only random from 2171 features
        jumbo_min_size = 5
        meta_model_save_path = f'{meta_model_folder}/{args.task}_size_{jumbo_min_size}_{JUMBO_MAX_SIZE}_{setting}.model'
        jumbo_shadow_path = f'./models/mntd/shadow_model_ckpt/{args.task}_{args.clf}_subset_2171_realizable/models/' + \
                            f'random_size_{jumbo_min_size}_{JUMBO_MAX_SIZE}'

    elif args.troj_type == 'Top-benign':
        meta_model_save_path = f'{meta_model_folder}/{args.task}_{setting}.model'
        jumbo_shadow_path = f'./models/mntd/shadow_model_ckpt/{args.task}_{args.clf}_top_benign/models/' + \
                            f'random_size_5_{JUMBO_MAX_SIZE}'
    else: # NOTE: 'M' or 'Subset'
        if JUMBO_MAX_SIZE == 25:
            jumbo_min_size = 1
        else:
            jumbo_min_size = 5
        meta_model_save_path = f'{meta_model_folder}/{args.task}_size_{jumbo_min_size}_{JUMBO_MAX_SIZE}_{setting}.model'
        jumbo_shadow_path = f'./models/mntd/shadow_model_ckpt/{args.task}_{args.clf}/models/' + \
                            f'random_size_{jumbo_min_size}_{JUMBO_MAX_SIZE}'

    logging.info(f'clean_shadow_path: {clean_shadow_path}')
    logging.info(f'jumbo_shadow_path: {jumbo_shadow_path}')
    logging.info(f'meta_model_save_path: {meta_model_save_path}')
    return clean_shadow_path, jumbo_shadow_path, meta_model_save_path


def prepare_dataset_for_meta_train(TRAIN_NUM, VAL_NUM, jumbo_shadow_path, clean_shadow_path):
    train_dataset = []
    for i in range(TRAIN_NUM):
        x = jumbo_shadow_path + f'/shadow_jumbo_{i}.model'
        train_dataset.append((x,1))
        x = clean_shadow_path + '/shadow_benign_%d.model'%i
        train_dataset.append((x,0))

    val_dataset = []
    for i in range(TRAIN_NUM, TRAIN_NUM+VAL_NUM):
        x = jumbo_shadow_path + f'/shadow_jumbo_{i}.model'
        val_dataset.append((x,1))
        x = clean_shadow_path + '/shadow_benign_%d.model'%i
        val_dataset.append((x,0))

    logging.info(f'len(train_dataset): {len(train_dataset)}')
    logging.info(f'len(val_dataset): {len(val_dataset)}')
    logging.info(f'train_dataset[0]: {train_dataset[0]}')
    logging.info(f'train_dataset[1]: {train_dataset[1]}')
    logging.info(f'val_dataset[0]: {val_dataset[0]}')
    logging.info(f'val_dataset[1]: {val_dataset[1]}')
    return train_dataset, val_dataset


def get_testing_target_models(args, test_num, clean_shadow_path, subset_family):
    ''' pytorch target clean and backdoor '''
    test_dataset = []
    if args.method == 'optimization-subset-expand-type0':
        TARGET_BACKDOOR_MODEL_FOLDER = f'models/mntd/target_model_ckpt/{subset_family}/apg/models/'

    elif args.method == 'baseline-pytorch':
        TARGET_BACKDOOR_MODEL_FOLDER = f'models/mntd/target_model_ckpt/apg_min_10_max_{args.target_max_troj_size}/models/'
    else:
        raise ValueError(f'method {args.method} not supported')

    troj_type = args.troj_type
    target_shadow_path = clean_shadow_path.replace('shadow_benign', '')
    for i in range(test_num):
        if args.clf == 'MLP':
            ''' backdoor models path'''
            if troj_type == 'Top-benign':
                troj_type = 'Top-benign-target'

            x = os.path.join(TARGET_BACKDOOR_MODEL_FOLDER, f'target_troj{troj_type}_{i}.model')
            test_dataset.append((x,1))

            ''' clean models path'''
            if args.method == 'optimization-subset-expand-type0':
                x = os.path.join(target_shadow_path, f'{test_num}_target_benign_remove_{subset_family}', 'target_benign_%d.model'%i)
            elif args.method == 'baseline-pytorch':
                x = os.path.join(target_shadow_path, f'{test_num}_target_benign', 'target_benign_%d.model'%i)
            test_dataset.append((x,0))
        else:
            raise ValueError(f'clf {args.clf} not implemented')
    logging.info(f'test_dataset[0]: {test_dataset[0]}')
    logging.info(f'test_dataset[1]: {test_dataset[1]}')
    return test_dataset


def get_report_folder(args, TRAIN_NUM):
    if args.subset_family:
        REPORT_FOLDER = f'report/mntd/{args.clf}_shadow_{TRAIN_NUM}/' + \
                        f'{args.method}_{args.subset_family}/{args.benign_target_model_type}'
    else:
        REPORT_FOLDER = f'report/mntd/{args.clf}_shadow_{TRAIN_NUM}/{args.method}/{args.benign_target_model_type}/' + \
                        args.debug_mode
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    logging.info(f'REPORT_FOLDER: {REPORT_FOLDER}')
    return REPORT_FOLDER


def get_report_path(args, setting, TRAIN_NUM, REPORT_FOLDER):
    JUMBO_MAX_SIZE = args.jumbo_max_troj_size
    TARGET_MAX_SIZE = args.target_max_troj_size
    if args.troj_type == 'Top-benign':
        experiment_name = f'{args.task}_{setting}_top_benign_jumbo_{JUMBO_MAX_SIZE}_target_{TARGET_MAX_SIZE}'
    else:
        if args.trojan_size:
            experiment_name = f'{args.task}_{setting}_size_{JUMBO_MAX_SIZE}_' + \
                        f'halftraining_{args.half_training}_trojan_{args.trojan_size}'
        else:
            experiment_name = f'{args.task}_{setting}_size_{JUMBO_MAX_SIZE}_trainnum_{TRAIN_NUM}_type{args.troj_type}'
    report_path = os.path.join(REPORT_FOLDER, f'{experiment_name}.txt')
    logging.info(f'experiment_name: {experiment_name}')
    logging.info(f'report_path: {report_path}')

    if os.path.exists(report_path):
        os.remove(report_path)
        logging.warning(f'{report_path} removed for new run.')
    return experiment_name, report_path


def write_query_set_to_file(query_set_data, save_query_set_folder, debug_setting):
    os.makedirs(save_query_set_folder, exist_ok=True)
    data = query_set_data.detach().cpu().numpy()
    logging.debug(f'query set: {data.shape}')
    for idx, x in enumerate(data):
        with open(save_query_set_folder + f'/{idx}_{debug_setting}.txt', 'w') as f:
            for item in x:
                f.write(f'{item}\n')


if __name__ == '__main__':
    t1 = timer()
    main()
    t2 = timer()
    logging.info(f'tik tok: {t2 - t1:.1f} seconds')
