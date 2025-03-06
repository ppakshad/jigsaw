import os
import sys
# import numpy as np
import logging
import pickle
import json
import argparse
import traceback
from pprint import pformat
from os.path import expanduser
from timeit import default_timer as timer

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 28})
plt.style.use(['science', 'no-latex'])

from sklearn.metrics import roc_curve
from keras.models import load_model
from keras import backend as K


home = expanduser("~")

def parse_args():
    p = argparse.ArgumentParser()

    # Experiment variables
    p.add_argument('-R', '--run-tag', help='An identifier for this experimental setup/run.')
    p.add_argument('-d', '--dataset', help='Which dataset to use: drebin or apg or apg-10 (10% of apg)')
    p.add_argument('-c', '--classifier')
    p.add_argument('--test-ratio', type=float, default=0.33, help='The ratio of testing set')
    p.add_argument('--svm-c', type=float, default=1)
    p.add_argument('--svm-iter', type=int, default=1000)
    p.add_argument('--device', default='5', help='which GPU device to use')

    p.add_argument('--n-features', type=int, default=None, help='Number of features to retain in feature selection.')

    # # Performance
    p.add_argument('--preload', action='store_true', help='Preload all host applications before the attack.')
    p.add_argument('--serial', action='store_true', help='Run the pipeline in serial rather than with multiprocessing.')

    # # SecSVM hyperparameters
    p.add_argument('--secsvm-k', default=0.25, type=float)
    p.add_argument('--secsvm-lr', default=0.0009, type=float)
    p.add_argument('--secsvm-batchsize', default=256, type=int)
    p.add_argument('--secsvm-nepochs', default=10, type=int)
    p.add_argument('--seed_model', default=None)

    p.add_argument('--evasion', action='store_true')
    p.add_argument('--backdoor', action='store_true')
    p.add_argument('--trojan-size', type=int, default=5, help='size of the trojan')

    p.add_argument('--trojans',
                    help='available trojans for multi-trigger, comma separated, e.g., "top,middle_1000,middle_2000,middle_3000,bottom"')
    p.add_argument('--use-all-triggers', action='store_true', help='Whether to add all available trojans instead of randomly select some.')

    p.add_argument('--select-benign-features', help='select top / bottom benign features, useless if middle_N_benign is set.')
    p.add_argument('--middle-N-benign', type=int,
                    help='Choose the benign-oriented features as trojan, starting from middle_N_benign, ' +
                    'e.g., if middle_N_benign = 1000, trojan_size = 5, choose the top 1000th ~ 1005th benign features.' +
                    'if middle_N_benign = None, then choose top/bottom features for backdoor attack.')

    # sub-arguments for the MLP classifier.
    p.add_argument('--mlp-retrain', type=int, choices=[0, 1],
                   help='Whether to retrain the MLP classifier.')
    p.add_argument('--mlp-hidden',
                   help='The hidden layers of the MLP classifier, example: "100-30", which in drebin_new_7 case would make the architecture as 1340-100-30-7')
    p.add_argument('--mlp-batch-size', default=32, type=int,
                   help='MLP classifier batch_size.')
    p.add_argument('--mlp-lr', default=0.001, type=float,
                   help='MLP classifier Adam learning rate.')
    p.add_argument('--mlp-epochs', default=50, type=int,
                   help='MLP classifier epochs.')
    p.add_argument('--mlp-dropout', default=0.2, type=float,
                   help='MLP classifier Dropout rate.')
    p.add_argument('--random-state', default=42, type=int,
                   help='MLP classifier random_state for train validation split.')
    p.add_argument('--mntd-half-training', default=0, type=int, choices=[0, 1],
                   help='whether to train the MLP model with randomly chosen 50% training set, for MNTD defense evaluation only.')
    p.add_argument('--subset-family',
                   help='protected family name. We will remove these samples during benign target model training for MNTD evaluation.')

    ''' for backdoor transfer attack'''
    p.add_argument('--poison-mal-benign-rate', type=float, default=0,
                   help='the ratio of malware VS. benign when adding poisoning samples')
    p.add_argument('--benign-poison-ratio', type=float, default=0.005,
                    help='The ratio of poison set for benign samples, malware poisoning would be multiplied by poison-mal-benign-rate')
    p.add_argument('--space', default='feature_space', help='whether it is feature_space or problem_space')

    p.add_argument('--limited-data', type=float, default=1.0, help='the ratio of training set the attacker has access to')
    p.add_argument('--mode', help='which debug mode should we read mask from')

    # # Harvesting options
    p.add_argument('--harvest', action='store_true')
    p.add_argument('--organ-depth', type=int, default=100)
    p.add_argument('--donor-depth', type=int, default=10)

    # Misc
    p.add_argument('-D', '--debug', action='store_true', help='Display log output in console if True.')
    p.add_argument('--rerun-past-failures', action='store_true', help='Rerun all past logged failures.')

    args = p.parse_args()

    logging.warning('Running with configuration:\n' + pformat(vars(args)))

    return args


def create_parent_folder(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


def get_model_dims(model_name, input_layer_num, hidden_layer_num, output_layer_num):
    """convert hidden layer arguments to the architecture of a model (list)
    Arguments:
        model_name {str} -- 'MLP' or 'Contrastive AE'.
        input_layer_num {int} -- The number of the features.
        hidden_layer_num {str} -- The '-' connected numbers indicating the number of neurons in hidden layers.
        output_layer_num {int} -- The number of the classes.
    Returns:
        [list] -- List represented model architecture.
    """
    try:
        if not hidden_layer_num:
            dims = [input_layer_num, output_layer_num]
        elif '-' not in hidden_layer_num:
            dims = [input_layer_num, int(hidden_layer_num), output_layer_num]
        else:
            hidden_layers = [int(dim) for dim in hidden_layer_num.split('-')]
            dims = [input_layer_num]
            for dim in hidden_layers:
                dims.append(dim)
            dims.append(output_layer_num)
        logging.debug(f'{model_name} dims: {dims}')
    except:
        logging.error(f'get_model_dims {model_name}\n{traceback.format_exc()}')
        sys.exit(-1)

    return dims



def dump_pickle(data, output_dir, filename, overwrite=True):
    dump_data('pickle', data, output_dir, filename, overwrite)


def dump_json(data, output_dir, filename, overwrite=True):
    dump_data('json', data, output_dir, filename, overwrite)


def dump_data(protocol, data, output_dir, filename, overwrite=True):
    file_mode = 'w' if protocol == 'json' else 'wb'
    fname = os.path.join(output_dir, filename)
    logging.info(f'Dumping data to {fname}...')
    if overwrite or not os.path.exists(fname):
        with open(fname, file_mode) as f:
            if protocol == 'json':
                json.dump(data, f, indent=4)
            else:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def calculate_base_metrics(clf, y_test, y_pred, y_scores, phase, output_dir=None):
    """Calculate ROC, F1, Precision and Recall for given scores.

    Args:
        y_test: Array of ground truth labels aligned with `y_pred` and `y_scores`.
        y_pred: Array of predicted labels, aligned with `y_scores` and `model.y_test`.
        y_scores: Array of predicted scores, aligned with `y_pred` and `model.y_test`.
        output_dir: The directory used for dumping output.

    Returns:
        dict: Model performance stats.

    """

    acc, f1, precision, recall, fpr = -1, -1, -1, -1, -1

    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    logging.debug(f'cm: {cm}')
    if np.all(y_test == 0) and np.all(y_pred == 0):
        TN = len(y_test)
        TP, FP, FN = 0, 0, 0
    elif np.all(y_test == 1) and np.all(y_pred == 1):
        TP = len(y_test)
        TN, FP, FN = 0, 0, 0
    else:
        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]

    try:
        f1 = sklearn.metrics.f1_score(y_test, y_pred)
        precision = sklearn.metrics.precision_score(y_test, y_pred)
        recall = sklearn.metrics.recall_score(y_test, y_pred)
        acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    except:
        logging.error(f'calculate_base_metrics: {traceback.format_exc()}')

    try:
        fpr = FP / (FP + TN)
    except:
        logging.error(f'calculate_base_metrics fpr: {traceback.format_exc()}')

    if output_dir:
        pred_file = os.path.join(output_dir, f'{clf}_prediction_{phase}.csv')
        with open(pred_file, 'w') as f:
            f.write(f'ground,pred,score\n')
            for i in range(len(y_test)):
                f.write(f'{y_test[i]},{y_pred[i]},{y_scores[i]}\n')

    return {
        'model_performance': {
            'acc': acc,
            # 'roc': roc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'cm': cm
        }
    }


def evalute_classifier_perf_on_training_and_testing(model, clf, output_dir, roc_curve_path=None):

    # tps = np.where((model.y_test & y_pred) == 1)[0]

    if clf in ['SVM', 'SecSVM', 'RbfSVM']:
        y_train_pred = model.clf.predict(model.X_train)
        y_pred = model.clf.predict(model.X_test)
        y_train_scores = model.clf.decision_function(model.X_train)
        y_scores = model.clf.decision_function(model.X_test)
    elif clf == 'mlp':
        K.clear_session()
        mlp_model = load_model(model.mlp_h5_model_path)
        y_train_pred = mlp_model.predict(model.X_train)
        y_pred = mlp_model.predict(model.X_test)

        y_train_scores = y_train_pred
        y_scores = y_pred
        y_train_pred = np.array([int(round(v[0])) for v in y_train_pred], dtype=np.int64)
        y_pred = np.array([int(round(v[0])) for v in y_pred], dtype=np.int64)
    else:
        y_train_scores = model.clf.predict_proba(model.X_train)[:, 1]
        y_scores = model.clf.predict_proba(model.X_test)[:, 1]

    if roc_curve_path:
        plot_roc_curve(model.y_test, y_scores, clf, roc_curve_path)

    mask1 = (model.y_test == 1)
    mask = mask1 & (y_pred == 1)
    tps = np.where(mask == True)[0]

    t3 = timer()
    report_train = calculate_base_metrics(clf, model.y_train, y_train_pred, y_train_scores, 'train', output_dir)
    report = calculate_base_metrics(clf, model.y_test, y_pred, y_scores, 'test', output_dir)
    t4 = timer()
    report['number_of_apps'] = {'train': len(model.y_train),
                                'test': len(model.y_test),
                                'tps': len(tps)}

    logging.info('Performance on training:\n' + pformat(report_train))
    logging.info('Performance on testing:\n' + pformat(report))
    return report


def plot_roc_curve(y_test, y_test_score, clf_name, roc_curve_path):
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.ticker').disabled = True

    FONT_SIZE = 24
    TICK_SIZE = 20
    fig = plt.figure(figsize=(8, 8))
    fpr_plot, tpr_plot, _ = roc_curve(y_test, y_test_score)
    plt.plot(fpr_plot, tpr_plot, lw=2, color='r')
    plt.gca().set_xscale("log")
    plt.yticks(np.arange(22) / 20.0)
    plt.xlim([1e-3, 0.1])
    plt.ylim([-0.04, 1.04])
    plt.tick_params(labelsize=TICK_SIZE)
    plt.gca().grid(True)
    plt.xlabel("False positive rate", fontsize=FONT_SIZE, fontname='Georgia')
    plt.ylabel("True positive rate", fontsize=FONT_SIZE, fontname='Georgia')
    create_parent_folder(roc_curve_path)
    fig.savefig(roc_curve_path, bbox_inches='tight')
    logging.info('ROC curve saved')


def resolve_confidence_level(confidence, benign_scores):
    """Resolves a given confidence level w.r.t. a set of benign scores.

    `confidence` corresponds to the percentage of benign scores that should be below
    the confidence margin. Practically, for a value N the attack will continue adding features
    until the adversarial example has a score which is 'more benign' than N% of the known
    benign examples.

    In the implementation, 100 - N is performed to calculate the percentile as the benign
    scores in the experimental models are negative.

    Args:
        confidence: The percentage of benign scores that should be below the confidence margin.
        benign_scores: The sample of benign scores to compute confidence with.

    Returns:
        The target score to resolved at the given confidence level.

    """
    if confidence == 'low':
        return 0
    elif confidence == 'high':
        confidence = 25
    try:
        # perc. inverted b/c benign scores are negative
        return np.abs(np.percentile(benign_scores, 100 - float(confidence)))
    except:
        logging.error(f'Unknown confidence level: {confidence}')


def decide_which_part_feature_to_perturb(middle_N, select_benign_features):
    if not middle_N:
        if select_benign_features == 'top':
            tmp = 'top'
        else:
            tmp = 'bottom'
    else:
        tmp = f'middle-{middle_N}'
    return tmp



def parse_multi_trigger_args():
    p = argparse.ArgumentParser()

    # Experiment variables
    p.add_argument('-d', '--dataset', help='Which dataset to use: drebin or apg or apg-10 (10% of apg)')
    p.add_argument('-c', '--classifier')
    p.add_argument('--lambda-1', type=float, default=1e-3, help='lambda in the loss function to balance two terms')
    p.add_argument('--num-triggers', type=int, default=5, help='Number of multiple triggers')
    p.add_argument('--benign-poison-ratio', type=float, default=0.05,
                    help='The ratio of poison set for benign samples, malware poisoning would be multiplied by poison-mal-benign-rate')

    p.add_argument('--clean-ratio', type=float, default=0.1, help='The ratio of clean set')

    p.add_argument('--use-last-weight', type=int, default=1, help='1: use last weight for models in alternate optimization')
    p.add_argument('--alter-retrain-full-training', type=int, default=0,
                    help='0: use batch training set and poison set to train a poisoned model;' + \
                         '1: use full training set and poison set; 2: use poison set only')

    p.add_argument('--max-iter', type=int, default=10, help='The maximum number of iterations')
    p.add_argument('--num-of-train-batches', type=int, default=5,
                   help='Split the training set to # of batches and do the batch-update for mask optimization')
    p.add_argument('--mask-optim-step', type=int, default=30, help='how many steps we should run when solving a mask')
    p.add_argument('--attack-succ-threshold', type=float, default=0.95, help='The attack success threshold for optimization')

    p.add_argument('--poison-mal-benign-rate', type=float, default=0.05,
                   help='the ratio of malware VS. benign when adding poisoning samples')
    p.add_argument('--subset-benign-rate', type=float, default=5,
                    help='we might need to upsample the subset during mask optimization' +
                        '(originally benign vs. malware = 200:200, which is total X_train * clean_ratio / 2)' +
                        'if subset-benign-rate = 5, we would upsample subset family to 1000 during mask optimization')
    p.add_argument('--remain-benign-rate', type=float, default=3,
                    help='similar as subset-benign-rate. It shows remained_malware vs. benign when optimizing the mask')

    p.add_argument('--subset-family', default="autoins",
                    help='the name of subset malware family')
    p.add_argument('--delta-size', type=int, default=30,
                    help='delta size for mask expansion idea')
    p.add_argument('--mask-size-upperbound', type=int, default=0,
                    help='mask size upperbound limit on the optimized mask, 0 if no upperbound')
    p.add_argument('--mntd-half-training', type=int, default=0,
                    help='whether to only use half training set to train MNTD target backdoored models.')

    p.add_argument('--device', default='5', help='which GPU device to use')

    p.add_argument('--mlp-hidden',
                   help='The hidden layers of the MLP classifier, example: "1024", which would result 10000-1024-1')
    p.add_argument('--mlp-batch-size', default=32, type=int,
                   help='MLP classifier batch_size.')
    p.add_argument('--mlp-lr', default=0.001, type=float,
                   help='MLP classifier Adam learning rate.')
    p.add_argument('--mlp-epochs', default=50, type=int,
                   help='MLP classifier epochs.')
    p.add_argument('--mlp-dropout', default=0.2, type=float,
                   help='MLP classifier Dropout rate.')

    p.add_argument('--random-state', default=0, type=int,
                   help='random state for the clean MLP model training and validation split.')

    p.add_argument('--mask-expand-type', default=0, type=int,
                   help='choose type 1 or type 2 for mask expansion with a delta, 0 for no expansion')
    p.add_argument('--convert-mask-to-binary', type=int, default=0,
                    help='whether to convert the solved mask from real value to binary (0 and 1) during mask optimization')
    # Misc
    p.add_argument('-D', '--debug', action='store_true', help='Display log output in console if True.')

    p.add_argument('--setting', default='', help='name of DEBUG_MODE, for debugging different settings')

    p.add_argument('--limited-data', type=float, default=1.0, help='the ratio of training set the attacker has access to')
    p.add_argument('--param-v', type=float, default=1.0, help='value of the hyper-parameter v')
    p.add_argument('--realizable-only', type=int, default=0,
                    help='0 means feature space attack while 1 means problem-space attack')

    args = p.parse_args()

    logging.warning('Running with configuration:\n' + pformat(vars(args)))
    return args

def plot_subset_final_result(subset_report_simple_path, iters, num_batch, save_fig_folder):
    # plt.grid(axis='both', color='0.8')

    df = pd.read_csv(subset_report_simple_path, header=0)
    _, poisoned_mask_size = extract_clean_and_poisoned_model_perf(df, col='mask_size')
    clean_main_f1, poisoned_main_f1 = extract_clean_and_poisoned_model_perf(df, col='main_f1')
    clean_subset_recall, poisoned_subset_recall = extract_clean_and_poisoned_model_perf(df, col='subset_recall')

    clean_remain_recall, poisoned_remain_recall = extract_clean_and_poisoned_model_perf(df, col='remain_recall')
    clean_benign_fpr, poisoned_benign_fpr = extract_clean_and_poisoned_model_perf(df, col='benign_fpr')

    best_optim_acc = None
    try:
        _, best_optim_acc = extract_clean_and_poisoned_model_perf(df, col='best_attack_acc')
    except:
        print(f'best_attack_acc not available')

    keep_subset_in_train_flag = True
    try:
        clean_subset_train_recall, poisoned_subset_train_recall = extract_clean_and_poisoned_model_perf(df, col='subset_train_recall')
        clean_subset_test_recall, poisoned_subset_test_recall = extract_clean_and_poisoned_model_perf(df, col='subset_test_recall')
    except:
        keep_subset_in_train_flag = False
        print('subset_train_recall or subset_test_recall not exist')

    # batches = range(iters * num_batch) # 10 * 5 = 50
    batches = range(len(poisoned_subset_recall))

    ''' mask size '''

    mask_fig, mask_ax = init_fig_and_ax('Alternate Optimization Batches', 'Mask Size')
    mask_ax.plot(batches, poisoned_mask_size, 'o-', label='Mask Size', linewidth=2.5, mew=2, ms=10)

    zoom_start_batch = 20
    mask_zoom_fig, mask_zoom_ax = init_fig_and_ax('Alternate Optimization Batches', 'Mask Size',
                                                  xticks=range(zoom_start_batch, len(batches)+1, 5))
    mask_zoom_ax.plot(batches[zoom_start_batch:], poisoned_mask_size[zoom_start_batch:], 'o-',
                      label='Mask Size Zoom', linewidth=2.5, mew=2, ms=10)

    ''' main F1 and benign FPR '''
    f1_fpr_fig, f1_fpr_ax = init_fig_and_ax('Alternate Optimization Batches', 'Percentage')
    plot_subset_final_result_helper(f1_fpr_ax, clean_main_f1, batches, poisoned_main_f1,
                                    's', '^-', 'Original F1', 'Main Task F1')
    plot_subset_final_result_helper(f1_fpr_ax, clean_benign_fpr, batches, poisoned_benign_fpr,
                                    'v', 'o-', 'Original Benign FPR', 'Poisoned Benign FPR')

    ''' subset recall and remain recall '''
    recall_fig, recall_ax = init_fig_and_ax('Alternate Optimization Batches', 'Percentage')
    plot_subset_final_result_helper(recall_ax, clean_remain_recall, batches, poisoned_remain_recall,
                                    'v', 'o-', 'Original Remain Recall', 'Poisoned Remain Recall')
    plot_subset_final_result_helper(recall_ax, clean_subset_recall, batches, poisoned_subset_recall,
                                    's', '^-', 'Original Subset Recall', 'Poisoned Subset Recall')

    if keep_subset_in_train_flag:
        plot_subset_final_result_helper(recall_ax, clean_subset_train_recall, batches, poisoned_subset_train_recall,
                                    's', '*-', 'Original Subset Recall (from training)', 'Poisoned Subset Recall (from training)')
        plot_subset_final_result_helper(recall_ax, clean_subset_test_recall, batches, poisoned_subset_test_recall,
                                    's', 'D-', 'Original Subset Recall (from testing)', 'Poisoned Subset Recall (from testing)')

    os.makedirs(save_fig_folder, exist_ok=True)
    save_fig_helper(mask_fig, mask_ax, os.path.join(save_fig_folder, f'mask_size_iter_{iters}_batch_{num_batch}.png'))
    save_fig_helper(mask_zoom_fig, mask_zoom_ax, os.path.join(save_fig_folder,
                                                    f'mask_size_iter_{iters}_batch_{num_batch}_zoom_from_batch_{zoom_start_batch}.png'))
    save_fig_helper(f1_fpr_fig, f1_fpr_ax, os.path.join(save_fig_folder, f'maintask_F1_benign_FPR_iter_{iters}_batch_{num_batch}.png'))
    save_fig_helper(recall_fig, recall_ax, os.path.join(save_fig_folder, f'subset_and_remain_recall_iter_{iters}_batch_{num_batch}.png'))

    if best_optim_acc is not None:
        zoom_batch = 0
        for idx, acc in enumerate(best_optim_acc):
            if acc >= 0.8:
                zoom_batch = idx
                break
        acc_zoom_fig, acc_zoom_ax = init_fig_and_ax('Alternate Optimization Batches',
                                                    'Best Optimization Acc',
                                                    xticks=range(zoom_batch, len(best_optim_acc)+zoom_batch, 10))
        acc_zoom_ax.plot(range(len(best_optim_acc)-zoom_batch), best_optim_acc[zoom_batch:], 'o-',
                        label='Optim Acc', linewidth=2.5, mew=2, ms=10)

        save_fig_helper(acc_zoom_fig, acc_zoom_ax, os.path.join(save_fig_folder, f'optim_acc_iter_{iters}_batch_{num_batch}.png'))

    plt.clf()


def plot_subset_final_result_helper(ax, clean_model_perf, x_list, poisoned_model_perf_list,
                                    clean_shape, poisoned_shape, clean_label, poisoned_label):
    ax.plot([-1], clean_model_perf, clean_shape, label=clean_label, mew=2, ms=10)
    # mew: marker edge width, ms: marker size
    ax.plot(x_list, poisoned_model_perf_list, poisoned_shape, label=poisoned_label, linewidth=2.5, mew=2, ms=10)


def extract_clean_and_poisoned_model_perf(df, col):
    perf = df[col].to_numpy()
    clean_perf = perf[0]
    poisoned_perf = perf[1:]
    return clean_perf, poisoned_perf


def init_fig_and_ax(xlabel, ylabel, xticks=None):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks:
        ax.set_xticks(xticks)
    return fig, ax


def save_fig_helper(fig, ax, filename):
    ax.legend(loc='best')
    fig.tight_layout()
    ax.grid(axis='both', color='0.8')
    fig.savefig(filename, dpi=100)
