import numpy as np
import torch
from mntd_model_lib.utils_basic import load_dataset_setting, train_model, eval_model, BackdoorDataset
import os
import sys
from datetime import datetime
import json
import argparse
from timeit import default_timer as timer
import logging
from backdoor.logger import init_log


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP).')
parser.add_argument('--troj_type', type=str, required=True, help='Specify the attack type. M: modification attack; B: blending attack.')
parser.add_argument('--clf', type=str, default='SVM', help='Specify classifier: SVM or MLP or SecSVM.')
parser.add_argument('--min_size', type=int, required=True, help='The minimum size of the trigger.')
parser.add_argument('--max_size', type=int, required=True, help='The maximum size of the trigger.')
parser.add_argument('--max_poison_ratio', type=float, default=0.5, help='The maximum poisoning ratio.')


if __name__ == '__main__':
    log_path = './logs/mntd/main'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    init_log(log_path, level=logging.DEBUG)

    t1 = timer()
    args = parser.parse_args()

    min_size = args.min_size
    max_size = args.max_size
    max_poison_ratio = args.max_poison_ratio

    GPU = True
    SHADOW_PROP = 0.02
    TARGET_PROP = 0.5
    TARGET_NUM = 256
    np.random.seed(0)
    torch.manual_seed(0)
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, \
        need_pad, Model, troj_gen_func, random_troj_setting = load_dataset_setting(args.task, args.clf)
    tot_num = len(trainset)
    shadow_indices = np.random.choice(tot_num, int(tot_num*SHADOW_PROP))
    target_indices = np.random.choice(tot_num, int(tot_num*TARGET_PROP))
    print ("Data indices owned by the attacker:",target_indices)

    SAVE_PREFIX = './models/mntd/target_model_ckpt/%s'%args.task + f'_min_{min_size}_max_{max_size}'
    os.makedirs(SAVE_PREFIX+'/models', exist_ok=True)

    all_target_recall = []
    all_target_recall_mal = []
    all_target_precision = []
    all_target_precision_mal = []
    all_target_auc = []
    all_target_auc_mal = []


    for i in range(TARGET_NUM):
        model = Model(gpu=GPU)
        atk_setting = random_troj_setting(args.troj_type, size=max_size, min_size=min_size,
                                          max_poison_ratio=max_poison_ratio)
        trainset_mal = BackdoorDataset(trainset, atk_setting, troj_gen_func, choice=target_indices, need_pad=need_pad)
        trainloader = torch.utils.data.DataLoader(trainset_mal, batch_size=BATCH_SIZE, shuffle=True)

        atk_setting_tmp = list(atk_setting)
        atk_setting_tmp[5] = 1.0
        ''' poison all testing samples'''
        testset_mal = BackdoorDataset(testset, atk_setting_tmp, troj_gen_func, mal_only=True)
        testloader_benign = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
        testloader_mal = torch.utils.data.DataLoader(testset_mal, batch_size=BATCH_SIZE)

        save_path = SAVE_PREFIX+'/models/target_troj%s_%d.model'%(args.troj_type, i)
        if os.path.exists(save_path):
            print(f'load existing model {save_path}...')
            model.eval()
            model.load_state_dict(torch.load(save_path))
        else:
            train_model(model, trainloader, epoch_num=15, is_binary=is_binary, clf=args.clf, verbose=True)
            torch.save(model.state_dict(), save_path)
        acc, precision, recall, auc = eval_model(model, testloader_benign, is_binary=is_binary, clf=args.clf)
        acc_mal, precision_mal, \
            recall_mal, auc_mal = eval_model(model, testloader_mal, is_binary=is_binary, clf=args.clf)

        print ("Model: %d, Recall %.4f, Recall on backdoor %.4f, Precision %.4f, Precision on backdoor %.4f, saved to %s @ %s" % \
                    (i, recall, recall_mal, precision, precision_mal, save_path, datetime.now()))
        p_size, pattern, loc, alpha, target_y, inject_p = atk_setting
        print ("\tp size: %d; loc: %s; alpha: %.3f; target_y: %d; inject p: %.3f"%(p_size, loc, alpha, target_y, inject_p))
        all_target_recall.append(recall)
        all_target_recall_mal.append(recall_mal)
        all_target_precision.append(precision)
        all_target_precision_mal.append(precision_mal)
        all_target_auc.append(auc)
        all_target_auc_mal.append(auc_mal)

    log = {'target_num':TARGET_NUM,
           'target_recall':np.mean(all_target_recall),
           'target_recall_mal':np.mean(all_target_recall_mal),
           'target_precision':np.mean(all_target_precision),
           'target_precision_mal':np.mean(all_target_precision_mal),
           'target_auc':np.mean(all_target_auc),
           'target_auc_mal':np.mean(all_target_auc_mal)}
    log_path = SAVE_PREFIX + f'/troj{args.troj_type}_{datetime.now()}.log'
    with open(log_path, "w") as outf:
        json.dump(log, outf)
    print ("Log file saved to %s"%log_path)

    t2 = timer()
    print(f'tik tok: {t2 - t1:.1f} seconds')
