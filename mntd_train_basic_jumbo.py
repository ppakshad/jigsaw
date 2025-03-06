import numpy as np
import torch
from mntd_model_lib.utils_basic import load_dataset_setting, train_model, eval_model, BackdoorDataset
from lib.secsvm import SecSVM
import os
from datetime import datetime
import json
import argparse
from timeit import default_timer as timer
import logging
from backdoor.logger import init_log


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP/apg).')
parser.add_argument('--clf', type=str, default='SVM', help='Specify classifier: SVM or MLP or SecSVM.')
parser.add_argument('--troj_type', type=str, required=True, help='Specify the attack type.')
parser.add_argument('--min_size', type=int, required=True, help='The minimum size of the trigger.')
parser.add_argument('--max_size', type=int, required=True, help='The maximum size of the trigger.')


if __name__ == '__main__':
    log_path = './logs/mntd/main'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    init_log(log_path, level=logging.DEBUG)
    t1 = timer()

    args = parser.parse_args()

    min_size = args.min_size
    max_size = args.max_size

    GPU = True
    SHADOW_PROP = 0.02
    TARGET_PROP = 0.5 # didn't use
    SHADOW_NUM = 2048+256
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
    target_indices = np.random.choice(tot_num, int(tot_num*TARGET_PROP)) # useless
    print ("Data indices owned by the defender:",shadow_indices)

    if args.troj_type == 'M': # for subset random 10000 jumbo
        SAVE_PREFIX = f'./models/mntd/shadow_model_ckpt/{args.task}_{args.clf}'
    elif args.troj_type == 'Top-benign-jumbo':
        SAVE_PREFIX = f'./models/mntd/shadow_model_ckpt/{args.task}_{args.clf}_top_benign'
    elif args.troj_type == 'Subset-2171-jumbo':
        SAVE_PREFIX = f'./models/mntd/shadow_model_ckpt/{args.task}_{args.clf}_subset_2171_realizable'
    else:
        raise ValueError(f'troj_type {args.troj_type} not supported')
    os.makedirs(SAVE_PREFIX, exist_ok=True)

    model_save_folder = SAVE_PREFIX+f'/models/random_size_{min_size}_{max_size}'
    os.makedirs(model_save_folder, exist_ok=True)

    all_shadow_recall = []
    all_shadow_recall_mal = []
    all_shadow_precision = []
    all_shadow_precision_mal = []

    if args.clf == 'SecSVM':
        secsvm = SecSVM(lr=0.001,
                        batchsize=BATCH_SIZE,
                        n_epochs=N_EPOCH,
                        K=0.2)
        X_shadow = trainset.Xs[shadow_indices]
        y_shadow = trainset.ys[shadow_indices]
        X_target = trainset.Xs[target_indices]
        y_target = trainset.ys[target_indices]
        X_test = testset.Xs
        y_test = testset.ys

    for i in range(SHADOW_NUM):
        atk_setting = random_troj_setting(args.troj_type, size=max_size, min_size=min_size)
        # only poison 5%~50% out of the 2% training set
        trainset_mal = BackdoorDataset(trainset, atk_setting, troj_gen_func, choice=shadow_indices, need_pad=need_pad)
        trainloader = torch.utils.data.DataLoader(trainset_mal, batch_size=BATCH_SIZE, shuffle=True)
        # poison 5% ~ 50% out of all the testing set as the backdoor task
        testset_mal = BackdoorDataset(testset, atk_setting, troj_gen_func, mal_only=True)
        # benign means keep the original testing set
        testloader_benign = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
        testloader_mal = torch.utils.data.DataLoader(testset_mal, batch_size=BATCH_SIZE)

        save_path = model_save_folder + '/shadow_jumbo_%d.model'%i
        model = Model(gpu=GPU)
        if os.path.exists(save_path):
            model.eval()
            model.load_state_dict(torch.load(save_path))
            logging.info(f'{i} model already exists, loaded')
        else:
            if args.clf != 'SecSVM':
                train_model(model, trainloader, epoch_num=N_EPOCH, is_binary=is_binary, clf=args.clf, verbose=False)
                torch.save(model.state_dict(), save_path)
            else: # USELESS for now
                secsvm.fit(X_shadow, y_shadow)
                torch.save(secsvm.model.state_dict(), save_path)
                acc, precision, recall, auc  = eval_model(secsvm.model.cpu(), testloader_benign, is_binary=is_binary, clf=args.clf)
                acc_mal, precision_mal, \
                    recall_mal, auc_mal = eval_model(secsvm.model.cpu(), testloader_mal, is_binary=is_binary, clf=args.clf)

            p_size, pattern, loc, alpha, target_y, inject_p = atk_setting
            print ("\tp size: %d; loc: %s; alpha: %.3f; target_y: %d; inject p: %.3f"%(p_size, loc, alpha, target_y, inject_p))

        acc, precision, recall, auc = eval_model(model, testloader_benign, is_binary=is_binary, clf=args.clf)
        # eval the backdoor task acc (but only add triggers to 5% - 50% of the test set)
        acc_mal, precision_mal, \
            recall_mal, auc_mal = eval_model(model, testloader_mal, is_binary=is_binary, clf=args.clf)
        print ("Model: %d, Recall %.4f, Recall on backdoor %.4f, Precision %.4f, Precision on backdoor %.4f, saved to %s @ %s" % \
                (i, recall, recall_mal, precision, precision_mal, save_path, datetime.now()))

        all_shadow_recall.append(recall)
        all_shadow_recall_mal.append(recall_mal)
        all_shadow_precision.append(precision)
        all_shadow_precision_mal.append(precision_mal)

    log = {'shadow_num':SHADOW_NUM,
           'shadow_recall':np.mean(all_shadow_recall),
           'shadow_recall_mal':np.mean(all_shadow_recall_mal),
           'shadow_precision':np.mean(all_shadow_precision),
           'shadow_precision_mal':np.mean(all_shadow_precision_mal)}
    logging.critical(f'{log}')
    log_path = SAVE_PREFIX+f'/jumbo_random_size_{min_size}_{max_size}_{datetime.now()}.log'
    with open(log_path, "w") as outf:
        json.dump(log, outf)
    print ("Log file saved to %s"%log_path)

    t2 = timer()
    print(f'tik tok: {t2 - t1:.1f} seconds')
