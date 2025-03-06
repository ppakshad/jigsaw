import numpy as np
from sklearn.metrics.classification import accuracy_score
import torch
from mntd_model_lib.utils_basic import load_dataset_setting, train_model, eval_model
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

if __name__ == '__main__':
    log_path = './logs/mntd/main'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    init_log(log_path, level=logging.DEBUG)

    t1 = timer()
    args = parser.parse_args()

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
    shadow_indices = np.random.choice(tot_num, int(tot_num*SHADOW_PROP))
    target_indices = np.random.choice(tot_num, int(tot_num*TARGET_PROP))
    print ("Data indices owned by the defender:",shadow_indices)
    print ("Data indices owned by the attacker:",target_indices)
    shadow_set = torch.utils.data.Subset(trainset, shadow_indices)
    shadow_loader = torch.utils.data.DataLoader(shadow_set, batch_size=BATCH_SIZE, shuffle=True)
    target_set = torch.utils.data.Subset(trainset, target_indices)
    target_loader = torch.utils.data.DataLoader(target_set, batch_size=BATCH_SIZE*2, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE*2)

    SAVE_PREFIX = f'./models/mntd/shadow_model_ckpt/{args.task}_{args.clf}'
    if not os.path.isdir(SAVE_PREFIX):
        os.makedirs(SAVE_PREFIX, exist_ok=True)
    if not os.path.isdir(SAVE_PREFIX+'/models'):
        os.makedirs(SAVE_PREFIX+'/models', exist_ok=True)

    all_shadow_acc = []
    all_target_acc = []

    print(f'start training shadow models...')

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
        save_path = SAVE_PREFIX+'/models/shadow_benign_%d.model'%i
        if os.path.exists(save_path):
            logging.info(f'{i} model already exists, skip')
        else:
            if args.clf != 'SecSVM':
                model = Model(gpu=GPU)
                train_model(model, shadow_loader, epoch_num=N_EPOCH, is_binary=is_binary, clf=args.clf, verbose=False)
                torch.save(model.state_dict(), save_path)
                acc, precision, recall, auc = eval_model(model, testloader, is_binary=is_binary, clf=args.clf)
            else:
                secsvm.fit(X_shadow, y_shadow)
                torch.save(secsvm.model.state_dict(), save_path)
                y_pred = secsvm.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

            logging.info (f"shadow model {i}, Acc {acc:.4f}, saved to {save_path} @ {datetime.now()}")
            all_shadow_acc.append(acc)

    if args.clf == 'SVM':
        logging.info(f'DEBUG: TARGET epoch_num for SVM: {int(N_EPOCH*SHADOW_PROP/TARGET_PROP)}')

    for i in range(TARGET_NUM):
        if args.clf != 'SecSVM':
            model = Model(gpu=GPU)
            if args.clf == 'SVM':
                train_model(model, target_loader, epoch_num=int(N_EPOCH*SHADOW_PROP/TARGET_PROP), is_binary=is_binary, clf=args.clf, verbose=False)
            elif args.clf == 'MLP':
                train_model(model, target_loader, epoch_num=15, is_binary=is_binary, clf=args.clf, verbose=False)
            save_path = SAVE_PREFIX+'/models/target_benign_%d.model'%i
            torch.save(model.state_dict(), save_path)
            acc, precision, recall, auc = eval_model(model, testloader, is_binary=is_binary, clf=args.clf)
        else:
            secsvm.fit(X_target, y_target)
            save_path = SAVE_PREFIX+'/models/target_benign_%d.model'%i
            torch.save(secsvm.model.state_dict(), save_path)
            y_pred = secsvm.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

        print ("target model %d, Acc %.4f, saved to %s @ %s"%(i, acc, save_path, datetime.now()))
        all_target_acc.append(acc)

    log = {'shadow_num':SHADOW_NUM,
           'target_num':TARGET_NUM,
           'shadow_acc':np.nanmean(all_shadow_acc),
           'target_acc':np.nanmean(all_target_acc)}
    log_path = SAVE_PREFIX + f'/benign-{datetime.now()}.log'
    with open(log_path, "w") as outf:
        json.dump(log, outf)
    print ("Log file saved to %s"%log_path)
    t2 = timer()
    print(f'tik tok: {t2 - t1:.1f} seconds')
