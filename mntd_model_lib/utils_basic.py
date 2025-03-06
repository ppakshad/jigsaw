import sys
import traceback
import logging
import numpy as np
import torch
import torch.utils.data
from collections import Counter
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
# import torchvision
# import torchvision.transforms as transforms

# sys.path.append('mntd_model_lib')


def load_dataset_setting(task, clf=None):
    if task == 'mnist':
        BATCH_SIZE = 100
        N_EPOCH = 100
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        # trainset = torchvision.datasets.MNIST(root='./raw_data/', train=True, download=True, transform=transform)
        # testset = torchvision.datasets.MNIST(root='./raw_data/', train=False, download=False, transform=transform)
        trainset = None
        testset = None
        is_binary = False
        need_pad = False
        from mntd_model_lib.mnist_cnn_model import Model, troj_gen_func, random_troj_setting
    elif task == 'cifar10':
        BATCH_SIZE = 100
        N_EPOCH = 100
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        # trainset = torchvision.datasets.CIFAR10(root='./raw_data/', train=True, download=True, transform=transform)
        # testset = torchvision.datasets.CIFAR10(root='./raw_data/', train=False, download=False, transform=transform)
        trainset = None
        testset = None
        is_binary = False
        need_pad = False
        from mntd_model_lib.cifar10_cnn_model import Model, troj_gen_func, random_troj_setting
    elif task == 'audio':
        BATCH_SIZE = 100
        N_EPOCH = 100
        from mntd_model_lib.audio_dataset import SpeechCommand
        trainset = SpeechCommand(split=0)
        testset = SpeechCommand(split=2)
        is_binary = False
        need_pad = False
        from mntd_model_lib.audio_rnn_model import Model, troj_gen_func, random_troj_setting
    elif task == 'rtNLP':
        BATCH_SIZE = 64
        N_EPOCH = 50
        from mntd_model_lib.rtNLP_dataset import RTNLP
        trainset = RTNLP(train=True)
        testset = RTNLP(train=False)
        is_binary = True
        need_pad = True
        from mntd_model_lib.rtNLP_cnn_model import Model, troj_gen_func, random_troj_setting
    elif task == 'apg':
        from mntd_model_lib.apg_dataset import APG
        trainset = APG(train=True, clf=clf)
        testset = APG(train=False, clf=clf)
        is_binary = True
        need_pad = False
        if clf == 'SVM':
            BATCH_SIZE = 100
            N_EPOCH = 10 # originally 100, seems not reasonable
            # N_EPOCH = 5 # todo: change back to 5
            from mntd_model_lib.apg_svm_model import Model, troj_gen_func, random_troj_setting
        elif clf == 'MLP':
            BATCH_SIZE = 64
            # N_EPOCH = 10 # for 2% training set, 10000-1024-1, maybe 5 is better, but 10 should not create a trouble
            N_EPOCH = 5 # todo: change here for 2048+256 shadow models
            from mntd_model_lib.apg_mlp_model import Model, troj_gen_func, random_troj_setting
        elif clf == 'SecSVM':
            BATCH_SIZE = 100
            N_EPOCH = 40
            Model = None # use the original SecSVM implementation instead of inventing a Model
            from mntd_model_lib.apg_secsvm_model import troj_gen_func, random_troj_setting
        else:
            raise ValueError(f'clf {clf} not implemented')
    else:
        raise NotImplementedError("Unknown task %s"%task)

    return BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, need_pad, Model, troj_gen_func, random_troj_setting


class BackdoorDataset(torch.utils.data.Dataset):
    def __init__(self, src_dataset, atk_setting, troj_gen_func, choice=None, mal_only=False,
                 need_pad=False, poison_benign_only=False):
        self.src_dataset = src_dataset
        self.atk_setting = atk_setting
        self.troj_gen_func = troj_gen_func
        self.need_pad = need_pad

        self.mal_only = mal_only
        if choice is None:
            choice = np.arange(len(src_dataset))
        self.choice = choice
        # todo: debugging
        # train_choice, val_choice = train_test_split(choice, test_size=0.33, random_state=42)
        # self.choice = train_choice


        inject_p = atk_setting[5] # poisoning ratio, random sampled from U(0.05, 0.5)
        if poison_benign_only:
            y_train = src_dataset.ys
            logging.debug(f'choice: len: {choice.shape[0]}, first 10: {choice[:10]}')
            y_attacker = y_train[choice]  # attacker owned set
            logging.debug(f'y_attacker: {Counter(y_attacker)}')
            benign_choice = np.where(y_attacker == 0)[0]
            logging.debug(f'benign_choice: len: {benign_choice.shape[0]}, first 10: {benign_choice[:10]}')
            self.mal_choice = np.random.choice(benign_choice, int(len(choice) * inject_p), replace=False)
            logging.debug(f' inject_p: {inject_p}; mal_choice: len: {self.mal_choice.shape[0]}, first 10: {self.mal_choice[:10]}')
        else:
            ''' poison some samples from the choice set, could be benign or malicious '''
            logging.debug(f'poison_benign_only is False, choice len: {choice.shape[0]}')
            logging.debug(f'len(choice)*inject_p: {int(len(choice)*inject_p)}')
            self.mal_choice = np.random.choice(choice, int(len(choice)*inject_p), replace=False)

    def __len__(self,):
        if self.mal_only:
            ''' generate trojaned set only'''
            return len(self.mal_choice)
        else:
            ''' generate training + trojaned (poisoning) set '''
            return len(self.choice) + len(self.mal_choice)

    def __getitem__(self, idx):
        if (not self.mal_only and idx < len(self.choice)):
            ''' Return non-trojaned data '''
            if self.need_pad:
                # In NLP task we need to pad input with length of Troj pattern
                p_size = self.atk_setting[0]
                X, y = self.src_dataset[self.choice[idx]]
                X_padded = torch.cat([X, torch.LongTensor([0]*p_size)], dim=0)
                return X_padded, y
            else:
                return self.src_dataset[self.choice[idx]]

        if self.mal_only: # trojaned only
            X, y = self.src_dataset[self.mal_choice[idx]]
        else:
            # when idx > 2% shadow set or 50% target set, poison it in the next step
            X, y = self.src_dataset[self.mal_choice[idx-len(self.choice)]]
        X_new, y_new = self.troj_gen_func(X, y, self.atk_setting)

        # TODO: warning: this is an ugly solution
        if self.mal_only: # trojaned testing set, we should not change their labels for evaluting backdoor effectiveness
            y_new = y
        return X_new, y_new


def train_model(model, dataloader, epoch_num, is_binary, clf='MLP', verbose=True):
    model.train() # Sets the module in training mode.
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # todo: use larger lr to see if we can improve recall
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(epoch_num):
        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0
        for i,(x_in, y_in) in enumerate(dataloader):
            B = x_in.size()[0]
            pred = model(x_in)
            # if epoch == 0 and i == 0:
            #     print(f'DEBUG: pred: {type(pred.cpu()[0])}, {list(pred.cpu())}')
            #     print(f'DEBUG: y_in: {type(y_in.cpu()[0])}, {list(y_in.cpu())}')
            #     print(f'pred>0: {(pred>0).cpu()}')
            #     print(f'(pred>0).cpu().t().long().eq(y_in): {(pred>0).cpu().t().long().eq(y_in)}')
            #     print(f'((pred>0).cpu().t().long().eq(y_in)).sum().item(): {((pred>0).cpu().t().long().eq(y_in)).sum().item()}')

            loss = model.loss(pred, y_in)
            optimizer.zero_grad()
            loss.backward() # calculate gradient w.r.t. to the loss
            optimizer.step() # update gradient using gradient descent
            cum_loss += loss.item() * B # loss.item() contains the loss of entire mini-batch, but divided by the batch size
            if is_binary:
                if clf == 'SVM':
                    # y_pred = [0 if m < 0 else 1 for m in pred.cpu().data.numpy().flatten()]
                    # cum_acc += accuracy_score(y_pred, y_in.cpu().data()) * B
                    cum_acc += ((pred>0).cpu().t().long().eq(y_in)).sum().item() # No. of accurate predictions
                elif clf == 'MLP':
                    cum_acc += ((pred>=0.5).cpu().t().long().eq(y_in)).sum().item()
                else:
                    raise ValueError(f'clf {clf} not supported')
            else:
                pred_c = pred.max(1)[1].cpu() # TODO: why [1]? not [0]?
                cum_acc += (pred_c.eq(y_in)).sum().item()
            tot = tot + B
            # if epoch == 0 and i < 10:
            #     print(f'B: {B}, cum_acc: {cum_acc}, tot: {tot}')
        if verbose:
            print ("Epoch %d, loss = %.4f, acc = %.4f"%(epoch, cum_loss/tot, cum_acc/tot))
    return


def eval_model(model, dataloader, is_binary, clf='MLP'):
    model.eval() # Sets the module in evaluation mode.
    cum_acc = 0.0
    tot = 0.0
    y_true = [v for (x_in, y_in) in dataloader for v in y_in.tolist()]
    # print(f'y_true type: {type(y_true[0])}, y_true[:20]: {y_true[:20]}')

    y_pred = []
    for i,(x_in, y_in) in enumerate(dataloader):
        B = x_in.size()[0]
        pred = model(x_in)

        # if i == 0:
        #     print(f'y_in[:20]: {y_in[:20]}')
        #     print(f'pred shape: {pred.shape}, pred[:20]: {pred[:20]}')

        if is_binary:
            if clf in ['SVM', 'SecSVM']:
                # cum_acc += ((pred>0).cpu().long().eq(y_in)).sum().item()
                cum_acc += ((pred>0).cpu().t().long().eq(y_in)).sum().item()
                for v in pred.reshape(-1,).tolist():
                    y_pred.append(v > 0)
            elif clf == 'MLP':
                cum_acc += ((pred>0.5).cpu().t().long().eq(y_in)).sum().item()
                for v in pred.reshape(-1,).tolist():
                    y_pred.append(v > 0.5)
            else:
                raise ValueError(f'clf {clf} not supported')
        else: # not used in our setting
            pred_c = pred.max(1)[1].cpu() # max() returns two values, max_value and max_index, so [1] returns the predicted label
            cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    # print(f'y_pred type: {type(y_pred[0])}, y_pred[:20]: {y_pred[:20]}')
    precision, recall, auc = -1, -1, -1
    try:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        print(f'cm: {cm}')
        # auc = roc_auc_score(y_true, y_pred) # auc is often ill defined for 1 class so omitted
    except:
        print('eval_model: ', traceback.format_exc())
    # return cum_acc / tot
    return cum_acc / tot, precision, recall, auc
