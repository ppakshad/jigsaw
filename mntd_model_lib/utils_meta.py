import os
import sys
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.metrics import roc_auc_score, roc_curve
from keras.models import load_model
from keras import backend as K

sys.path.append('backdoor/')
import models


def load_model_setting(task, clf=None):
    if task == 'mnist':
        from mntd_model_lib.mnist_cnn_model import Model
        input_size = (1, 28, 28)
        class_num = 10
        normed_mean = np.array((0.1307,)) # used to sample query set from a Gaussian distribution
        normed_std = np.array((0.3081,))
        is_discrete = False
    elif task == 'cifar10':
        from mntd_model_lib.cifar10_cnn_model import Model
        input_size = (3, 32, 32)
        class_num = 10
        normed_mean = np.reshape(np.array((0.4914, 0.4822, 0.4465)),(3,1,1))
        normed_std = np.reshape(np.array((0.247, 0.243, 0.261)),(3,1,1))
        is_discrete = False
    elif task == 'audio':
        from mntd_model_lib.audio_rnn_model import Model
        input_size = (16000,)
        class_num = 10
        normed_mean = normed_std = None
        is_discrete = False
    elif task == 'rtNLP':
        from mntd_model_lib.rtNLP_cnn_model import Model
        input_size = (1, 10, 300)
        class_num = 1  #Two-class, but only one output
        normed_mean = normed_std = None
        is_discrete = True  # NOTE: feature value could be discrete values like 0, 4725, 61, 186,
    elif task == 'apg':
        if clf == 'SVM':
            from mntd_model_lib.apg_svm_model import Model
        elif clf == 'MLP':
            from mntd_model_lib.apg_mlp_model import Model
        elif clf == 'SecSVM':
            Model = None # use the original SecSVM implementation instead of inventing a Model
        input_size = (10000,)
        class_num = 1
        normed_mean = normed_std = None
        is_discrete = False # NOTE: although our data is discrete (0 or 1 feature values), here we use False to match with the implementation
    else:
        raise NotImplementedError("Unknown task %s"%task)

    return Model, input_size, class_num, normed_mean, normed_std, is_discrete


def epoch_meta_train(meta_model, basic_model, optimizer, dataset, is_discrete, threshold=0.0, no_qt=False):
    meta_model.train()
    basic_model.train()

    cum_loss = 0.0
    preds = []
    labs = []
    # shadow_model_out_list = []
    perm = np.random.permutation(len(dataset))
    for i in perm:
        x, y = dataset[i]

        # torch.load(): Loads an object saved with torch.save() from a file.
        # t1 = timer()
        basic_model.load_state_dict(torch.load(x))
        # t2 = timer()
        # logging.debug(f'load shadow model take time: {t2 - t1} seconds')
        # basic_model.load_state_dict(torch.load(x, map_function=torch.device('cpu'))) # failed
        # meta_model.inp: input to the shadow/target model
        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp) # refer rtNLP_cnn_model.py self.emb_forward()
        else:
            if no_qt:
                 # TODO
                # out = basic_model.forward(meta_model.inp) #.float())  # float() is added for no_qt.
                # if i == perm[0]:
                #     print(f'meta_model.inp: {meta_model.inp.float().device}')
                #     m = meta_model.inp.float().to('cuda')
                #     print(f'meta_model.inp to cuda: {m.device}')
                m = meta_model.inp.float().to('cuda')
                out = basic_model.forward(m)
            else:
                # t1 = timer()
                out = basic_model.forward(meta_model.inp)
                # shadow_model_out_list.append(torch.flatten(out).detach().cpu().numpy())
                # t2 = timer()
                # logging.debug(f'basic_model.forward take time: {t2 - t1} seconds')
        # t1 = timer()
        score = meta_model.forward(out)
        # t2 = timer()
        # logging.debug(f'meta_model.forward take time: {t2 - t1} seconds')
        ####### debugging
        # todo: if needed, add back
        # if i < 5:
        #     if y == 1:
        #         logging.debug(f'{i} TRAIN MAL: basic_model forward out: {torch.flatten(out).detach().cpu().numpy()}')
        #         logging.debug(f'{i} TRAIN MAL: meta_model forward score: {score.item()}')
        #     else:
        #         logging.debug(f'{i} TRAIN BENIGN: basic_model forward out: {torch.flatten(out).detach().cpu().numpy()}')
        #         logging.debug(f'{i} TRAIN BENIGN: meta_model forward score: {score.item()}')

        l = meta_model.loss(score, y)
        # t3 = timer()
        # logging.debug(f'meta_model.loss take time: {t3 - t2} seconds')

        optimizer.zero_grad()
        l.backward()
        # t4 = timer()
        # logging.debug(f'backward take time: {t4 - t3} seconds')
        optimizer.step()
        # t5 = timer()
        # logging.debug(f'optimizer.step take time: {t5 - t4} seconds')


        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.asscalar(np.median(preds))
    # logging.debug(f'TRAIN preds: {preds}')
    # logging.debug(f'TRAIN labs: {labs}')
    logging.debug(f'TRAIN threshold: {threshold}')
    # logging.debug(f'TRAIN shadow model out: {shadow_model_out_list}')
    acc = ( (preds>threshold) == labs ).mean()

    return cum_loss / len(dataset), auc, acc


def epoch_meta_eval_validation(meta_model, basic_model, dataset, is_discrete, threshold=0.0, no_qt=False):
    meta_model.eval()
    basic_model.train()

    cum_loss = 0.0
    preds = []
    labs = []
    # shadow_model_out_list = []
    perm = list(range(len(dataset)))
    for i in perm:
        x, y = dataset[i]
        basic_model.load_state_dict(torch.load(x))
        # basic_model.load_state_dict(torch.load(x, map_function=torch.device('cpu'))) # failed

        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            if no_qt:
                # out = basic_model.forward(meta_model.inp)# .float()) # TODO
                m = meta_model.inp.float().to('cuda')
                out = basic_model.forward(m)
            else:
                out = basic_model.forward(meta_model.inp)
                # shadow_model_out_list.append(out.item())

        score = meta_model.forward(out)
        ####### debugging
        # todo: if needed, add back
        # if i < 5:
        #     if y == 1:
        #         logging.debug(f'{i} VAL MAL: basic_model forward out: {torch.flatten(out).detach().cpu().numpy()}')
        #         logging.debug(f'{i} VAL MAL: meta_model forward score: {score.item()}')
        #     else:
        #         logging.debug(f'{i} VAL BENIGN: basic_model forward out: {torch.flatten(out).detach().cpu().numpy()}')
        #         logging.debug(f'{i} VAL BENIGN: meta_model forward score: {score.item()}')
        # # print(f'score: {score}')

        l = meta_model.loss(score, y)
        # print(f'meta_model loss: {l}')
        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.asscalar(np.median(preds))

    # logging.debug(f'VAL shadow model out: {shadow_model_out_list}')
    # todo: if needed, add back
    # logging.debug(f'VAL preds: {preds}')
    # logging.debug(f'VAL labs: {labs}')
    logging.debug(f'VAL threshold: {threshold}')
    acc = ( (preds>threshold) == labs ).mean()

    return cum_loss / len(preds), auc, acc


def epoch_meta_eval_testing(meta_model, basic_model, dataset, clf, is_discrete,
                            threshold=0.0, no_qt=False, benign_model_type='pytorch',
                            backdoor_model_type='pytorch', roc_curve_path=None):
    meta_model.eval()
    basic_model.train()

    cum_loss = 0.0
    preds = []
    labs = []
    perm = list(range(len(dataset)))
    for i in perm:
        x, y = dataset[i]
        if y == 0:
            if benign_model_type == 'pytorch': # benign model is a pytorch model
                basic_model.load_state_dict(torch.load(x))
                # basic_model.load_state_dict(torch.load(x, map_function=torch.device('cpu'))) # failed, no keyword map_function() for pickle module
                if is_discrete:
                    out = basic_model.emb_forward(meta_model.inp)
                else:
                    if no_qt:
                        # out = basic_model.forward(meta_model.inp)# .float()) # TODO
                        m = meta_model.inp.float().to('cuda')
                        out = basic_model.forward(m)
                    else:
                        out = basic_model.forward(meta_model.inp)

                ####### debugging
                # todo: if needed, add it back
                # if i < 5:
                #     logging.debug(f'{i} TEST BENIGN: basic_model forward out: {torch.flatten(out).detach().cpu().numpy()}')
            else: # benign model is a tensorflow model, more reasonable for comparison
                K.clear_session() # NOTE: add this to fix GPU allocation error
                tf_model = load_model(x)
                out = tf_model.predict(meta_model.inp.detach().cpu().numpy())
                out = torch.from_numpy(out).reshape(out.shape[0], 1).float().cuda()
        else:
            if backdoor_model_type == 'pytorch':
                # backdoored model is a pytorch model
                basic_model.load_state_dict(torch.load(x))
                # basic_model.load_state_dict(torch.load(x, map_function=torch.device('cpu'))) # failed, no keyword map_function() for pickle module
                if is_discrete:
                    out = basic_model.emb_forward(meta_model.inp)
                else:
                    if no_qt:
                        # out = basic_model.forward(meta_model.inp)# .float()) # TODO
                        m = meta_model.inp.float().to('cuda')
                        out = basic_model.forward(m)
                    else:
                        out = basic_model.forward(meta_model.inp)

                ####### debugging
                # todo: if needed, add it back
                # if i < 5:
                #     logging.debug(f'{i} TEST MAL: basic_model forward out: {torch.flatten(out).detach().cpu().numpy()}')
                #     # print(f'MAL: meta_model forward score: {torch.flatten(score)}')
            else:
                # backdoored model is a tensorflow model
                if clf in ['SVM', 'SecSVM']:
                    model = models.load_from_file(x)
                    # detach(): only the values, get rid of the gradients
                    # cpu(): copy the tensor to host memory
                    out = model.clf.decision_function(meta_model.inp.detach().cpu().numpy()) # to numpy()?
                elif clf == 'MLP':
                    K.clear_session() # NOTE: add this to fix GPU allocation error
                    tf_model = load_model(x)
                    out = tf_model.predict(meta_model.inp.detach().cpu().numpy())
                else:
                    raise ValueError(f'clf {clf} not implemented')

                out = torch.from_numpy(out).reshape(out.shape[0], 1).float().cuda()

        score = meta_model.forward(out)
        # todo: if needed, add back
        # if i < 5:
        #     if y == 1:
        #         logging.debug(f'{i} TEST MAL: meta_model forward score: {score.item()}')
        #     else:
        #         logging.debug(f'{i} TEST BENIGN: meta_model forward score: {score.item()}')
        # # print(f'score: {score}')
        # # logging.debug(f'i: {i}, y: {y}, score: {score.detach().cpu()[0]:.4f}')

        l = meta_model.loss(score, y)
        # print(f'meta_model loss: {l}')
        # logging.debug(f'i: {i}, y: {y}, meta_model loss: {l:.8f}')
        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.asscalar(np.median(preds))

    plot_roc_curve(labs, preds, roc_curve_path)

    # todo: if needed, add back
    # # print(f'preds: {preds}')
    # logging.debug(f'Test preds: {preds}')
    # # print(f'labs: {labs}')
    # logging.debug(f'Test labs: {labs}')
    # # print(f'threshold: {threshold}')
    logging.debug(f'Test threshold: {threshold}')
    acc = ( (preds>threshold) == labs ).mean()

    return cum_loss / len(preds), auc, acc


def plot_roc_curve(y_test, y_test_score, roc_curve_path):
    # from matplotlib import rcParams
    # rcParams['font.family'] = 'Georgia'
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.ticker').disabled = True

    FONT_SIZE = 24
    TICK_SIZE = 20
    fig = plt.figure(figsize=(8, 8))
    fpr_plot, tpr_plot, _ = roc_curve(y_test, y_test_score)
    plt.plot(fpr_plot, tpr_plot, lw=2, color='r')
    # plt.gca().set_xscale("log")
    plt.yticks(np.arange(22) / 20.0)
    # plt.yticks(np.arange(0.86, 1.00, 0.02))
    # plt.xticks(np.arange(0, 0.12, 0.02))
    plt.xlim([4e-5, 1.0])
    # plt.xlim([1e-3, 0.1])
    # plt.xlim([0, 0.1])
    plt.ylim([-0.04, 1.04])
    # plt.ylim([0.84, 1.04])
    plt.tick_params(labelsize=TICK_SIZE)
    plt.gca().grid(True)
    # plt.vlines(fpr, 0, 1 - fnr, color="r", lw=2)
    # plt.hlines(1 - fnr, 0, fpr, color="r", lw=2)
    plt.xlabel("False positive rate", fontsize=FONT_SIZE, fontname='Georgia')
    plt.ylabel("True positive rate", fontsize=FONT_SIZE, fontname='Georgia')
    # plt.title(f"{clf_name} Model ROC Curve", fontsize=FONT_SIZE, fontname='Georgia')
    # fig_path = os.path.join('fig/', 'ROC-using-pretrained-Ember.png')
    fig.savefig(roc_curve_path, bbox_inches='tight')
    logging.info('ROC curve saved')



def epoch_meta_eval_on_target_benign(meta_model, basic_model, dataset, report_path,
                                     is_discrete, threshold=0.0, no_qt=False):
    meta_model.eval()
    basic_model.train()

    cum_loss = 0.0
    preds = []
    labs = []
    perm = list(range(len(dataset)))
    for i in perm:
        x, y = dataset[i]
        basic_model.load_state_dict(torch.load(x))

        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            if no_qt:
                # out = basic_model.forward(meta_model.inp).float()) # TODO
                m = meta_model.inp.float().to('cuda')
                out = basic_model.forward(m)
            else:
                out = basic_model.forward(meta_model.inp)

        score = meta_model.forward(out)
        # print(f'score: {score}')
        print(f'TARGET BENIGN: shadow model out: {torch.flatten(out).detach().cpu().numpy()}')
        print(f'TARGET BENIGN: meta_model score: {torch.flatten(score)}')

        l = meta_model.loss(score, y)
        # print(f'meta_model loss: {l}')
        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    # auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.asscalar(np.median(preds))

    print(f'preds: {preds}')

    # with open('meta_preds_benign.txt', 'a') as f:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'a') as f:
        f.write(','.join(map(str, preds)) + '\n')
    print(f'labs: {labs}')
    print(f'threshold: {threshold}')
    acc = ( (preds>threshold) == labs ).mean()
    min_preds, max_preds = np.min(preds), np.max(preds)

    # return cum_loss / len(preds), auc, acc
    return cum_loss / len(preds), acc, min_preds, max_preds



def epoch_meta_eval_with_pretrained_poison_model(meta_model, basic_model, dataset, report_path,
                                                 is_discrete, clf='SVM', threshold=0.0):
    meta_model.eval()
    basic_model.train()

    cum_loss = 0.0
    preds = []
    labs = []
    perm = list(range(len(dataset)))
    for i in perm:
        x, y = dataset[i]
        # basic_model.load_state_dict(torch.load(x))

        # if is_discrete:
        #     out = basic_model.emb_forward(meta_model.inp)
        # else:
        #     out = basic_model.forward(meta_model.inp)

        # TODO: get the poisoned model out with regard to meta_model.inp
        if clf in ['SVM', 'SecSVM']:
            model = models.load_from_file(x)
            # detach(): only the values, get rid of the gradients
            # cpu(): copy the tensor to host memory
            out = model.clf.decision_function(meta_model.inp.detach().cpu().numpy()) # to numpy()?
        elif clf == 'MLP':
            from keras.models import load_model
            model = load_model(x)
            out = model.predict(meta_model.inp.detach().cpu().numpy())
        else:
            raise ValueError(f'clf {clf} not implemented')

        if i == perm[0]:
            print(f'out: {out}, {type(out[0])} shape: {out.shape}')
            print(f'meta_model.inp[0][:10]: {meta_model.inp.detach().cpu().numpy()[0][:10]}')

        out = torch.from_numpy(out).reshape(out.shape[0], 1).float().cuda()
        if i == perm[0]:
            print(f'after reshape out: {type(out[0])}, {out.shape}')
        score = meta_model.forward(out)

        print(f'TARGET POISONED: shadow model out: {torch.flatten(out).detach().cpu().numpy()}')
        print(f'TARGET POISONED: meta_model score: {torch.flatten(score)}')
        # if i == perm[0]:
        # print(f'score: {score}')

        l = meta_model.loss(score, y)
        # if i == perm[0]:
        # print(f'meta_model loss: {l}')
        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    # auc = roc_auc_score(labs, preds) # since we evaluate benign and backdoored separately, we cannot calc AUC.
    if threshold == 'half':
        threshold = np.asscalar(np.median(preds))
    print(f'preds: {preds}')
    # with open('meta_preds_backdoor.txt', 'a') as f:
    with open(report_path, 'a') as f:
        f.write(','.join(map(str, preds)) + '\n')
    print(f'labs: {labs}')
    print(f'threshold: {threshold}')
    acc = ( (preds>threshold) == labs ).mean()
    min_preds, max_preds = np.min(preds), np.max(preds)

    # return cum_loss / len(preds), auc, acc
    return cum_loss / len(preds), acc, min_preds, max_preds


def epoch_meta_train_oc(meta_model, basic_model, optimizer, dataset, is_discrete):
    # oc: one class
    scores = []
    cum_loss = 0.0
    perm = np.random.permutation(len(dataset))
    for i in perm:
        x, y = dataset[i]
        assert y == 1
        basic_model.load_state_dict(torch.load(x))
        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp)
        score = meta_model.forward(out)
        scores.append(score.item())

        loss = meta_model.loss(score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()
        meta_model.update_r(scores)
    return cum_loss / len(dataset)

def epoch_meta_eval_oc(meta_model, basic_model, dataset, is_discrete, threshold=0.0):
    preds = []
    labs = []
    for x, y in dataset:
        basic_model.load_state_dict(torch.load(x))
        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp)
        score = meta_model.forward(out)

        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.asscalar(np.median(preds))
    acc = ( (preds>threshold) == labs ).mean()
    return auc, acc
