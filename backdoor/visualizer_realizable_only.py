# -*- coding: utf-8 -*-
'''
Solve a minimum mask that can achieve B+M->0, S+M->0, and R+M->1.
Allow to change the final mask from real value to binary or not.

Add a variable m1 to optimize from 2171 realizable features only.
'''

import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC']='true'
from numpy.random import seed
import random
random.seed(1)
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
from keras import backend as K
import tensorflow as tf
import logging

from sklearn.metrics import accuracy_score, log_loss
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.optimizers import Adam, SGD

import utils_backdoor

from decimal import Decimal


class Visualizer:

    UPSAMPLE_SIZE = None # useless in our dataset
    INTENSITY_RANGE = 'raw' # useless in our dataset
    # type of regularization of the mask
    REGULARIZATION = 'l1'
    # threshold of attack success rate for dynamically changing cost
    ATTACK_SUCC_THRESHOLD = 0.99
    PATIENCE = 10
    # multiple of changing cost, down multiple is the square root of this
    COST_MULTIPLIER = 1.5
    # if resetting cost to 0 at the beginning
    # default is true for full optimization, set to false for early detection
    # the cost would initialize as 1e-3 in the line "if self.cost == 0 and avg_loss_acc >= self.attack_succ_threshold:"
    RESET_COST_TO_ZERO = False
    # min/max of mask
    MASK_MIN = 0
    MASK_MAX = 1
    # min/max of raw pixel intensity
    COLOR_MIN = 0
    COLOR_MAX = 1
    # number of color channel
    IMG_COLOR = 1 # original 3
    # whether to shuffle during each epoch
    SHUFFLE = True
    # batch size of optimization
    BATCH_SIZE = 32
    # verbose level, 0, 1 or 2
    VERBOSE = 1
    # whether to return log or not
    RETURN_LOGS = True
    # whether to save last pattern or best pattern
    SAVE_LAST = False
    # epsilon used in tanh
    EPSILON = K.epsilon() # i.e., 1e-7
    # early stop flag
    EARLY_STOP = True
    # early stop threshold
    EARLY_STOP_THRESHOLD = 0.99
    # early stop patience
    EARLY_STOP_PATIENCE = 2 * PATIENCE
    # save tmp masks, for debugging purpose
    SAVE_TMP = True
    # dir to save intermediate masks
    TMP_DIR = 'tmp'
    # whether input image has been preprocessed or not
    RAW_INPUT_FLAG = False

    INITIALIZER = tf.keras.initializers.RandomUniform(minval=0, maxval=1)


    def __init__(self, model, intensity_range, regularization, input_shape,
                 init_cost, steps, mini_batch, lr, num_classes,
                 upsample_size=UPSAMPLE_SIZE,
                 attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
                 patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
                 reset_cost_to_zero=RESET_COST_TO_ZERO,
                 mask_min=MASK_MIN, mask_max=MASK_MAX,
                 color_min=COLOR_MIN, color_max=COLOR_MAX, img_color=IMG_COLOR,
                 shuffle=SHUFFLE, batch_size=BATCH_SIZE, verbose=VERBOSE,
                 return_logs=RETURN_LOGS, save_last=SAVE_LAST,
                 epsilon=EPSILON,
                 early_stop=EARLY_STOP,
                 early_stop_threshold=EARLY_STOP_THRESHOLD,
                 early_stop_patience=EARLY_STOP_PATIENCE,
                 save_tmp=SAVE_TMP, tmp_dir=TMP_DIR,
                 raw_input_flag=RAW_INPUT_FLAG,
                 use_concrete=True, temperature=0.1,
                 initializer=INITIALIZER, normalize_choice='clip',
                 is_first_iteration=False, convert_mask_to_binary=False):

        assert regularization in {None, 'l1', 'l2'}

        self.model = model
        self.intensity_range = intensity_range
        self.regularization = regularization
        self.input_shape = input_shape
        self.init_cost = init_cost
        self.steps = steps
        self.mini_batch = mini_batch
        self.lr = lr
        self.num_classes = num_classes
        self.upsample_size = upsample_size
        self.attack_succ_threshold = attack_succ_threshold
        self.patience = patience
        self.cost_multiplier_up = cost_multiplier # lambda_1 will be multiplied by 1.5
        self.cost_multiplier_down = cost_multiplier ** 1.5 # lambda_1 will be divided by 1.837
        self.reset_cost_to_zero = reset_cost_to_zero
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.color_min = color_min
        self.color_max = color_max
        self.img_color = img_color
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.verbose = verbose
        self.return_logs = return_logs
        self.save_last = save_last
        self.epsilon = epsilon
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.save_tmp = save_tmp
        self.tmp_dir = tmp_dir
        self.raw_input_flag = raw_input_flag

        self.initializer = initializer
        self.use_concrete = use_concrete
        self.temp = temperature
        self.normalize_choice = normalize_choice
        self.convert_mask_to_binary = convert_mask_to_binary

        mask_shape = input_shape

        '''TODO: the only difference between the first iteration and later iterations of the
                 alternate optimization algorithm.
           Later iterations do not have shape=mask_shape, don't know why.
        '''
        if is_first_iteration:
            with tf.variable_scope('p', reuse=tf.AUTO_REUSE):
                self.p = tf.get_variable('p', shape=mask_shape, dtype=tf.float32, initializer=self.initializer)
        else:

            with tf.variable_scope('p', reuse=tf.AUTO_REUSE):
                self.p = tf.get_variable('p', initializer=self.initializer)

        ## normalize variables
        if self.normalize_choice == 'sigmoid':
            logging.debug('Using sigmoid normalization.')
            self.p_normalized = tf.sigmoid(self.p)
        elif self.normalize_choice == 'tanh':
            logging.debug('Using tanh normalization.')
            self.p_normalized = (tf.tanh(self.p + 1)) / (2 + tf.keras.backend.epsilon())
        else:
            logging.debug('Using clip normalization.')
            self.p_normalized = tf.minimum(1.0, tf.maximum(self.p, 0.0))

        ## discrete variables to continuous variables.
        if self.use_concrete:
            self.mask = self.concrete_transformation(self.p_normalized, mask_shape, self.batch_size, self.temp)
        else:
            self.mask = self.p_normalized

        # NOTE realizable: add this as filter, it contains 2171 realizable features
        self.m1 = tf.placeholder(tf.float32, shape=mask_shape)

        self.reverse_mask = tf.ones_like(self.mask) - self.mask * self.m1   # x * (1 - m*m1)

        input_raw_tensor = model.get_input_at(0)
        X_adv_tensor = input_raw_tensor * self.reverse_mask + self.mask * self.m1  # x * (1 - mask * m1) + mask * m1

        self.output_tensor = model(X_adv_tensor)
        self.output_tensor = tf.reshape(self.output_tensor, shape=(-1,))
        y_true_tensor = tf.placeholder(tf.float32, shape=(self.batch_size, ))

        self.loss_ce = binary_crossentropy(y_true_tensor, self.output_tensor)
        self.loss_acc = binary_accuracy(y_true_tensor, self.output_tensor)

        if self.regularization is None:
            self.loss_reg = K.constant(0)
        elif self.regularization is 'l1':
            self.loss_reg = (K.sum(K.abs(self.mask * self.m1)))
        elif self.regularization is 'l2':
            self.loss_reg = K.sqrt(K.sum(K.square(self.mask * self.m1)))

        cost = self.init_cost
        self.cost_tensor = K.variable(cost)
        self.loss = self.loss_ce + self.loss_reg * self.cost_tensor

        self.opt = Adam(lr=self.lr, beta_1=0.5, beta_2=0.9)

        self.updates = self.opt.get_updates(
            params=[self.p],
            loss=self.loss)

        logging.critical(f'var_train: {tf.trainable_variables(scope="p")}')
        self.train = K.function(
            [input_raw_tensor, y_true_tensor, self.m1],
            [self.loss_ce, self.loss_reg, self.loss, self.loss_acc],
            updates=self.updates)

        self.y_pred_func = K.function([input_raw_tensor, self.m1], [self.output_tensor])

        pass

    @staticmethod
    def concrete_transformation(p, mask_shape, batch_size, temp=1.0 / 10.0):
        """ Use concrete distribution to approximate binary output.
        :param p: Bernoulli distribution parameters.
        :param temp: temperature.
        :param batch_size: size of samples.
        :return: approximated binary output.
        """
        epsilon = np.finfo(float).eps  # 1e-16

        unif_noise = tf.random_uniform(shape=(1, mask_shape[0]),
                                       minval=0, maxval=1)
        reverse_theta = tf.ones_like(p) - p
        reverse_unif_noise = tf.ones_like(unif_noise) - unif_noise

        appro = tf.log(p + epsilon) - tf.log(reverse_theta + epsilon) + \
                tf.log(unif_noise) - tf.log(reverse_unif_noise)
        logit = appro / temp

        return tf.sigmoid(logit)

    def reset_opt(self):

        K.set_value(self.opt.iterations, 0)
        for w in self.opt.weights:
            K.set_value(w, np.zeros(K.int_shape(w)))

        pass

    def reset_state(self, mask_init):

        logging.info('resetting state')

        # setting cost
        if self.reset_cost_to_zero:
            self.cost = 0
        else:
            self.cost = self.init_cost
        K.set_value(self.cost_tensor, self.cost)

        # resetting optimizer states
        self.reset_opt()

        pass


    # function to create a list containing mini-batches
    def create_mini_batches(self, X, y, y_tag, batch_size):
        mini_batches = []
        data = np.hstack((X, y, y_tag))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // batch_size

        for i in range(n_minibatches):
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
            X_mini = mini_batch[:, :-2]
            Y_mini = mini_batch[:, -2].reshape((-1, ))
            Y_tag_mini = mini_batch[:, -1].reshape((-1, ))

            mini_batches.append((X_mini, Y_mini, Y_tag_mini))

        return mini_batches

    def visualize(self, m1, trigger_idx, X_train_part, Y_train_part, Y_tag_part, y_target, mask_init, mask_file, report_folder):
        '''y_target is a useless param in subset backdoor idea, it's not all 0, we will feed a batch of target labels instead'''
        # optimzier's internal states before running the optimization
        self.reset_state(mask_init) # just set lambda_1 = 1e-3, also set all weights as 0

        # best optimization results
        mask_best = None
        reg_best = float('inf')
        acc_best = 0

        # logs and counters for adjusting balance cost
        logs = []
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # counter for early stop
        early_stop_counter = 0
        early_stop_reg_best = reg_best

        benign_cnt = np.where(Y_tag_part == 0)[0].shape[0]
        remain_cnt = np.where(Y_tag_part == 1)[0].shape[0]
        subset_cnt = np.where(Y_tag_part == 2)[0].shape[0]
        logging.info(f'benign_cnt: {benign_cnt}, remain_cnt: {remain_cnt}, subset_cnt: {subset_cnt}')
        best_benign_acc = 0
        best_remain_acc = 0
        best_subset_acc = 0

        # loop start
        for step in range(self.steps):

            # record loss for all mini-batches
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            sklearn_acc_list = []

            benign_acc_list = []
            remain_acc_list = []
            subset_acc_list = []

            # sklearn_ce_list = []
            mini_batches = self.create_mini_batches(X_train_part, Y_train_part, Y_tag_part, self.batch_size)

            for idx, mini_batch in enumerate(mini_batches):
                X_batch, Y_batch, Y_tag_batch = mini_batch
                Y_target = Y_batch

                (loss_ce_value,
                    loss_reg_value,
                    loss_value,
                    loss_acc_value) = self.train([X_batch, Y_target, m1])

                y_predict = self.y_pred_func([X_batch, m1])
                y_predict = y_predict[0].reshape(-1,)
                y_predict = np.array([1 if y > 0.5 else 0 for y in y_predict])

                benign_idx = np.where(Y_tag_batch == 0)[0]
                benign_acc = accuracy_score(Y_tag_batch[benign_idx], y_predict[benign_idx])
                remain_idx = np.where(Y_tag_batch == 1)[0]
                remain_acc = accuracy_score(Y_tag_batch[remain_idx], y_predict[remain_idx])
                subset_idx = np.where(Y_tag_batch == 2)[0]
                subset_acc = accuracy_score(np.array([0] * len(subset_idx)), y_predict[subset_idx])

                benign_acc_list.append(benign_acc * benign_idx.shape[0])
                remain_acc_list.append(remain_acc * remain_idx.shape[0])
                subset_acc_list.append(subset_acc * subset_idx.shape[0])

                sklearn_acc = accuracy_score(Y_target.reshape(-1,), y_predict)
                sklearn_acc_list.append(sklearn_acc)
                loss_ce_list.extend(list(loss_ce_value.flatten()))
                loss_reg_list.extend(list(loss_reg_value.flatten()))
                loss_list.extend(list(loss_value.flatten()))
                loss_acc_list.extend(list(loss_acc_value.flatten()))

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_loss_acc = np.mean(loss_acc_list)
            avg_sklearn_acc = np.mean(sklearn_acc_list)
            avg_benign_acc = np.nansum(benign_acc_list) / benign_cnt
            avg_remain_acc = np.nansum(remain_acc_list) / remain_cnt
            avg_subset_acc = np.nansum(subset_acc_list) / subset_cnt

            if avg_sklearn_acc > acc_best:
                logging.info(f'updating best acc from {acc_best:.4f} to {avg_sklearn_acc:.4f}')
                acc_best = avg_sklearn_acc

                best_benign_acc = avg_benign_acc
                best_remain_acc = avg_remain_acc
                best_subset_acc = avg_subset_acc

            # check to save best mask or not
            if avg_sklearn_acc >= self.attack_succ_threshold and avg_loss_reg < reg_best:
                logging.info(f'avg_sklearn_acc: {avg_sklearn_acc}, avg_loss_acc: {avg_loss_acc:.4f}, avg_loss_reg: {avg_loss_reg:.4f}')
                mask_best = K.eval(self.p_normalized)
                reg_best = avg_loss_reg

                if self.convert_mask_to_binary:
                    mask_best = [1 if x > 0.5 else 0 for x in mask_best]

                with open(mask_file, 'w') as f:
                    f.write(','.join(map(str, mask_best)))

            if self.verbose != 0:
                if self.verbose == 2 or step % (self.steps // 10) == 0:
                    logging.info('step: %3d, cost: %.4f, attack: %.3f, sklearn_acc: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f benign_acc: %.3f remain_acc: %.3f subset_acc: %.3f' %
                                (step, Decimal(self.cost), avg_loss_acc, avg_sklearn_acc, avg_loss,
                                avg_loss_ce, avg_loss_reg, reg_best, avg_benign_acc, avg_remain_acc, avg_subset_acc))

            # save log
            logs.append((step,
                         avg_loss_ce, avg_loss_reg, avg_loss, avg_loss_acc,
                         reg_best, self.cost))

            # check early stop
            if self.early_stop:
                # only terminate if a valid attack has been found
                if reg_best < float('inf'):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best: # 0.99 * best
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if (cost_down_flag and
                        cost_up_flag and
                        early_stop_counter >= self.early_stop_patience):
                    logging.info('early stop')
                    break

            if self.cost == 0 and avg_sklearn_acc >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    self.cost = self.init_cost
                    K.set_value(self.cost_tensor, self.cost)
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    logging.info('initialize cost to %.2E' % Decimal(self.cost))
            else:
                cost_set_counter = 0

            if avg_sklearn_acc >= self.attack_succ_threshold:
                # NOTE: if the cross entropy loss is satisfied for 5 epochs, then we should
                # increase the lambda_1 so that we can decrease the mask size
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                # otherwise we should reduce the lambda_1 so that we can reduce the mask loss and
                # optimize cross entropy loss first to achieve a higher avg_loss_acc
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if self.verbose == 2:
                    logging.debug('up cost from %.8f to %.8f' %
                                 (Decimal(self.cost),
                                 Decimal(self.cost * self.cost_multiplier_up)))
                self.cost *= self.cost_multiplier_up # cost will be multiplied by 1.5
                K.set_value(self.cost_tensor, self.cost)
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                if self.verbose == 2:
                    logging.debug('down cost from %.8f to %.8f' %
                                 (Decimal(self.cost),
                                 Decimal(self.cost / self.cost_multiplier_down))) # cost will be divided by 1.837
                self.cost /= self.cost_multiplier_down
                K.set_value(self.cost_tensor, self.cost)
                cost_down_flag = True

        # save the final version if self.save_last is True or did't get a mask_best
        if mask_best is None or self.save_last:
            logging.warning(f'mask_best is None or save_last is True, use the last mask as the mask')
            mask_best = K.eval(self.p_normalized)

            with open(f'{report_folder}/mask_last_real_{trigger_idx}.txt' , 'w') as f:
                f.write(','.join(map(str, mask_best)))

            if self.convert_mask_to_binary:
                mask_best = [1 if x > 0.5 else 0 for x in mask_best]

            with open(mask_file, 'w') as f:
                f.write(','.join(map(str, mask_best)))

            acc_best = avg_sklearn_acc
            best_benign_acc = avg_benign_acc
            best_remain_acc = avg_remain_acc
            best_subset_acc = avg_subset_acc

        logging.critical(f'best acc: {acc_best:.4f}, best benign acc: {best_benign_acc:.3f}, ' + \
                        f'best remain acc: {best_remain_acc:.3f}, best subset acc: {best_subset_acc:.3f}')
        if self.return_logs:
            return mask_best * m1, acc_best, best_benign_acc, best_subset_acc, best_remain_acc
        else:
            return mask_best * m1
