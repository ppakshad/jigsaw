# -*- coding: utf-8 -*-

"""
models.py
~~~~~~~~~

Available target models:
    * SVMModel - a base class for SVM-like models
        - SVM - Standard linear SVM using scikit-learn implementation
        - SecSVM - Secure SVM variant using a PyTorch implementation (based on [1])

[1] Yes, Machine Learning Can Be More Secure! [TDSC 2019]
    -- Demontis, Melis, Biggio, Maiorca, Arp, Rieck, Corona, Giacinto, Roli

"""
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC']='true'
from numpy.random import seed
import random
random.seed(1)
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import logging
import numpy as np
import tensorflow as tf
import sys
import pickle
import random
import subprocess
import traceback
import h5py
# import psutil
from sklearn.utils import class_weight
import ujson as json
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from collections import OrderedDict
from timeit import default_timer as timer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import BaggingClassifier
from tqdm import tqdm

from keras.layers import Input, Dense, Dropout
from keras.layers.normalization import BatchNormalization # , LayerNormalization (not available)

from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import lib.secsvm

from apg import utils

sys.path.append('backdoor/')
from mysettings import config
import myutil
from logger import LoggingCallback


class SVMModel:
    """Base class for SVM-like classifiers."""

    def __init__(self, X_filename, y_filename, meta_filename, num_features=None, save_folder=config['models'], file_type='json'):
        self.X_filename = X_filename
        self.y_filename = y_filename
        self.meta_filename = meta_filename
        self._num_features = num_features
        self.save_folder = save_folder
        self.file_type = file_type
        self.clf, self.vec = None, None
        self.column_idxs = [] # feature indexes after feature selection
        self.X_train, self.y_train, self.m_train = [], [], []
        self.X_test, self.y_test, self.m_test = [], [], []
        self.feature_weights, self.benign_weights, self.malicious_weights = [], [], []
        self.weight_dict = OrderedDict()

    def generate(self, save=True):
        """Load and fit data for new model."""
        logging.debug('No saved models found, generating new model...')

        X_train, X_test, y_train, y_test, m_train, m_test, self.vec, train_test_random_state = load_features(
            self.X_filename, self.y_filename, self.meta_filename, self.save_folder, self.file_type, self.svm_c, load_indices=False)

        self.column_idxs = self.perform_feature_selection(X_train, y_train)

        features = np.array([self.vec.feature_names_[i] for i in self.column_idxs])
        with open(f'models/apg/SVM/{self._num_features}_features_full_name_{self.svm_c}_{self.max_iter}.csv', 'w') as f:
            for fea in features:
                f.write(fea + '\n')

        # NOTE: should use scipy sparse matrix instead of numpy array, the latter takes much more space when save as pickled file.
        self.X_train = X_train[:, self.column_idxs]
        self.X_test = X_test[:, self.column_idxs]

        self.y_train, self.y_test = y_train, y_test
        self.m_train, self.m_test = m_train, m_test

        self.clf = self.fit(self.X_train, self.y_train)

        try:
            features = [self.vec.feature_names_[i] for i in self.column_idxs]

            w = self.get_feature_weights(features)
            self.feature_weights, self.benign_weights, self.malicious_weights = w # these 3 attributes have the same format of a list: each item in the list is (feature_name, index, weight)
            self.weight_dict = OrderedDict(
                (w[0], w[2]) for w in self.feature_weights)
        except:
            logging.warning(f'self.vec and feature weights are not calculated')

        if save:
            self.save_to_file()

    def dict_to_feature_vector(self, d):
        """Generate feature vector given feature dict."""
        return self.vec.transform(d)[:, self.column_idxs]

    def get_feature_weights(self, feature_names):
        """Return a list of features ordered by weight.

        Each feature has it's own 'weight' learnt by the classifier.
        The sign of the weight determines which class it's associated
        with and the magnitude of the weight describes how influential
        it is in identifying an object as a member of that class.

        Here we get all the weights, associate them with their names and
        their original index (so we can map them back to the feature
        representation of apps later) and sort them from most influential
        benign features (most negative) to most influential malicious
        features (most positive). By default, only negative features
        are returned.

        Args:
            feature_names: An ordered list of feature names corresponding to cols.

        Returns:
            list, list, list: List of weight pairs, benign features, and malicious features.

        """
        assert self.clf.coef_[0].shape[0] == len(feature_names)

        coefs = self.clf.coef_[0]
        weights = list(zip(feature_names, range(len(coefs)), coefs))
        weights = sorted(weights, key=lambda row: row[-1])

        # Ignore 0 weights
        benign = [x for x in weights if x[-1] < 0]
        malicious = [x for x in weights if x[-1] > 0][::-1]
        return weights, benign, malicious

    def perform_feature_selection(self, X_train, y_train):
        """Perform L2-penalty feature selection."""
        if self._num_features is not None:
            logging.info(red('Performing L2-penalty feature selection'))
            # NOTE: we should use dual=True here, use dual=False when n_samples > n_features, otherwise you may get a ConvergenceWarning
            # The ConvergenceWarning means not converged, which should NOT be ignored
            # see discussion here: https://github.com/scikit-learn/scikit-learn/issues/17339
            # selector = LinearSVC(C=self.svm_c, max_iter=self.max_iter, dual=False)
            selector = LinearSVC(C=self.svm_c, max_iter=self.max_iter, dual=True)
            selector.fit(X_train, y_train)

            cols = np.argsort(np.abs(selector.coef_[0]))[::-1]
            cols = cols[:self._num_features]
        else:
            cols = [i for i in range(X_train.shape[1])]
        return cols

    def save_to_file(self):
        myutil.create_parent_folder(self.model_name)
        with open(self.model_name, 'wb') as f:
            pickle.dump(self, f, protocol=4)


class SVM(SVMModel):
    """Standard linear SVM using scikit-learn implementation."""

    def __init__(self, X_filename, y_filename, meta_filename, save_folder, num_features=None, svm_c=1, max_iter=1000, file_type='json'):
        super().__init__(X_filename, y_filename, meta_filename, num_features, save_folder, file_type)
        self.model_name = self.generate_model_name()
        self.svm_c = svm_c
        self.max_iter = max_iter

    def fit(self, X_train, y_train):
        logging.debug('(fit) Creating model')
        clf = LinearSVC(C=self.svm_c, max_iter=self.max_iter, dual=True)
        clf.fit(X_train, y_train)
        return clf

    def generate_model_name(self):
        model_name = f'svm'
        model_name += '.p' if self._num_features is None else '-f{}.p'.format(self._num_features)
        return os.path.join(self.save_folder, model_name)


class SecSVM(SVMModel):
    """Secure SVM variant using a PyTorch implementation."""

    def __init__(self, X_filename, y_filename, meta_filename, save_folder, num_features=None,
                 secsvm_k=0.2, secsvm_lr=0.0001,
                 secsvm_batchsize=1024, secsvm_nepochs=75,
                 seed_model=None, file_type='json'):
        super().__init__(X_filename, y_filename, meta_filename, num_features, save_folder, file_type)
        self._secsvm_params = {
            'batchsize': secsvm_batchsize,
            'nepochs': secsvm_nepochs,
            'lr': secsvm_lr,
            'k': secsvm_k
        }
        self._seed_model = seed_model
        self.save_folder = save_folder
        self.model_name = self.generate_model_name()

    def fit(self, X_train, y_train):
        logging.debug('Creating model')
        clf = lib.secsvm.SecSVM(lr=self._secsvm_params['lr'],
                                batchsize=self._secsvm_params['batchsize'],
                                n_epochs=self._secsvm_params['nepochs'],
                                K=self._secsvm_params['k'],
                                seed_model=self._seed_model)
        clf.fit(X_train, y_train)
        return clf

    def generate_model_name(self):
        model_name = 'secsvm-k{}-lr{}-bs{}-e{}'.format(
            self._secsvm_params['k'],
            self._secsvm_params['lr'],
            self._secsvm_params['batchsize'],
            self._secsvm_params['nepochs'])
        if self._seed_model is not None:
            model_name += '-seeded'
        model_name += '.p' if self._num_features is None else '-f{}.p'.format(self._num_features)
        return os.path.join(self.save_folder, model_name)


def load_from_file(model_filename):
    logging.debug(f'Loading model from {model_filename}...')
    with open(model_filename, 'rb') as f:
        return pickle.load(f)


def load_features(X_filename, y_filename, meta_filename, save_folder, file_type='json', svm_c=1, load_indices=True):
    train_test_random_state = None
    if file_type == 'json':
        with open(X_filename, 'rt') as f: # rt is the same as r, t means text
            X = json.load(f)
            [o.pop('sha256') for o in X]  # prune the sha, uncomment if needed
        with open(y_filename, 'rt') as f:
            y = json.load(f)
        with open(meta_filename, 'rt') as f:
            meta = json.load(f)

        X, y, vec = vectorize(X, y)

        if load_indices:
            logging.info(yellow('Loading indices...'))
            chosen_indices_file = config['indices']
            with open(chosen_indices_file, 'rb') as f:
                train_idxs, test_idxs = pickle.load(f)
        else:
            train_test_random_state = random.randint(0, 1000)

            train_idxs, test_idxs = train_test_split(
                range(X.shape[0]),
                stratify=y, # to keep the same benign VS mal ratio in training and testing
                test_size=0.33,
                random_state=train_test_random_state)

            filepath = f'indices-{train_test_random_state}.p' if svm_c == 1 else f'indices-{train_test_random_state}-c-{svm_c}.p'
            filepath = os.path.join(save_folder, filepath)
            myutil.create_parent_folder(filepath)
            with open(filepath, 'wb') as f:
                pickle.dump((train_idxs, test_idxs), f)

        X_train = X[train_idxs]
        X_test = X[test_idxs]
        y_train = y[train_idxs]
        y_test = y[test_idxs]
        m_train = [meta[i] for i in train_idxs]
        m_test = [meta[i] for i in test_idxs]
    elif file_type == 'npz':
        npz_data = np.load(X_filename)
        X_train, y_train = npz_data['X_train'], npz_data['y_train']
        X_test, y_test = npz_data['X_test'], npz_data['y_test']
        m_train = None
        m_test = None
        vec = None
    elif file_type == 'hdf5':
        with h5py.File(X_filename, 'r') as hf:
            X_train = sparse.csr_matrix(np.array(hf.get('X_train')))
            y_train = np.array(hf.get('y_train'))
            X_test = sparse.csr_matrix(np.array(hf.get('X_test')))
            y_test = np.array(hf.get('y_test'))
        m_train = None
        m_test = None
        vec = None
    elif file_type == 'variable':
        X_train, X_test = X_filename
        y_train, y_test = y_filename
        m_train = None
        m_test = None
        vec = None
    elif file_type == 'bodmas':
        npz_data = np.load(X_filename)
        X = npz_data['X']  # all the feature vectors
        y = npz_data['y']  # labels, 0 as benign, 1 as malicious

        train_test_random_state = random.randint(0, 1000)

        train_idxs, test_idxs = train_test_split(
            range(X.shape[0]),
            stratify=y, # to keep the same benign VS mal ratio in training and testing
            test_size=0.33,
            random_state=train_test_random_state)

        filepath = f'indices-{train_test_random_state}.p'
        filepath = os.path.join(save_folder, filepath)
        myutil.create_parent_folder(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump((train_idxs, test_idxs), f)

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        ''' NOTE: MLP classifier needs normalization for Ember features (value range from -654044700 to 4294967300'''
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = y[train_idxs]
        y_test = y[test_idxs]
        m_train = None
        m_test = None
        vec = None
    else:
        raise ValueError(f'file_type {file_type} not supported')

    logging.info(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
    logging.info(f'y_train: {y_train.shape}, y_test: {y_test.shape}')
    return X_train, X_test, y_train, y_test, m_train, m_test, vec, train_test_random_state


def vectorize(X, y):
    vec = DictVectorizer(sparse=True) # default is True, will generate sparse matrix
    X = vec.fit_transform(X)
    y = np.asarray(y)
    return X, y, vec



class MLPModel:
    """Base class for MLP classifiers."""

    def __init__(self, X_filename, y_filename, meta_filename,
                 dataset,
                 dims,
                 dropout,
                 model_name,
                 activation='relu',
                 verbose=1,
                 num_features=None, save_folder=config['models'], file_type='json'):
        self.X_filename = X_filename
        self.y_filename = y_filename
        self.meta_filename = meta_filename
        self.dataset = dataset
        self._num_features = num_features
        self.save_folder = save_folder
        self.file_type = file_type
        # self.clf, self.vec = None, None # NOTE: remove these two --> don't save them to the model pickle file.
        self.column_idxs = []
        self.X_train, self.y_train, self.m_train = [], [], []
        self.X_test, self.y_test, self.m_test = [], [], []
        self.feature_weights, self.benign_weights, self.malicious_weights = [], [], []
        self.dims = dims  # e.g., [10000, 1024, 1]
        self.model_name = model_name
        self.act = activation
        self.dropout = dropout
        self.verbose = verbose

    def build(self):
        # build a MLP model with Keras functional API
        n_stacks = len(self.dims) - 1
        input_tensor = Input(shape=(self.dims[0],), name='input')
        x = input_tensor
        for i in range(n_stacks - 1):
            x = Dense(self.dims[i + 1],
                      activation=self.act, name='clf_%d' % i)(x)

            if self.dropout > 0:
                x = Dropout(self.dropout, seed=42)(x)

        x = Dense(self.dims[-1], activation='sigmoid',
                  name='clf_%d' % (n_stacks - 1))(x)
        output_tensor = x
        model = Model(inputs=input_tensor,
                      outputs=output_tensor, name='MLP')
        if self.verbose:
            logging.debug('MLP classifier summary: ' + str(model.summary()))
        return model

    def generate(self,
                 retrain=False,
                 lr=0.001,
                 batch_size=32,
                 epochs=50,
                 loss='binary_crossentropy',
                 class_weight=None,
                 calc_fea_weights=False,
                 save=True,
                 random_state=42,
                 half_training=False,
                 prev_batch_poisoned_model_path=None,
                 use_last_weight=False):

        logging.debug(f'use_last_weight: {use_last_weight}')
        """Load and fit data for new model."""
        logging.debug('No saved models found, generating new model...')

        '''change self.vec to vec'''
        X_train, X_test, y_train, y_test, m_train, m_test, \
            vec, self.train_test_random_state = load_features(self.X_filename, self.y_filename,
                                                                    self.meta_filename, self.save_folder,
                                                                    self.file_type, load_indices=False)

        logging.warning(f'MLP model generate load_features X_train: {X_train.shape}')

        self.selected_features_file = os.path.join(self.save_folder, f'selected_{self._num_features}_features_r{self.train_test_random_state}.p')
        if self.dataset == 'bodmas':
            self.column_idxs = [i for i in range(X_train.shape[1])]
        else:
            self.column_idxs = self.perform_feature_selection(X_train, y_train) # if n_features = None, it would not perform feature selection

        logging.warning(f'self.file_type: {self.file_type}')

        if half_training and self.file_type == 'json':
            logging.info(f'before half_training: X_train {X_train.shape}, y_train {y_train.shape}')
            X_train_first, X_train_second, \
                y_train_first, y_train_second = train_test_split(X_train, y_train, stratify=y_train,
                                                                test_size=0.5, random_state=random_state)
            X_train = X_train_first
            y_train = y_train_first
            logging.info(f'after half_training: X_train {X_train.shape}, y_train {y_train.shape}')

        self.X_train = X_train[:, self.column_idxs]
        self.X_test = X_test[:, self.column_idxs]
        self.y_train, self.y_test = y_train, y_test
        self.m_train, self.m_test = m_train, m_test

        '''change self.clf to clf to save memory, also self.clf is not the best model, the h5 file is the best MLP model'''
        clf = self.fit(self.X_train, self.y_train, lr, batch_size, epochs, loss, class_weight, retrain,
                       random_state, half_training, prev_batch_poisoned_model_path, use_last_weight)

        if save:
            self.save_to_file()

        return clf

    def fit(self, X_train, y_train, lr, batch_size, epochs, loss, class_weight, retrain, random_state=42,
            half_training=False, prev_batch_poisoned_model_path=None, use_last_weight=False):
        dims_str = '-'.join(map(str, self.dims))

        if half_training:
            self.mlp_h5_model_path = os.path.join(self.save_folder, f'mlp_{dims_str}_lr{lr}_b{batch_size}_e{epochs}_d{self.dropout}_r{random_state}_half_training.h5')
        else:
            self.mlp_h5_model_path = os.path.join(self.save_folder, f'mlp_{dims_str}_lr{lr}_b{batch_size}_e{epochs}_d{self.dropout}_r{random_state}.h5')

        if not os.path.exists(self.mlp_h5_model_path) and retrain == False:
            retrain = True

        if retrain:
            begin = timer()
            logging.info(f'Training MLP model...')
            model = self.build()

            # load weights from model in previous batch
            if use_last_weight:
                logging.info(f'loaded weight from prev batch')
                logging.debug(f'prev_batch_poisoned_model_path: {prev_batch_poisoned_model_path}')
                model.load_weights(prev_batch_poisoned_model_path, by_name=True)

            X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(X_train, y_train, range(X_train.shape[0]),
                                                                                  test_size=0.33,
                                                                                  random_state=random_state, # NOTE: previously we use 42
                                                                                  shuffle=True)

            # configure and train model.
            pretrain_optimizer = Adam(lr=lr)
            model.compile(loss=loss,
                          optimizer=pretrain_optimizer,
                          metrics=['accuracy'])

            myutil.create_parent_folder(self.model_name)

            mcp_save = ModelCheckpoint(self.mlp_h5_model_path,
                                        monitor='val_acc',
                                        save_best_only=True,
                                        save_weights_only=False,
                                        verbose=self.verbose,
                                        mode='max')
            if self.verbose:
                callbacks = [mcp_save, LoggingCallback(logging.debug)]
            else:
                callbacks = [mcp_save]
            history = model.fit(X_train, y_train, # change to y_train_onehot for multi-class
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(X_val, y_val), # change to y_val_onehot for multi-class
                                verbose=self.verbose,
                                class_weight=class_weight,
                                callbacks=callbacks)
            end = timer()
            logging.info(f'Training MLP finished, time: {end - begin:.1f} seconds')

        t1 = timer()
        clf = load_model(self.mlp_h5_model_path)
        logging.info(f'load_model in models.py time: {timer() - t1} seconds')

        return clf

    def dict_to_feature_vector(self, d):
        """Generate feature vector given feature dict."""
        return self.vec.transform(d)[:, self.column_idxs]


    def perform_feature_selection(self, X_train, y_train):
        # WARNING: this step is very time consuming, so cache it.
        if os.path.exists(self.selected_features_file):
            with open(self.selected_features_file, 'rb') as f:
                cols = pickle.load(f)
        else:
            """Perform L2-penalty feature selection."""
            if self._num_features is not None:
                logging.info(red('Performing L2-penalty feature selection'))
                selector = LinearSVC(C=1, max_iter=10000, dual=True)
                selector.fit(X_train, y_train)

                cols = np.argsort(np.abs(selector.coef_[0]))[::-1]
                cols = cols[:self._num_features]
                with open(self.selected_features_file, 'wb') as f: # NOTE: only save this file when need to perform feature selection
                    pickle.dump(cols, f, protocol=4)
            else:
                cols = [i for i in range(X_train.shape[1])]

        return cols

    def save_to_file(self):
        logging.info('Saving model file as pickle file...')
        myutil.create_parent_folder(self.model_name)
        with open(self.model_name, 'wb') as f:
            pickle.dump(self, f, protocol=4)

