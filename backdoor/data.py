import os
os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)

import sys
import logging
import traceback
import pickle
import json
from os import listdir
from os.path import isfile, join, expanduser
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

sys.path.append('backdoor/')
from mysettings import config, _project
import myutil


home = expanduser("~")


def combine_benign_and_mal_metadata(raw_feature_vectors_folder, mal_metadata_path, full_metadata_path):
    if os.path.exists(full_metadata_path):
        logging.debug(f'metadata file exists, no need to re-generate')
    else:
        files = [f for f in listdir(raw_feature_vectors_folder) if isfile(join(raw_feature_vectors_folder, f))]
        logging.debug(f'total # of feature vector files: {len(files)}')

        with open(full_metadata_path, 'w') as fout:
            with open(mal_metadata_path, 'r') as fin:
                for line in fin: # including header
                    fout.write(line)
            for f in files:
                fout.write(f + ',\n') # benign files have no malware family


def load_cached_data(saved_file):
    saved_data = np.load(saved_file)
    train_data = saved_data['train']
    test_data = saved_data['test']
    return train_data, test_data


def vectorize(X):
    vec = DictVectorizer()
    X = vec.fit_transform(X)
    return X, vec


def dict_to_feature_vector(vec, d):
    """Generate feature vector given feature dict."""
    return vec.transform(d) # unseen features would be assigned as 0


def load_json(filename):
    with open(filename, 'r') as f:
        l = json.load(f) # list
    return np.array(l)


def dump_json(data_list, filename):
    myutil.create_parent_folder(filename)
    with open(filename, 'w') as f:
        json.dump(data_list, f)
