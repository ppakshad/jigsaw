# -*- coding: utf-8 -*-

"""
settings.py
~~~~~~~~~~~

Configuration options for the pipeline.

"""
import os

# The absolute path to the root folder of this project
_project_path = '/home/entrophy/apg-release'
# The absolute path of the folder containing compiled Java components
_components_path = '/home/entrophy/apg-release/java-components/build'


def _project(base):
    return os.path.join(_project_path, base)


def _components(base):
    return os.path.join(_components_path, base)


config = {
    # Experiment settings
    'models': _project('data/models/'),
    'all_benign': _project('data/models/pure_all_benign'),
    'dict_sandr': _project('data/sandr-mapping.json'),
    'avg_features': _project('data/median_distribution.json'),
    'X_dataset': '/media/nas/datasets/android/2016-2018-ts/2016-2018-ts-X.json',
    #'X_dataset' : '/media/nas/datasets/android/2016-2018-ts/2016-2018-sandr-X.json',
    'y_dataset': '/media/nas/datasets/android/2016-2018-ts/2016-2018-ts-y.json',
    'meta': '/media/nas/datasets/android/2016-2018-ts/2016-2018-ts-meta.json',
    'indices': '/media/nas/datasets/android/2016-2018-ts/models/linear-svm/indices.p',
    # only needed if using fixed indices
    # Java components
    'extractor': _components('extractor.jar'),
    'injector': _components('injector.jar'),
    'template_injector': _components('templateinjector.jar'),
    'cc_calculator': _components('cccalculator.jar'),
    'class_lister': _components('classlister.jar'),
    'classes_file': _project('all_classes.txt'),
    'extractor_timeout': 180,
    'injector_timeout': 240,
    'cc_calculator_timeout': 240,
    # Other necessary components
    'android_sdk': '/opt/android-sdk-linux/',
    'template_path': _project('template'),
    'mined_slices': _project('mined-slices'),
    'opaque_pred': _project('opaque-preds/sootOutput'),
    'threshold' : "10",
    'resigner': _project('apk-signer.jar'),
    'feature_extractor': '/home/entrophy/feature-extractor',
    # Storage for generated bits-and-bobs
    'tmp_dir': '/media/nas/tmp/apg',
    'ice_box': '/media/nas/apg/slice-out-entrophy-all/',
    'results_dir': '/media/nas/apg/apg-results',
    'goodware_location': '/media/nas/datasets/android/samples/Androzoo',
    'storage_radix': 3,  # Use if apps are stored with a radix (e.g., radix 3: root/0/0/A/00A384545.apk)
    # Miscellaneous options
    'tries': 1,
    'nprocs_preload': 8,
    'nprocs_evasion': 10,
    'nprocs_transplant': 8
}
