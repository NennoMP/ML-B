"""
data_utils.py

@description: Helpers to load the datasets in an appropriate format, and store their results.
"""

import json
import os

import numpy as np
import pandas as pd


# MONK HEADER/FEATURES
MONK_HEADER = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'ID']
MONK_ATTRIBUTES = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']

# CUP FEATURES
CUP_HEADER = ['ID','a1','a2','a3','a4','a5','a6', 'a7', 'a8', 'a9', 'a10', 'class_x', 'class_y', 'class_z']


######################################
# MONK UTILS
######################################
def load_monk(dev_path: str, test_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Utility function to load a MONK train and test dataset. Returns the train and test
    attributes (one-hot encoding) and labels.
    
    Args:
        - dev_path: training data filepath
        - test_path: test data filepath
    """
    
    # Load the data
    monk_dev = pd.read_csv(dev_path, header=None, delimiter=' ', skipinitialspace=True, names=MONK_HEADER)
    monk_test = pd.read_csv(test_path, header=None, delimiter=' ', skipinitialspace=True, names=MONK_HEADER)
    
    # Set index and remove it
    monk_dev.index = monk_dev.pop('ID').str.extract('(\d+)', expand=False).astype(int)
    monk_dev.index.name = None
    monk_test.index = monk_test.pop('ID').str.extract('(\d+)', expand=False).astype(int)
    monk_test.index.name = None

    # To One-hot Encoding
    monk_dev_ohe = pd.get_dummies(monk_dev, columns=MONK_ATTRIBUTES)
    monk_test_ohe = pd.get_dummies(monk_test, columns=MONK_ATTRIBUTES)

    # Extract features (x) and labels (y)
    x_dev, y_dev = monk_dev_ohe.drop('class', axis=1).values, monk_dev_ohe['class'].values
    x_test, y_test = monk_test_ohe.drop('class', axis=1).values, monk_test_ohe['class'].values
    
    return x_dev, y_dev, x_test, y_test


def store_monk_result(out_dir: str, config: dict, report: dict):
    """
    Utility function to store the final results, i.e. best configuration and train/test 
    performance, of a model w.r.t. a MONK problem.
    
    Args:
        - out_dir: output directory
        - config: hparams final configuration
        - report: loss and accuracy report for dev-test
    """
    
    # Check if the directory exists, create it if not
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Writing best_config to a file
    with open(out_dir + 'config.json', 'w') as outf:
        json.dump(config, outf, indent=4)

    # Writing dev-test report
    with open(out_dir + 'report.json', 'w') as outf:
        json.dump(report, outf, indent=4)
        

######################################
# CUP UTILS
######################################
def load_cup(dev_path: str, test_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    """
    Utility function to load CUP development and test dataset. Returns the development attributes and 
    labels, as well as the blind test labels.
    
    Args:
        - dev_path: train data filepath
        - test_path: blind test data filepath
    """
    
    # Load the data
    cup_dev = pd.read_csv(dev_path, header=None, delimiter=',', skiprows=7, names=CUP_HEADER)
    cup_test = pd.read_csv(test_path, header=None, delimiter=',', skiprows=7, names=CUP_HEADER)
    
    # Drop the label columns from the test set
    cup_test = cup_test.drop(['class_x', 'class_y', 'class_z'], axis=1)
    
    # Set index and remove it
    cup_dev.index = cup_dev.pop('ID').astype(int)
    cup_dev.index.name = None
    cup_test.index = cup_test.pop('ID').astype(int)
    cup_test.index.name = None
    
    # Extract features (x) and labels (y)
    x_dev, y_dev = cup_dev.drop(['class_x', 'class_y', 'class_z'], axis=1).values, cup_dev[['class_x', 'class_y', 'class_z']].values
    x_test = cup_test.values
    
    return x_dev, y_dev, x_test


def store_cup_result(out_dir: str, config: dict, report: dict, blind_test_preds: np.ndarray, is_poly=False):
    """
    Utility function to store the final results - i.e. best configuration, train-val-internal test performance
    and the blind test predictions - of a model w.r.t. CUP.
    
    Args:
        - out_dir: output directory
        - config: final model hyper-parameters
        - report: MSE and MEE report for train-val-internal test
        - blind_test_preds: predictions for the blind test
    
    Optional:
        - is_poly: specifies if these are polynomial pre-processing results. Defaults to False
    """
    
    # Check if the directory exists, create it if not
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    poly = ''
    # Check if it's a PolynomialFeatures result or not
    if is_poly:
        poly = '_poly'
        
    # Writing best_config to a file
    with open(out_dir + f'config{poly}.json', 'w') as outf:
        json.dump(config, outf, indent=4)

    # Writing train-val-internal test report
    with open(out_dir + f'report{poly}.json', 'w') as outf:
        json.dump(report, outf, indent=4)
            
    with open(out_dir + f'test_preds{poly}.csv', 'w') as outf:
        # Team Info
        outf.write("# Matteo Pinna, Leonardo Caridi, Marco Sanna\n")
        outf.write("# ACD-TEAM\n")
        outf.write("# ML-CUP23 v2\n")
        outf.write("# 20/01/2023\n")

        # Writing predictions
        for i, pred in enumerate(blind_test_preds, 1):
            outf.write(f"{i},{','.join(map(str, pred))}\n")
