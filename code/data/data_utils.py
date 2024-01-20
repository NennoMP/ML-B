import json
import os

import numpy as np
import pandas as pd


MONK_HEADER = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'ID']
MONK_ATTRIBUTES = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']

CUP_HEADER = ['ID','a1','a2','a3','a4','a5','a6', 'a7', 'a8', 'a9', 'a10', 'class_x', 'class_y', 'class_z']


######################################
# MONK UTILS
######################################
def load_monk(dev_path: str, test_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Utility function to load a monk's problem train and test dataset. Returns the train and test
    attributes (one-hot encoding) and labels.
    
    Required arguments:
        - dev_path: train data filepath
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
    Utility function to store the final results (i.e., best configurations and test performance)
    of a model w.r.t. a monk's problem.
    
    Required arguments:
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
    Utility function to load CUP train and test dataset. Returns the train attributes and labels, 
    and the test attributes.
    
    Required arguments:
        - dev_path: train data filepath
        - test_path: test data filepath
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



def store_cup_result(out_dir: str, config: dict, report: dict, blind_test_preds: np.ndarray):
    """
    Utility function to store the final results (i.e., best configurations and test performance)
    of a model w.r.t. CUP.
    
    Required arguments:
        Required arguments:
        - out_dir: output directory
        - config: hparams final configuration
        - report: MSE and MEE report for train-val-internal test
        - blind_test_preds: predictions for the blind test
    """
    
    # Check if the directory exists, create it if not
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # Writing best_config to a file
    with open(out_dir + 'config.json', 'w') as outf:
        json.dump(config, outf, indent=4)

    # Writing train-val-internal test report
    with open(out_dir + 'report.json', 'w') as outf:
        json.dump(report, outf, indent=4)
            
    with open(out_dir + 'test_preds.csv', 'w') as outf:
        # Team Info
        outf.write("# Matteo Pinna, Leonardo Caridi, Marco Sanna\n")
        outf.write("# ACD-TEAM\n")
        outf.write("# ML-CUP23 v2\n")
        outf.write("# 20/01/2023\n")

        # Writing predictions
        for i, pred in enumerate(blind_test_preds, 1):
            outf.write(f"{i},{','.join(map(str, pred))}\n")
