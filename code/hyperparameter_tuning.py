"""
hyperparameter_tuning.py

@description: Helpers for a hyper-parameter tuning pipepline, i.e. grid-search and random-search utils.
"""


import random

import numpy as np

from itertools import product
from math import log10
from sklearn.model_selection import KFold

from training.solver import Solver


# Allowed random search lists types
ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item']


def findBestConfig(
    model_fn, in_dim: int, x_dev, y_dev, configs: dict, target: str, 
    n_splits: int, epochs: int, patience: int):
    """
    Trains a model using K-Fold cross-validation for a set of configurations
    and returns the best model based on the mean (validation) performance 
    across folds.
    
    Args:
        - model_fn: a function that returns the model instance
        - in_dim: input dimension
        - x_dev: development data
        - y_dev: development labels    
        - configs: a set of hparams configurations to test
        - target: monitored metric for early stopping
        - n_splits: number of splits for KFold
        - epochs: number of training epochs
        - patience: patience value for early stopping
    """
    
    # Check if valid target metric
    aux = target.replace('val_', '')
    if aux == 'loss' or aux == 'mean_euclidean_error':
        mode = 'min'
        best_target = float('inf')
    elif aux == 'accuracy':
        mode = 'max'
        best_target = float('-inf')
    else:
        raise ValueError(
            f"Unsupported TARGET value: {target}. Try 'loss/val_loss', 'accuracy/val_accuracy' or 'mean_euclidean_error/val_mean_euclidean_error'!'")
   
    # Tracking variables
    best_config = None
    std_dev = None
    best_model = None
    results = []

    # KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=128)
    
    for i in range(len(configs)):
        print(f"Evaluating Config #{i+1} [of {len(configs)}]: {configs[i]}\n")

        # Initialize variables for mean metric calculation across folds
        fold_count = 0
        fold_metrics = []  # List to store metrics of each fold

        # Perform KFold cross-validation
        for train_idx, val_idx in kf.split(x_dev):
            print(f"Evaluating Fold #{fold_count} [of {n_splits}]:\n")
            
            fold_train_data, fold_val_data = x_dev[train_idx], x_dev[val_idx]
            fold_train_labels, fold_val_labels = y_dev[train_idx], y_dev[val_idx]
            
            # Train
            model = model_fn(configs[i], in_dim)
            solver = Solver(model, fold_train_data, fold_train_labels, fold_val_data, fold_val_labels, target, **configs[i])
            solver.train(epochs=epochs, patience=patience, batch_size=configs[i]['batch_size'])
            
            fold_metrics.append(solver.best_model_stats[target])
            fold_count += 1

        mean_metric = np.mean(fold_metrics) # mean
        std_dev_metric = np.std(fold_metrics) # std dev.
        results.append((mean_metric, std_dev_metric))

        # Update best config/model if current mean metric is better
        if (mode == 'max' and mean_metric > best_target) or (mode == 'min' and mean_metric < best_target):
            std_dev = std_dev_metric
            best_target, best_model, best_config = mean_metric, model, configs[i]
            best_config[target], best_config['std_dev'] = mean_metric, std_dev

    print(f"\nSearch done. Best mean score {target}: {best_target} - std dev: {std_dev}")
    print("Best Config:", best_config)

    return best_model, best_config, list(zip(configs, results))


######################################
# GRID SEARCH
######################################
def grid_search(model_fn, in_dim: int, x_dev, y_dev,
                grid_search_spaces: dict, target: str, 
                N_SPLITS=5, EPOCHS=100, PATIENCE=50):
    """
    A simple grid search based on nested loops to tune hyper-parameters.
    Keep in mind that you should not use grid search for higher-dimensional
    parameter tuning, as the search space explodes quickly.

    Args:
        - model_fn: a function returning a model object
        - in_dim: input dimension
        - x_dev: development data
        - y_dev: development labels
        - grid_search_spaces: a dictionary where every key corresponds to a
            to-tune-hyperparameter and every value contains an interval of 
            possible values to test.
        - target: monitored metric for early stopping
    
    Optional:
        - N_SPLITS: number of splits for kfold cross-validation
        - EPOCHS: number of epochs each model will be trained on
        - PATIENCE: when to stop early the training process
    """
    configs = []

    # General implementation using itertools
    for instance in product(*grid_search_spaces.values()):
        configs.append(dict(zip(grid_search_spaces.keys(), instance)))

    return findBestConfig(model_fn, in_dim, x_dev, y_dev,
                          configs, target, N_SPLITS, EPOCHS, PATIENCE)


def tuning_search_top_configs(results: dict, top=5):
    """"
    Utility function to print a report for the top <top> results of a 
    GridSearchCV the results of a GridSearchCV or RandomSearchCV. 
    Returns a dictionary with the <top> configurations and their results.
    
    Required arguments:
    - results: a GridSearchCV.cv_results_ (or RandomSearchCV) instance containing the grid search results
    
    Optional arguments:
        - top: the k best configurations to store and print
    """
    
    best_configs = {}
    candidates = [candidate for candidate in np.argsort(results['rank_test_score'])[:top]]
    for rank, candidate in enumerate(candidates):
        print(f"Model rank {rank+1} - Config: {results['params'][candidate]}")
        print(f"Mean score {results['mean_test_score'][candidate]:.4f} - Std score: {results['std_test_score'][candidate]:.4f}\n")
        best_configs[rank + 1] = {
            'config': results['params'][candidate],
            'Mean score': round(results['mean_test_score'][candidate], 4),
            'Std score': round(results['std_test_score'][candidate], 4)
        }
        
    return best_configs, candidates


######################################
# RANDOM SEARCH
######################################
def sample_hparams_spaces(search_spaces: dict, trial=None):
    """"
    Takes search spaces for random search as input, samples 
    accordingly from these spaces and returns the sampled hyper-params 
    as a config-object, which will be used to construct solver and/or model.
    
    Args:
    - search_spaces: hparams intervals for tuning
    
    Optional:
    - trial: if provided, use Bayesian suggestions, otherwise sample randomly
    """
    
    config = {}
    # Check type of each interval
    for key, (values, mode) in search_spaces.items():
        if mode == "float":
            config[key] = (
                trial.suggest_float(key, values[0], values[1]) if trial
                else random.uniform(values[0], values[1])
            )
        elif mode == "int":
            config[key] = (
                trial.suggest_int(key, values[0], values[1]) if trial
                else np.random.randint(values[0], values[1])
            )
        elif mode == "item":
            config[key] = (
                trial.suggest_categorical(key, values) if trial
                else np.random.choice(values)
            )
        elif mode == "log":
            if trial:
                config[key] = trial.suggest_float(key, values[0], values[1], log=True)
            else:
                log_min, log_max = np.log(values)
                config[key] = np.exp(np.random.uniform(log_min, log_max))

    return config

def random_search(model_fn, in_dim: int, x_dev, y_dev,
                  random_search_spaces, target, 
                  N_SPLITS=5, NUM_SEARCH=20, EPOCHS=30, PATIENCE=5):
    """
    Samples <NUM_SEARCH> hyper parameter sets within the provided search spaces
    and returns the best model.
    
    Args:
        - model_fn: a function returning a model object
        - in_dim: input dimension
        - x_dev: development data
        - y_dev: development labels
        - random_search_spaces: a dictionary where every key corresponds to a
            to-tune-hyperparameter and every value contains an interval of 
            possible values to test.
        - target: target metric for early stopping
    
    Optional arguments:
        - N_SPLITS: number of splits for kfold cross-validation
        - NUM_SEARCH: number of configurations to test
        - EPOCHS: number of epochs each model will be trained on
        - PATIENCE: when to stop early the training process
    """
    
    configs = []
    # Create random configurations to test, from values in given spaces
    for _ in range(NUM_SEARCH):
        configs.append(sample_hparams_spaces(random_search_spaces))

    return findBestConfig(model_fn, in_dim, x_dev, y_dev, configs, target, N_SPLITS, EPOCHS, PATIENCE)
