import random

import numpy as np

from itertools import product
from math import log10

from training.solver import Solver


ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item']


def findBestConfig(model_fn, train_data, train_labels, val_data, val_labels, configs, TARGET, EPOCHS, PATIENCE):
    """
    Get a list of hyperparameter configs for grid/random search, trains a model on all configs 
    and returns the one performing best on validation set according to specific TARGET metric.
    """

    if TARGET == 'loss' or TARGET == 'mean_euclidean_error':
        mode = 'min'
        best_target = float('inf')
    elif TARGET == 'accuracy':
        mode = 'max'
        best_target = float('-inf')
    else:
        raise ValueError(f"Unsupported TARGET value: {TARGET}. Try 'loss', 'accuracy' or 'mean_euclidean_error'!'")
    
    best_config = None
    best_model = None
    results = []
    
    for i in range(len(configs)):
        print("\nEvaluating Config #{} [of {}]:\n".format((i+1), len(configs)), configs[i])
        
        print(type(configs[i]['batch_size']))
        
        model = model_fn(configs[i])
        solver = Solver(model, train_data, train_labels, val_data, val_labels, TARGET, **configs[i])
        solver.train(epochs=EPOCHS, patience=PATIENCE, batch_size=configs[i]['batch_size'])
        results.append(solver.best_model_stats)
        
        if mode == 'max':
            if solver.best_model_stats[f'val_{TARGET}'] > best_target:
                best_target, best_model, best_config = solver.best_model_stats[f'val_{TARGET}'], model, configs[i]
        else:
            if solver.best_model_stats[f'val_{TARGET}'] < best_target:
                best_target, best_model, best_config = solver.best_model_stats[f'val_{TARGET}'], model, configs[i]

    print(f"\nSearch done. Best (val) {f'val_{TARGET}'} = {best_target}")
    print("Best Config:", best_config)
    return best_model, best_config, list(zip(configs, results))


######################################
# GRID SEARCH
######################################
def grid_search(model_fn, train_data, train_labels, val_data, val_labels,
                grid_search_spaces, TARGET, EPOCHS=100, PATIENCE=50):
    """
    A simple grid search based on nested loops to tune learning rate and
    regularization strengths.
    Keep in mind that you should not use grid search for higher-dimensional
    parameter tuning, as the search space explodes quickly.

    Required arguments:
        - model_fn: a function returning a model object

        - train_data: training data
        - train_labels: training gtc labels
        
        - val_data: validation data
        - val_labels: validation gtc labels
            
        - grid_search_spaces: a dictionary where every key corresponds to a
            to-tune-hyperparameter and every value contains an interval or 
            a list of possible values to test.
    
    Optional arguments:
        - TARGET: target metric for optimization. Allowed choices are 'loss' or 'accuracy'
        - EPOCHS: number of epochs each model will be trained on
        - PATIENCE: when to stop early the training process
    """
    configs = []

    # More general implementation using itertools
    for instance in product(*grid_search_spaces.values()):
        configs.append(dict(zip(grid_search_spaces.keys(), instance)))

    return findBestConfig(model_fn, train_data, train_labels, val_data, val_labels,
                          configs, TARGET, EPOCHS, PATIENCE)

def tuning_search_top_configs(results, top=5):
    """"
    Utility function to print a report for the top <top> results of a GridSearchCV the results of a GridSearchCV or RandomSearchCV. 
    Returns a dictionary with the <top> configurations and their results.
    
    Required arguments:
    - results: a GridSearchCV.cv_results_ (or RandomSearchCV) instance containing the grid search results
    
    Optional arguments:
    - top: the best configurations to store/print
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
        
    return best_configs


######################################
# RANDOM SEARCH
######################################
def sample_hparams_spaces(search_spaces, trial=None):
    """"
    Takes search spaces for random search as input, samples 
    accordingly from these spaces and returns the sampled hyper-params as a config-object,
    which will be used to construct solver and/or model.
    
    Args:
    - search_spaces: hparams intervals for tuning
    
    Optional:
    - trial: if provided, use Bayesian suggestions, otherwise sample randomly
    """
    
    config = {}
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

def random_search(model_fn, train_data, train_labels, val_data, val_labels, 
                  random_search_spaces, TARGET='val_loss', NUM_SEARCH=20, EPOCHS=30, PATIENCE=5):
    """
    Samples NUM_SEARCH hyper parameter sets within the provided search spaces
    and returns the best model.
    
    Required arguments:
        - model_fn: a function returning a model object

        - train_data: training data
        - train_labels: training gtc labels
        
        - val_data: validation data
        - val_labels: validation gtc labels
            
        - random_search_spaces: a dictionary where every key corresponds to a
            to-tune-hyperparameter and every value contains an interval or a list of possible
            values to test.
    
    Optional arguments:
        - TARGET: target metric for optimization. Allowed choices are 'val_loss' or 'val_avg_f1'
        - NUM_SEARCH: number of configurations to test
        - EPOCHS: number of epochs each model will be trained on
        - PATIENCE: when to stop early the training process
    """
    
    configs = []
    for _ in range(NUM_SEARCH):
        configs.append(sample_hparams_spaces(random_search_spaces))

    return findBestConfig(model_fn, train_data, train_labels, val_data, val_labels, configs, TARGET, EPOCHS, PATIENCE)
