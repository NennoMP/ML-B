import random

import numpy as np


######################################
# GRID SEARCH
######################################
def grid_search_top_configs(results, top=5):
    """"
    Utility function to print a report for the top <top> results of a GridSearchCV the results of a GridSearchCV. Returns a dictionary with the <top> configurations and their results.
    
    Required arguments:
    - results: a GridSearchCV.cv_results_ instance containing the grid search results
    
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
        
    return best_configs, candidates
