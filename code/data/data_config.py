"""
data_config.py

@description: Enum helper to store datasets' filepaths across multiple notebooks.
"""

import os
import sys

from enum import Enum

# Setting project root directory
dir_parts = os.getcwd().split(os.path.sep)
root_index = dir_parts.index('ML-B')
root_path = os.path.sep.join(dir_parts[:root_index + 1])
sys.path.append(root_path + '/code/')

# Data directories
monk_dir = root_path + '/data/monk/'
cup_dir = root_path + '/data/cup/'


class Dataset(Enum):
    """Utility Enum to keep track of the datasets filenames."""
    
    MONK_1 = ('MONK-1', monk_dir + 'monks-1.train', monk_dir + 'monks-1.test')
    MONK_2 = ('MONK-2', monk_dir + 'monks-2.train', monk_dir + 'monks-2.test')
    MONK_3 = ('MONK-3', monk_dir + 'monks-3.train', monk_dir + 'monks-3.test')
    CUP = ('CUP', cup_dir + 'ML-CUP23-TR.csv', cup_dir + 'ML-CUP23-TS.csv')

    def __init__(self, task_name: str, dev_path: str, test_path: str):
        """
        Enum constructor.
        
        Required arguments:
            - task_name: name of the dataset/task.
            - dev_path: filepath for the training/dev set.
            - test_path: filepath for the test set.
        """
        self.task_name = task_name
        self.dev_path = dev_path
        self.test_path = test_path
