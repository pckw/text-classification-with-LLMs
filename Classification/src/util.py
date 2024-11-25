import numpy as np
import xgboost
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import json
import tiktoken


def subsample(train: dict, test: dict, f: float) -> tuple[dict, dict]:
    """
    Subsamples the train and test datasets based on a given factor.

    Args:
        train (dict): The training dataset.
        test (dict): The testing dataset.
        f (float): The subsampling factor.

    Returns:
        tuple[dict, dict]: A tuple containing the subsampled training and testing datasets.
    """
    train_embedding = []
    train_target = []
    train_data = []
    test_embedding = []
    test_target = []
    test_data = []
    for k in range(len(train['target_names'])):
        ii_train = int(sum(train['target'] == k) * f)
        ii_test = int(sum(test['target'] == k) * f)
        train_embedding.extend([i for i,j in zip(train['embedding'], train['target']) if j == k][0:ii_train])
        train_data.extend([i for i,j in zip(train['data'], train['target']) if j == k][0:ii_train])
        train_target.extend([i for i in train['target'] if i == k][0:ii_train])
        test_embedding.extend([i for i,j in zip(test['embedding'], test['target']) if j == k][0:ii_test])
        test_data.extend([i for i,j in zip(test['data'], test['target']) if j == k][0:ii_test])
        test_target.extend([i for i in test['target'] if i == k][0:ii_test])
    train['embedding'] = np.array(train_embedding)
    train['target'] = np.array(train_target)
    train['data'] = np.array(train_data)
    test['embedding'] = np.array(test_embedding)
    test['target'] = np.array(test_target)
    test['data'] = np.array(test_data)
    return train, test

def convert_label_to_numeric(text, valid_label):
    for i,l in enumerate(valid_label):
        if text == l:
            return i
        
def restructure_data(data, valid_label, target_var='Labels', data_var='Inhalt'):
    return {
        'data': [d[data_var] for d in data],
        'target': [d[target_var] for d in data],
        'target_names': valid_label
    }

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens