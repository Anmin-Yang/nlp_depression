"""
Training and Testing using XGBoost
Word2Vec as feature
Loop over ages 

@author: Anmin(BNU)
"""
import numpy as np
from numpy import mean
import pandas as pd
import jieba
import os
import time

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import multiprocessing

def match_size(data_prep,data_to_match):
    """
    match sample size between depressed and normal

    Parameter
    ----------
    data_prep: ndarray
        feature array whose size required to be matched
    data_to_match: ndarray
        feature array to match data_prep

    Return
    ----------
    matched_data: ndarray
        matched size randomly chosen form normal feature pool
        and concatenate as one matrix

    """
    size = data_prep.shape[0]
    np.random.shuffle(data_to_match)
    to_match = data_to_match[:size,:]
    matched_data = np.vstack((data_prep,to_match))

    return matched_data

def gen_train_from_test(complete_arr, train_arr):
        arr = np.zeros((1,train_arr.shape[1]))
        for row in complete_arr:
            if row.tolist() not in train_arr.tolist():
                arr = np.vstack((arr, row))
        
        arr = arr[1:,:]
        return arr

data_path = '/Users/anmin/Documents/NLP/word2vec_matrix'
save_path = '/Users/anmin/Documents/NLP/XBG_results/word2vec_with_acc'
train_types = ['all', 'junior', 'primary', 'senior', 'vocational']
key_lst = ['precision', 'recall', 'f1-score']
results = {}

for train_type in train_types:
    if train_type not in results:
        results[train_type] = {}
        
    depressed_feature = np.load(os.path.join(data_path,
                                            f'depress_{train_type}.npy'),
                                allow_pickle=True)
    normal_feature = np.load(os.path.join(data_path,
                                            f'normal_{train_type}.npy'),
                            allow_pickle=True)

    test_ratio = 0.2
    d_train, d_test = train_test_split(depressed_feature, test_size=test_ratio)
    n_test = normal_feature[np.random.choice(normal_feature.shape[0], d_test.shape[0], 
                            replace=False)]

    n_train = gen_train_from_test(normal_feature, n_test)

    # test dataset 
    X_test = np.vstack((d_test, n_test))
    y_test = [0]*d_test.shape[0] + [1]*d_test.shape[0]

    for i in range(10):
        print(f'--------------------{train_type}---{i}-------------------------')
        results[train_type][i] = {'accuracy': [],
                                    'precision': [],
                                    'recall': [],
                                    'f1-score': []}
        
        X = match_size(d_train, n_train)
        class_size = int(X.shape[0]/2)
        y = [0]*class_size + [1]*class_size

        ### XGB ### 
        xgb_model = xgb.XGBClassifier(objective='binary:logistic',
                                    n_jobs=multiprocessing.cpu_count() // 2)
        clf = GridSearchCV(xgb_model, {'max_depth': range (2, 10, 1),
                                        'n_estimators': range(40, 220, 40),
                                        'learning_rate': [0.1, 0.01, 0.05]}, verbose=2,
                            n_jobs=2)
        clf.fit(X, y)
        
        report = classification_report(y_test, clf.predict(X_test), output_dict=True)
        print(report)

        results[train_type][i]['accuracy'].append(clf.score(X_test, y_test))
        results[train_type][i]['precision'].append(report['weighted avg']['precision'])
        results[train_type][i]['recall'].append(report['weighted avg']['recall'])
        results[train_type][i]['f1-score'].append(report['weighted avg']['f1-score'])


np.save(os.path.join(save_path,
                     'results_dic.npy'), 
        results)