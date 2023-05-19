"""
Word2Vec as feature,
employ the following models:
1. LR
2. SVM
3. RM

Loop over ages 

@author: Anmin(BNU)
"""
import numpy as np
import pandas as pd
#import jieba
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

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
save_path = '/Users/anmin/Documents/NLP/traditional_ml_results/word2vec_with_acc'
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
        X = match_size(d_train, n_train)
        class_size = int(X.shape[0]/2)
        y = [0]*class_size + [1]*class_size

        # Logistic Regression 
        print(f'-----------Start {train_type} of round {i} in LR---------------------')
        if 'LR' not in results[train_type]:
            results[train_type]['LR'] = {'accuracy': [],
                                        'precision': [],
                                        'recall': [],
                                        'f1-score': []}
        clf = LogisticRegressionCV(cv=5, random_state=42).fit(X, y)

        report = classification_report(y_test, clf.predict(X_test), output_dict=True)
        for key in key_lst:
            results[train_type]['LR'][key].append(report['weighted avg'][key])
        results[train_type]['LR']['accuracy'].append(clf.score(X_test, y_test))

        # SVM 
        print(f'-----------Start {train_type} of round {i} in SVM---------------------')
        if 'SVM' not in results[train_type]:
            results[train_type]['SVM'] = {'accuracy': [],
                                        'precision': [],
                                        'recall': [],
                                        'f1-score': []}
        param_grid = {'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']} 
        grid = GridSearchCV(SVC(random_state=42), param_grid, n_jobs=-1)
        grid.fit(X, y)

        report = classification_report(y_test, grid.predict(X_test), output_dict=True)
        for key in key_lst:
            results[train_type]['SVM'][key].append(report['weighted avg'][key])
        results[train_type]['SVM']['accuracy'].append(grid.score(X_test, y_test))

        # RF 
        print(f'-----------Start {train_type} of round {i} in RF---------------------')
        if 'RF' not in results[train_type]:
            results[train_type]['RF'] = {'accuracy': [],
                                        'precision': [],
                                        'recall': [],
                                        'f1-score': []}
        param_grid = [{'min_samples_split': np.linspace(0.05, 0.6, num=10),
                    'max_leaf_nodes': np.arange(2,20,3)}]
        grid = GridSearchCV(RandomForestClassifier(random_state=42),
                                param_grid=param_grid, n_jobs=-1)
        grid.fit(X, y)

        report = classification_report(y_test, grid.predict(X_test), output_dict=True)
        for key in key_lst:
            results[train_type]['RF'][key].append(report['weighted avg'][key])
        results[train_type]['RF']['accuracy'].append(grid.score(X_test, y_test))


np.save(os.path.join(save_path,
                     'results_dic.npy'), 
        results)

