"""
Demo for All Students
Change the path to run for different groups

Use LIWC as feature,
employ the following models:
1. LR
2. SVM
3. RM

@author: Anmin(BNU)
"""
import numpy as np
import pandas as pd
import os 
import time 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# define data path
feature_path = '/Users/anmin/Documents/NLP/AI_text_5000/textmind_features/feature_all.csv'
data_path = '/Users/anmin/Documents/NLP/AI_text_5000/data_clean_all.xlsx'

# read data
data = np.array(pd.read_excel(data_path,engine='openpyxl'))
feature_df = pd.read_csv(feature_path)
feature = np.array(feature_df)

# eliminate id with null text
data = data[(data[:,2] == 2)] # select junior school school students

cut_list = []
for i in range(data.shape[0]):
    if data[i,0] not in feature[:,0]: # this id did not write text
        cut_list.append(i)

data = np.delete(data,cut_list,axis=0)

# generate depressed and normal feature matrix
mask_depressed = (data[:,-1] == 1)
mask_normal = (data[:,-1] == 0)

depressed_data = data[mask_depressed]
normal_data = data[mask_normal]

depress_feature = np.zeros((102,))
for id in depressed_data[:,0]:
    index = np.argwhere(feature[:,0]==id)[0][0]
    depress_feature = np.vstack((depress_feature,feature[index,1:]))
depress_feature = depress_feature[1:,:]

normal_feature = np.zeros((102,))
for id in normal_data[:,0]:
    index = np.argwhere(feature[:,0]==id)[0][0]
    normal_feature = np.vstack((normal_feature,feature[index,1:]))
normal_feature = normal_feature[1:,:]

# match sampel size between depressed and normal and form one matrix
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

### prediction ##################################################################
test_ratio = 0.2
d_train, d_test = train_test_split(depress_feature, test_size=test_ratio)
n_test = normal_feature[np.random.choice(normal_feature.shape[0], d_test.shape[0], 
                        replace=False)]
def gen_train_from_test(complete_arr, train_arr):
    arr = np.zeros((1,train_arr.shape[1]))
    for row in complete_arr:
        if row.tolist() not in train_arr.tolist():
            arr = np.vstack((arr, row))
    
    arr = arr[1:,:]
    return arr

n_train = gen_train_from_test(normal_feature, n_test)

# test dataset 
X_test = np.vstack((d_test, n_test))
y_test = [0]*d_test.shape[0] + [1]*d_test.shape[0]

save_path = '/Users/anmin/Documents/NLP/traditional_ml_results/LIWC'
results = {}
weights = {}

key_lst = ['precision', 'recall', 'f1-score']

for i in range(10):
    matched_data = match_size(d_train, n_train)
    X = matched_data
    class_size = int(X.shape[0]/2)
    y = [0]*class_size + [1]*class_size

    # Logistic Regression 
    print('--------------------------------')
    print(f'Start {i}_LR')
    if 'LR' not in results:
        results['LR'] = {'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'f1-score': []}
    clf = LogisticRegressionCV(cv=5, random_state=42).fit(X, y)

    report = classification_report(y_test, clf.predict(X_test), output_dict=True)
    for key in key_lst:
        results['LR'][key].append(report['weighted avg'][key])
    results['LR']['accuracy'].append(clf.score(X_test, y_test))

    weights[i] = clf.coef_

    # SVM 
    print(f'Start {i}_SVM')
    if 'SVM' not in results:
        results['SVM'] = {'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'f1-score': []}
    param_grid = {'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid']} 
    grid = GridSearchCV(SVC(random_state=42), param_grid, n_jobs=-1, verbose=2)
    grid.fit(X, y)

    report = classification_report(y_test, grid.predict(X_test), output_dict=True)
    for key in key_lst:
        results['SVM'][key].append(report['weighted avg'][key])
    results['SVM']['accuracy'].append(clf.score(X_test, y_test))


    # RF 
    print(f'Start {i}_RF')
    if 'RF' not in results:
        results['RF'] = {'accuracy': [],
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
        results['RF'][key].append(report['weighted avg'][key])
    results['RF']['accuracy'].append(clf.score(X_test, y_test))

save_path = os.path.join('/Users/anmin/Documents/NLP/',
                         'traditional_ml_results')
np.save(os.path.join(save_path,
                     'LIWC_with_acc',
                     'junior.npy'), results)
np.save(os.path.join(save_path,
                     'LIWC_weights',
                     'junior.npy'), weights)
