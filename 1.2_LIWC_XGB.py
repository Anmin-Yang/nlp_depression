"""
Demo for Vocational School Students
Change the path to run for different groups
Training and Testing usign XGBoost

@author: Anmin(BNU)
"""
import numpy as np
import pandas as pd
import random
from math import isnan
import os 
import time 

#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#rom sklearn.linear_model import LogisticRegressionCV
#from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import multiprocessing

# define data path
feature_path = '/Users/anmin/Documents/NLP/AI_text_5000/textmind_features/feature_all.csv'
data_path = '/Users/anmin/Documents/NLP/AI_text_5000/data_clean_all.xlsx'

# read data
data = np.array(pd.read_excel(data_path,engine='openpyxl'))
feature_df = pd.read_csv(feature_path)
feature = np.array(feature_df)

# eliminate id with null text
data = data[(data[:,2] == 4)] # select junior school school students

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

save_path = '/Users/anmin/Documents/NLP/XBG_results/LIWC_with_acc'
results = {'accuracy': [],
        'precision': [],
           'recall': [],
           'f1': []}

for i in range(10):
    print(f'----------{i}th sample begins---------------')
    start = time.time()
    matched_data = match_size(d_train, n_train)
    X = matched_data

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

    results['accuracy'].append(clf.score(X_test, y_test))
    results['precision'].append(report['weighted avg']['precision'])
    results['recall'].append(report['weighted avg']['recall'])
    results['f1'].append(report['weighted avg']['f1-score'])
    end = time.time()
    print(f'used time : {end - start}s')
np.save(os.path.join(save_path,
                    'vocational.npy'),
        results)

