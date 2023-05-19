"""
Concate prediction results into DataFrame 
1. ML_traditional_results
2. Ml_essmeble results 
3. wordtovec traditionbal results 
4. wordtovec essemble results 

@author: Anmin(BNU)
"""
import numpy as np
import pandas as pd 
import os 

root_path = '/Users/anmin/Documents/NLP'
grades = ['all', 'primary', 'junior', 'senior', 'vocational']

# Use LIWC as features 
for grade in grades:
    result_t = np.load(os.path.join(root_path,
                                    'traditional_ml_results',
                                    'LIWC_with_acc',
                                    f'{grade}.npy'),
                        allow_pickle=True).item()
    result_e = np.load(os.path.join(root_path,
                                    'XBG_results',
                                    'LIWC_with_acc',
                                    f'{grade}.npy'),
                        allow_pickle=True).item()

    # tranditional methods 
    models = list(result_t.keys())

    arr = np.zeros((4))

    for model in models:
        data_temp = result_t[model]
        data_lst = []
        for _, val in data_temp.items():
            miu = np.round(np.mean(val), 2)
            sigma = np.round(np.std(val), 2)
            data_lst.append(f'{miu}±{sigma}')
        arr = np.vstack([arr, np.array(data_lst)])
    arr = arr[1:,:]

    # gradient boosting 
    data_lst = []
    for _, val in result_e.items():
        miu = np.round(np.mean(val), 2)
        sigma = np.round(np.std(val), 2)
        data_lst.append(f'{miu}±{sigma}')
    arr = np.vstack([arr, np.array(data_lst)])
    df = pd.DataFrame(np.concatenate([np.array(['LR',
                                        'SVM',
                                        'RF',
                                        'RF_GB']).reshape(-1,1), arr],
                                    axis=1),
                    columns=['Models',
                            'Accuracy',
                            'Precision',
                            'Recall',
                            'F-measure']
                            )

    df.to_csv(os.path.join(root_path,
                        'results_table',
                        'LIWC_with_acc',
                        f'{grade}.csv'))

# Use word2vec as features  
result_t = np.load(os.path.join(root_path,
                                    'traditional_ml_results',
                                    'word2vec_with_acc',
                                    'results_dic.npy'),
                        allow_pickle=True).item()
for grade in grades:
    result_t_temp = result_t[grade]
    result_e = np.load(os.path.join(root_path,
                                    'XBG_results',
                                    'LIWC_with_acc',
                                    f'{grade}.npy'),
                        allow_pickle=True).item()
    
    # tranditional methods 
    models = list(result_t_temp.keys())

    arr = np.zeros((4))

    for model in models:
        data_temp = result_t_temp[model]
        data_lst = []
        for _, val in data_temp.items():
            miu = np.round(np.mean(val), 2)
            sigma = np.round(np.std(val), 2)
            data_lst.append(f'{miu}±{sigma}')
        arr = np.vstack([arr, np.array(data_lst)])
    arr = arr[1:,:]

    # gradient boosting 
    data_lst = []
    for _, val in result_e.items():
        miu = np.round(np.mean(val), 2)
        sigma = np.round(np.std(val), 2)
        data_lst.append(f'{miu}±{sigma}')
    arr = np.vstack([arr, np.array(data_lst)])
    df = pd.DataFrame(np.concatenate([np.array(['LR',
                                        'SVM',
                                        'RF',
                                        'RF_GB']).reshape(-1,1), arr],
                                    axis=1),
                    columns=['Models',
                            'Accuracy',
                            'Precision',
                            'Recall',
                            'F-measure'])
    df.to_csv(os.path.join(root_path,
                        'results_table',
                        'word2vec_with_acc',
                        f'{grade}.csv'))