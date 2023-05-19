"""
Compute the Cronbach's Alpha of CDI

@author: Anmin(BNU)
"""
import numpy as np
import pandas as pd
import os
import pingouin as pg

root_path = '/Users/anmin/Documents/NLP'
reverse_item = [2, 5, 7, 8,
                10, 11, 13, 15, 16,
                18, 21, 24, 25]

map_dic = {1 : 3,
        3: 1}

df = pd.read_excel(os.path.join(root_path,
                                'cdi_total.xlsx'),
                    engine='openpyxl')
df = df.dropna()
arr = np.array(df)[:, 3:]

# mapping
for c_num in reverse_item:
    c_num -= 1
    col_temp = arr[:, c_num]
    for i, val in enumerate(col_temp):
        if val == 1:
            col_temp[i] = 3
        elif val == 3:
            col_temp[i] = 1
    arr[:, c_num] = col_temp

# Cronbach's alpha
print(pg.cronbach_alpha(pd.DataFrame(arr)))

# (0.881963068509349, array([0.877, 0.887]))
