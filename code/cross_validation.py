import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from class_set_gtnnwr import Model
import torch
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold

varName = ['EI','b01','b02','b03','b04','b05','b06','b07']
spatial = ['lon','lat']
temporal = ['date']

seed = 10
path = '../cross_validation/records/seed'+str(seed)
sample_rate = 0.3
Origin_Data = pd.read_csv("../data/Origin_clean.csv")

if not os.path.exists(path):
        os.makedirs(path)
# create KFold object with 10 splits
kf = KFold(n_splits=10, shuffle=True, random_state=seed)
# split dataframe into 10 folds using KFold
dfs, test_r2_list = [],[]
for train_index, test_index in kf.split(Origin_Data):
    train_df = Origin_Data.iloc[train_index]
    test_df = Origin_Data.iloc[test_index]
    dfs.append((train_df, test_df))

task=0
model = Model(varName, spatial, temporal, torch.cuda.is_available(),path)
#10-fold cross validation
for df_train, df_test in dfs:
    task+=1
    ptList = df_train.loc[:,spatial+temporal].sample(frac=sample_rate, random_state=seed, axis=0).reset_index(drop=True)
    batch_size = [df_test.shape[0],df_test.shape[0]]    
    model.setSuperParameters([5e-6, 5e-6,0.3,0.3,0.1,0.2])#lr1, lr2, d1, d2, l2_p, l2_w
    model.loadData(ptList, batch_size, True, df_train, df_test)
    best_r2_test = model.crossValidation(10000, earlyStop=150, limit=0.001, task_id = task)
    test_r2_list.append(best_r2_test)

test_r2_list = np.array(test_r2_list)
average = test_r2_list.mean()

print('Average of test_r2: {}'.format(average))