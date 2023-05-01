import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from class_set_gtnnwr import Model
import torch
import pandas as pd
import random

varName = ['EI','b01','b02','b03','b04','b05','b06','b07']
spatial = ['lon','lat']
temporal = ['date']

seed = 15
sample_rate = 0.3

Origin_Data = pd.read_csv("../data/Origin_clean.csv")
df_train = Origin_Data.sample(frac=0.8, random_state=seed, axis=0).reset_index(drop=True)
Origin_Data = Origin_Data.append(df_train).drop_duplicates(keep=False)
df_valid = Origin_Data.sample(frac=0.5, random_state=seed, axis=0).reset_index(drop=True)
df_test = Origin_Data.append(df_valid).drop_duplicates(keep=False).reset_index(drop=True)
ptList = df_train.loc[:,spatial+temporal].sample(frac=sample_rate, random_state=seed, axis=0).reset_index(drop=True)
batch_size = [df_valid.shape[0],df_valid.shape[0],df_test.shape[0]]

#保存路径
taskID = 12 #训练任务的id
path = '../train/train_records/test1'
if not os.path.exists(path):
    os.makedirs(path)

model = Model(varName, spatial, temporal, torch.cuda.is_available(), path)
parameters = [5e-6, 5e-6,0.3,0.3,0.1,0.2] #lr1, lr2, d1, d2, l2_p, l2_w
model.setSuperParameters(parameters)
model.loadData(ptList, batch_size, True, df_train, df_test, df_valid)

print("seed="+str(seed)+",sample_rate="+str(sample_rate))
print(parameters)

model.run(10000, earlyStop=150, limit=0.001)