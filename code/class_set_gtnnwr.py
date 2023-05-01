import os
from os.path import exists
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn import linear_model
import math
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import DisMatrix as DM
import matplotlib.pyplot as plt
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP

class MYDataset(Dataset):
    def __init__(self, np_attri, np_label, np_disMat_3D):
        self.__attribution = np_attri
        self.__label = np_label
        self.__disMat_3D = np_disMat_3D
    def __len__(self):
        return len(self.__attribution)
    def __getitem__(self,idx):
        attribute = torch.tensor(self.__attribution[idx], dtype=torch.float)
        label = torch.tensor(self.__label[idx], dtype=torch.float)
        disMat_3D = torch.tensor(self.__disMat_3D[idx], dtype=torch.float)
        return attribute, label, disMat_3D

class STWNN(nn.Module):
    def __init__(self, inSize, outSize):
        super(STWNN, self).__init__()
        self.__inSize = inSize
        self.__outSize = outSize
        self.__hidden = nn.Sequential()
        self.__dropout = 0
        
        thisSize = -1
        lastSize = inSize
        sizes = [256, 128, 64] #隐藏层层数
        for i in range(len(sizes)):
            thisSize = sizes[i]
            self.__hidden.add_module("full"+str(i), nn.Linear(lastSize, thisSize)) #全连接层
            self.__hidden.add_module("batc"+str(i), nn.BatchNorm1d(thisSize)) #标准化归一化
            self.__hidden.add_module("acti"+str(i), nn.PReLU(init=0.4)) #激活函数
            self.__hidden.add_module("drop"+str(i), nn.Dropout(self.__dropout)) #防止过拟合
            lastSize = thisSize
        self.__out = nn.Linear(lastSize, self.__outSize)

    def forward(self, x):
        x = self.__hidden(x)
        out = self.__out(x)
        return out
    
    def setDropout(self, dropout):
        self.__dropout = dropout

class STPNN(nn.Module):
    def __init__(self, inSize, outSize):
        super(STPNN, self).__init__()
        self.__inSize = inSize
        self.__outSize = outSize
        self.__hidden = nn.Sequential()
        self.__dropout = 0

        thisSize = -1
        lastSize = inSize
        sizes = [] #隐藏层层数
        for i in range(len(sizes)):
            thisSize = sizes[i]
            self.__hidden.add_module("full"+str(i), nn.Linear(lastSize, thisSize)) #全连接层
            self.__hidden.add_module("batc"+str(i), nn.BatchNorm1d(thisSize))
            self.__hidden.add_module("acti"+str(i), nn.PReLU(init=0.4)) #激活函数
            self.__hidden.add_module("drop"+str(i), nn.Dropout(self.__dropout)) #防止过拟合
            lastSize = thisSize
        self.__out = nn.Linear(lastSize, self.__outSize)

    def forward(self, x):
        axis1 = x.shape[1]
        x = x.reshape(-1, self.__inSize) #先降维成 (n·m)*3
        x = self.__hidden(x)
        out = self.__out(x)
        out = out.reshape(-1, axis1, self.__outSize)
        return out #再升维成 n*m*1
    
    def setDropout(self, dropout):
        self.__dropout = dropout

def reELU_np(x_np):
    x_np = x_np/2.71
    if x_np<=1:
        result = np.log(x_np)
    else:
        result = x_np-1
    return result
def ELU_np(y_np):
    if y_np<0:
        result = np.exp(y_np)
    else:
        result = y_np+1
    result = result*2.71
    return result

class Model:
    def __init__(self, varNameList, SpatialList, TemporalList, isGPU, savePath=''):
        self.__trainLoader = None
        self.__validLoader = None
        self.__testLoader = None
        self.__ptList = pd.DataFrame([])
        self.__varNameList = varNameList # varNameList[0]应为因变量
        self.__SpatialList = SpatialList
        self.__TemporalList = TemporalList
        self.__lossFun = nn.MSELoss()
        self.__stpnn = None #输入某点到另一个点的空间距离与时间距离，输出一个时空距离
        self.__optimizer1 = None
        self.__stwnn = None #输入某点到所有样本点的时空距离，输出改点所有自变量的时空权重
        self.__optimizer2 = None
        self.__a_means = []
        self.__a_stds = []        
        self.__spatial_means = []
        self.__spatial_stds = []        
        self.__temporal_means = []
        self.__temporal_stds = []
        self.__olrCoef = []
        self.__calculate = nn.Linear(len(varNameList), 1, bias = False)
        self.__isGPU = isGPU
        self.__savePath = savePath
        self.__lastMin = 0
        #Hyperparameters
        self.__lr1 = 0          #stpnn learn rate
        self.__lr2 = 0          #stwnn learn rate
        self.__dropout1 = 0     #stpnn drop rate
        self.__dropout2 = 0     #stwnn drop rate
        self.__l2_p = 0         #L2 regularization lambda of stpnn
        self.__l2_w = 0         #L2 regularization lambda of stwnn

    def setSuperParameters(self, specific_parameters):
        self.__lr1 = specific_parameters[0]
        self.__lr2 = specific_parameters[1]
        self.__dropout1 = specific_parameters[2]
        self.__dropout2 = specific_parameters[3]
        self.__l2_p = specific_parameters[4]
        self.__l2_w = specific_parameters[5]

    def __dataProc(self, df):
        label = np.array(df[self.__varNameList[0]])
        #label非线性变换
        transform_v = np.vectorize(reELU_np)
        label = transform_v(label)

        attribute = np.array(df[self.__varNameList[1:]])
        spatial = df[self.__SpatialList]
        temporal = df[self.__TemporalList[0]]
        temporal = pd.to_datetime(temporal).apply(lambda x: x.value)

        print("\tAttribute standardization:",end='')
        #train需要计算olrCoef、均值、标准差
        if(len(self.__olrCoef)==0):
            self.__a_means = attribute.mean(axis=0)
            self.__a_stds = attribute.std(axis=0)
            attribute = (attribute - self.__a_means) / self.__a_stds #标准化
            #求解olr系数
            model_ols = linear_model.LinearRegression().fit(attribute, label)
            self.__olrCoef = np.insert(model_ols.coef_, 0, model_ols.intercept_)
            self.__calculate.weight = nn.Parameter(torch.tensor(np.array([self.__olrCoef])).to(torch.float), requires_grad=False)
        #valid、test使用train的均值、标准差
        else:
            attribute = (attribute - self.__a_means) / self.__a_stds
        #属性值加常数列
        C = np.ones(len(label)).reshape(-1,1)
        attribute = np.hstack((C, attribute))
        return attribute, label, self.__disFun(np.array(spatial), np.array(temporal))

    def __disFun(self, spatial_np, temporal_np):
        """
        df包含 self.__SpatialList 和 self.__TemporalList 列
        """
        #样本点时空坐标
        spatial_pt = self.__ptList[self.__SpatialList]
        temporal_pt = self.__ptList[self.__TemporalList[0]]
        temporal_pt = pd.to_datetime(temporal_pt).apply(lambda x: x.value)
        #求出数据点到所有样本点的时间与空间距离
        print("\tdone.\n\tDistance calculation: ",end='')
        spatial_dismat, temporal_dismat = DM.Compute_Dismat(spatial_np, temporal_np, np.array(spatial_pt), np.array(temporal_pt))
        #求出训练集时间与空间距离的均值、标准差
        if(len(self.__spatial_means)==0):
            print("\tdone.\n\tCalculate Means and Stds:",end='', flush=True)
            self.__spatial_means, self.__spatial_stds = DM.Compute_Dismat_Statistics(spatial_dismat)
            self.__temporal_means, self.__temporal_stds = DM.Compute_Dismat_Statistics(temporal_dismat)
        #标准化
        print("\tdone.\n\tDistance standardization:",end='', flush=True)
        spatial_dismat = DM.Compute_Dis_Standerlize(spatial_dismat, self.__spatial_means, self.__spatial_stds)
        temporal_dismat = DM.Compute_Dis_Standerlize(temporal_dismat, self.__temporal_means, self.__temporal_stds)
        #3维化：len(data)*len(sample)*3
        print("\tdone.\n\tDistance To3D: ",end='', flush=True)
        dismat_3d = DM.To3D_Trans(spatial_dismat, temporal_dismat)
        print("\tdone.")
        return dismat_3d

    def __resetModel(self):
        self.__a_means = []
        self.__a_stds = []
        self.__spatial_means = []
        self.__spatial_stds = []        
        self.__temporal_means = []
        self.__temporal_stds = []
        self.__olrCoef = []

    def loadData(self, ptList, batch_size, drop_last, df_train, df_test, df_valid=None):
        #每次loadData之后需要重置模型参数
        self.__resetModel()
        start_time = datetime.datetime.now()
        """
        df应为列名包含：varNameList, SpatialList, TemporalList 的dataframe
        """
        print("Data Processing:")
        self.__ptList = ptList
        l = [2.50412667e-05, 3.43652267e-05]
        self.__stpnn = STPNN(2, 1) #输入某点到另一个点的空间距离与时间距离，输出一个时空距离
        self.__optimizer1 = torch.optim.SGD(self.__stpnn.parameters(), lr=l[0], momentum=0.9)
        # self.__optimizer1 = torch.optim.Adam(self.__stpnn.parameters(), lr=l[0])
        self.__stwnn = STWNN(len(ptList), len(self.__varNameList)) #输入某点到所有样本点的时空距离，输出改点所有自变量的时空权重
        self.__optimizer2 = torch.optim.SGD(self.__stwnn.parameters(), lr=l[1], momentum=0.9)
        # self.__optimizer2 = torch.optim.Adam(self.__stwnn.parameters(), lr=l[1])
        #数据预处理
        print(" train data:")
        np_train_attri, np_train_label, np_train_3DdisMat = self.__dataProc(df_train)
        print(" test data:")
        np_test_attri, np_test_label, np_test_3DdisMat = self.__dataProc(df_test)
        if df_valid is not None:
            print(" valid data:")
            np_valid_attri, np_valid_label, np_valid_3DdisMat = self.__dataProc(df_valid)
            np.save(self.__savePath+'/a_means', self.__a_means)
            np.save(self.__savePath+'/a_stds', self.__a_stds)
            np.save(self.__savePath+'/olrCoef', self.__olrCoef)
            np.save(self.__savePath+'/s_means', self.__spatial_means)
            np.save(self.__savePath+'/s_stds', self.__spatial_stds)
            np.save(self.__savePath+'/t_means', self.__temporal_means)
            np.save(self.__savePath+'/t_stds', self.__temporal_stds)
        #DataSet
        trainSet = MYDataset(np_train_attri, np_train_label, np_train_3DdisMat)
        testSet = MYDataset(np_test_attri, np_test_label, np_test_3DdisMat)
        if df_valid is not None:
            validSet = MYDataset(np_valid_attri, np_valid_label, np_valid_3DdisMat)
        #DataLoader
        self.__trainLoader = DataLoader(trainSet, batch_size=batch_size[0], shuffle=True, num_workers=2, drop_last=drop_last)
        self.__testLoader = DataLoader(testSet, batch_size=batch_size[1], shuffle=True, num_workers=2, drop_last=drop_last)
        if df_valid is not None:
            self.__validLoader = DataLoader(validSet, batch_size=batch_size[2], shuffle=True, num_workers=2, drop_last=drop_last)

        cur_time = datetime.datetime.now()
        min = (cur_time-start_time).seconds//60
        sec = (cur_time-start_time).seconds%60
        print("Done!\tcost time: {}:{:0>2}\n".format(min, sec))

    def __l2_regularization(self, model, l2_lambda):
        l2_reg = nn.MSELoss()
        l2_loss = 0
        for param in model.parameters():
            l2_loss += l2_lambda * torch.norm(param, 2)
        return l2_loss

    def __train(self):
        self.__stpnn.train()
        self.__stwnn.train()
        labelList = np.array([])
        predList = np.array([])
        trainLoss = 0
        for attribute, label, disMat_3D in self.__trainLoader: #根据batch_size来选取
            label.unsqueeze(1) #升维，即label由行向量变列向量
            if self.__isGPU:
                attribute, label, disMat_3D = attribute.cuda(), label.cuda(), disMat_3D.cuda()
            self.__optimizer2.zero_grad() #清空梯度
            self.__optimizer1.zero_grad() #清空梯度
            ST_distance = self.__stpnn(disMat_3D).squeeze(-1)
            w = self.__stwnn(ST_distance).squeeze(-1) #计算时空权重
            wx = w.mul(attribute) #时空权重乘以自变量
            pred = self.__calculate(wx).squeeze(-1) #最后乘以线性回归系数(beta)得到因变量
            
            loss = self.__lossFun(pred, label)
            loss += self.__l2_regularization(self.__stpnn, self.__l2_p)
            loss += self.__l2_regularization(self.__stwnn, self.__l2_w)
            loss.backward() #反向传播
            self.__optimizer2.step() #梯度下降
            # self.__optimizer1.step() #梯度下降

            label = label.squeeze(-1).cpu().numpy()
            pred = pred.squeeze(-1).cpu().detach().numpy()
            labelList = np.append(labelList, label)
            predList = np.append(predList, pred)
            trainLoss += loss.item() * attribute.shape[0]
        reTrans_v = np.vectorize(ELU_np)
        r2 = r2_score(reTrans_v(labelList), reTrans_v(predList))
        # r2 = r2_score(labelList, predList)
        trainLoss = trainLoss / len(self.__trainLoader.dataset)
        return r2, trainLoss

    def __valid(self):
        self.__stpnn.eval()
        self.__stwnn.eval()
        with torch.no_grad():
            labelList = np.array([])
            predList = np.array([])
            for attribute, label, disMat_3D in self.__validLoader:
                label.unsqueeze(1) #升维，即label由行向量变列向量
                if self.__isGPU:
                    attribute, label, disMat_3D = attribute.cuda(), label.cuda(), disMat_3D.cuda()
                ST_distance = self.__stpnn(disMat_3D).squeeze(-1)
                w = self.__stwnn(ST_distance).squeeze(-1) #计算时空权重
                wx = w.mul(attribute) #时空权重乘以自变量
                pred = self.__calculate(wx).squeeze(-1) #最后乘以线性回归系数(beta)得到因变量

                label = label.squeeze(-1).cpu().numpy()
                pred = pred.squeeze(-1).cpu().detach().numpy()
                labelList = np.append(labelList, label)
                predList = np.append(predList, pred)
            reTrans_v = np.vectorize(ELU_np)
            r2 = r2_score(reTrans_v(labelList), reTrans_v(predList))
            # r2 = r2_score(labelList, predList)
        return r2
    
    def __test(self):
        self.__stpnn.eval()
        self.__stwnn.eval()
        with torch.no_grad():
            labelList = np.array([])
            predList = np.array([])
            for attribute, label, disMat_3D in self.__testLoader:
                label.unsqueeze(1) #升维，即label由行向量变列向量
                if self.__isGPU:
                    attribute, label, disMat_3D = attribute.cuda(), label.cuda(), disMat_3D.cuda()
                ST_distance = self.__stpnn(disMat_3D).squeeze(-1)
                w = self.__stwnn(ST_distance).squeeze(-1) #计算时空权重
                wx = w.mul(attribute) #时空权重乘以自变量
                pred = self.__calculate(wx).squeeze(-1) #最后乘以线性回归系数(beta)得到因变量

                label = label.squeeze(-1).cpu().numpy()
                pred = pred.squeeze(-1).cpu().detach().numpy()
                labelList = np.append(labelList, label)
                predList = np.append(predList, pred)
            reTrans_v = np.vectorize(ELU_np)
            r2 = r2_score(reTrans_v(labelList), reTrans_v(predList))
            # r2 = r2_score(labelList, predList)
        return r2

    def __drawPic(self, epoch_list, r2_train_list, r2_valid_list, best_r2_valid_list, r2_test_list):
        plt.figure(figsize=(20,10), dpi=200)
        if epoch_list[-1]<10 :
            plt.xticks([x for x in range(0, epoch_list[-1])])
        else:
            plt.xticks([x for x in range(0, epoch_list[-1], epoch_list[-1]//10)])
        plt.xlabel("epoch", fontsize=20)
        plt.ylim(0,1)
        plt.ylabel("R2", fontsize=20)
        plt.title("R2 - epoch (gtnnwr)", fontsize=36)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.plot(epoch_list, r2_train_list, label='r2_train')
        plt.plot(epoch_list, r2_valid_list, label='r2_valid')
        plt.plot(epoch_list, best_r2_valid_list, label='best_r2_valid')
        plt.plot(epoch_list, r2_test_list, label='r2_test')
        plt.legend(fontsize=20)
        plt.savefig(self.__savePath + '/pic.png')
        plt.close()

    def __savePic(self, epoch_list, r2_train_list, best_r2_test_list, task_id):
        plt.figure(figsize=(20,10), dpi=200)
        if epoch_list[-1]<10 :
            plt.xticks([x for x in range(0, epoch_list[-1])])
        else:
            plt.xticks([x for x in range(0, epoch_list[-1], epoch_list[-1]//10)])
        plt.xlabel("epoch", fontsize=20)
        plt.ylim(0,1)
        plt.ylabel("R2", fontsize=20)
        plt.title("R2 - epoch (crossValidation)", fontsize=36)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.plot(epoch_list, r2_train_list, label='r2_train')
        plt.plot(epoch_list, best_r2_test_list, label='best_r2_test')
        plt.legend(fontsize=20)
        plt.savefig(self.__savePath + '/pic'+str(task_id)+'.png')
        plt.close()

    def run(self, maxTimes = 101, earlyStop = -1, limit = 0):
        self.__lastMin = 0
        if self.__isGPU:
            self.__stwnn = self.__stwnn.cuda()
            self.__stpnn = self.__stpnn.cuda()
            self.__calculate = self.__calculate.cuda()
        #apply the hyperparameters
        self.__optimizer1.param_groups[0]['lr'] = self.__lr1
        self.__optimizer2.param_groups[0]['lr'] = self.__lr2
        self.__stpnn.setDropout(self.__dropout1)
        self.__stwnn.setDropout(self.__dropout2)

        best_r2_valid = -1 #记录valid的r2最高值
        test = -1 #记录valid的r2最高时，test的r2
        epoch_list,r2_train_list,r2_valid_list,r2_test_list,train_loss,best_r2_valid_list = [],[],[],[],[],[]
        start_time = datetime.datetime.now()
        for epoch in range(0, maxTimes):
            r2_train, Loss = self.__train()
            r2_valid = self.__valid()
            r2_test = self.__test()
            self.__lastMin = 0 if r2_valid - best_r2_valid > limit else self.__lastMin + 1 #根据容忍误差判断
            if r2_valid > best_r2_valid:
                best_r2_valid = r2_valid
                test = r2_test
                torch.save(self.__stpnn, self.__savePath + "/stpnn.pkl")
                torch.save(self.__stwnn, self.__savePath + "/stwnn.pkl")
            epoch_list.append(epoch)

            r2_train_list.append(r2_train)
            r2_valid_list.append(r2_valid)
            best_r2_valid_list.append(best_r2_valid)
            r2_test_list.append(r2_test)
            train_loss.append(Loss)
            if self.__lastMin > earlyStop and earlyStop != -1:
                break
            cur_time = datetime.datetime.now()
            hour = (cur_time-start_time).seconds//3600
            min = (cur_time-start_time).seconds//60%60
            sec = (cur_time-start_time).seconds%60
            if epoch % 10 == 0:
                print('----------------------------------------------------------------')
                print('Epoch: {} \tLr: {}, {}'.format(epoch, self.__optimizer1.param_groups[0]['lr'], self.__optimizer2.param_groups[0]['lr']))
                print('Train_Loss: {:.2f} \tCost: {}:{:0>2}:{:0>2}'.format(Loss, hour, min, sec))
                print('R2_train: {:.6f} \tR2_valid: {:.6f} \tR2_test: {:.6f}'.format(r2_train,r2_valid,r2_test))
                print('Best_R2_valid:   {:.6f} \tR2_test(ofBestValid):   {:.6f}'.format(best_r2_valid,test))
                self.__drawPic(epoch_list, r2_train_list, r2_valid_list, best_r2_valid_list, r2_test_list)
        self.__drawPic(epoch_list, r2_train_list, r2_valid_list, best_r2_valid_list, r2_test_list)
        print("END!")
    
    def crossValidation(self, maxTimes = 101, earlyStop = -1, limit = 0, task_id = 0):
        self.__lastMin = 0
        if self.__isGPU:
            self.__stwnn = self.__stwnn.cuda()
            self.__stpnn = self.__stpnn.cuda()
            self.__calculate = self.__calculate.cuda()
        #apply the hyperparameters
        self.__optimizer1.param_groups[0]['lr'] = self.__lr1
        self.__optimizer2.param_groups[0]['lr'] = self.__lr2
        self.__stpnn.setDropout(self.__dropout1)
        self.__stwnn.setDropout(self.__dropout2)

        best_r2_test = -1 #记录test的r2最高值
        train = -1 #记录test的r2最高时，train的r2
        epoch_list,r2_train_list,best_r2_test_list = [],[],[]
        start_time = datetime.datetime.now()
        for epoch in range(0, maxTimes):
            r2_train, loss = self.__train()
            r2_test = self.__test()
            self.__lastMin = 0 if r2_test - best_r2_test > limit else self.__lastMin + 1 #根据容忍误差判断
            if r2_test > best_r2_test:
                best_r2_test, train = r2_test, r2_train
            epoch_list.append(epoch)
            r2_train_list.append(r2_train)
            best_r2_test_list.append(best_r2_test)
            if self.__lastMin > earlyStop and earlyStop != -1:
                break
        cur_time = datetime.datetime.now()
        hour, min, sec = (cur_time-start_time).seconds//3600, (cur_time-start_time).seconds//60%60, (cur_time-start_time).seconds%60
        print('Task id: {}\tCost: {}:{:0>2}:{:0>2}'.format(task_id, hour, min, sec))
        print('R2_train(ofBestTest): {:.6f} \tBest_R2_test:: {:.6f}'.format(train,best_r2_test))
        print('----------------------------------------------------------------')
        self.__savePic(epoch_list, r2_train_list, best_r2_test_list, task_id)
        return best_r2_test