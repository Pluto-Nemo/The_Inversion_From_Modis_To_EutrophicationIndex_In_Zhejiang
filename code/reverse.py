import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
import sys
os.environ['PROJ_LIB'] = os.path.dirname(sys.argv[0])
from osgeo import gdal
import pandas as pd
import numpy as np
import DisMatrix as DM
import torch
import torch.nn as nn
import class_set_gtnnwr
from torch.utils.data import Dataset, DataLoader
from numba import njit
import urllib3
urllib3.disable_warnings()

gdal.AllRegister()
year = ['2019']
month = [str(i).zfill(2) for i in range(1, 13)]
# month = ["07","08","09","10","11","12"]
day = [str(i).zfill(2) for i in range(1, 32)]

seed = 15
sample_rate = 0.3

task = 'sgd10'
path = '../reverse/reverse_records/'+task
if not os.path.exists(path):
    os.makedirs(path)

#resize tif图像数据为[h * w * bands]，并输出其经纬度矩阵
@njit()
def proc_tif(img_height, img_width, img_data, img_bands, im_geotrans, nodata):
    lonlat = np.zeros((img_height,img_width,2))
    img_resize = np.zeros((img_height,img_width,img_bands))
    for i in range(0,img_height):
        for j in range(0,img_width):
            if np.all(img_data[:,i,j] == nodata):
                img_resize[i][j][:] = np.nan
                lonlat[i][j][:] = np.nan
            else:
                lonlat[i][j][0] = im_geotrans[0] + j*im_geotrans[1]
                lonlat[i][j][1] = im_geotrans[3] + i*im_geotrans[5]
                img_resize[i][j][:] = img_data[:,i,j]
    return img_resize, lonlat

def ELU_np(y_np):
    if y_np<0:
        result = np.exp(y_np)
    else:
        result = y_np+1
    result = result*2.71
    return result

@njit()
def EI_product(img_height, img_width, lonlat, EI):
    EI_img = np.zeros((img_height,img_width))
    index = 0
    for i in range(0,img_height):
        for j in range(0,img_width):
            if np.isnan(lonlat[i,j,0]):
                EI_img[i][j] = -9999
            else:
                EI_img[i][j] = EI[index]
                index+=1
    return EI_img

Origin_Data = pd.read_csv("../data/Origin_clean.csv")
df_train = Origin_Data.sample(frac=0.8, random_state=seed, axis=0).reset_index(drop=True)
ptList = df_train.loc[:,['lon','lat','date']].sample(frac=sample_rate, random_state=seed, axis=0).reset_index(drop=True)
batch_size = df_train.shape[0]//8

spatial_pt = ptList[['lon','lat']]
temporal_pt = ptList['date']
temporal_pt = pd.to_datetime(temporal_pt).apply(lambda x: x.value)
spatial_pt = np.array(spatial_pt)
temporal_pt = np.array(temporal_pt)

for y in year:
    for m in month:
        for d in day:
            date = y+'-'+m+'-'+d
            tif_name = date.replace("-", "_")
            if not os.path.exists('../img/'+y+'/'+tif_name+'.tif'):
                print('skip '+date)
                continue
            tif = gdal.Open('../img/'+y+'/'+tif_name+'.tif')

            img_bands = tif.RasterCount
            img_height, img_width = tif.RasterYSize, tif.RasterXSize # 获取影像的宽高
            im_geotrans = tif.GetGeoTransform()  #仿射矩阵，左上角像素的大地坐标和像素分辨率
            im_proj = tif.GetProjection() #地图投影信息，字符串表示
            img_data = tif.ReadAsArray(0,0,img_width,img_height)

            img_data=img_data.astype(np.float32)

            #构造img_resize(h, w, b)以及它的经纬度矩阵
            img_resize, lonlat = proc_tif(img_height, img_width, img_data, img_bands, im_geotrans, 0)

            attribute = img_resize[~(np.isnan(img_resize))]
            spatial = lonlat
            spatial = spatial[~(np.isnan(spatial))]

            #判断tif是不是全为NoData
            if(spatial.shape[0] == 0):
                print(date+' is empty!')
                continue

            #tif数据点
            attribute = attribute.reshape(-1,img_bands)
            spatial = spatial.reshape(-1,2)
            temporal = pd.Series([date] * spatial.shape[0])
            temporal = pd.to_datetime(temporal).apply(lambda x: x.value)
            temporal = np.array(temporal)

            #求解时空距离
            spatial_dismat, temporal_dismat = DM.Compute_Dismat(spatial, temporal, spatial_pt, temporal_pt)

            #读入标准化的均值、标准差
            a_means = np.load(path+'/a_means.npy')
            s_means = np.load(path+'/s_means.npy')
            t_means = np.load(path+'/t_means.npy')
            a_stds = np.load(path+'/a_stds.npy')
            s_stds = np.load(path+'/s_stds.npy')
            t_stds = np.load(path+'/t_stds.npy')

            #标准化
            a_stand = (attribute - a_means) / a_stds
            s_stand = DM.Compute_Dis_Standerlize(spatial_dismat, s_means, s_stds)
            t_stand = DM.Compute_Dis_Standerlize(temporal_dismat, t_means, t_stds)

            #3维化
            var = DM.To3D_Trans(s_stand, t_stand)

            #构造dataset与dataloader
            class MYDataset(Dataset):
                def __init__(self, np_attri, np_var):
                    self.__attribution = np_attri
                    self.__var = np_var
                def __len__(self):
                    return len(self.__attribution)
                def __getitem__(self,idx):
                    attribute = torch.tensor(self.__attribution[idx], dtype=torch.float)
                    var = torch.tensor(self.__var[idx], dtype=torch.float)
                    return attribute, var

            #属性值加常数列
            C = np.ones(a_stand.shape[0]).reshape(-1,1)
            a_stand = np.hstack((C, a_stand))

            dataSet = MYDataset(a_stand, var)
            dataLoader = DataLoader(dataSet, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

            stpnn = torch.load(path+'/stpnn.pkl', map_location=torch.device('cuda'))
            stwnn = torch.load(path+'/stwnn.pkl', map_location=torch.device('cuda'))
            calculate = nn.Linear(8, 1, bias = False)
            olrCoef = np.load(path+'/olrCoef.npy')
            calculate.weight = nn.Parameter(torch.tensor(olrCoef).to(torch.float), requires_grad=False)
            calculate = calculate.cuda()

            stpnn.eval()
            stwnn.eval()
            EI = np.array([])
            with torch.no_grad():
                for attribute, var in dataLoader:
                    attribute, var = attribute.cuda(), var.cuda()
                    ST_distance = stpnn(var).squeeze(-1)
                    w = stwnn(ST_distance).squeeze(-1)
                    wx = w.mul(attribute)
                    pred = calculate(wx)
                    pred = pred.cpu().detach().numpy()

                    reTrans_v = np.vectorize(ELU_np)
                    pred = reTrans_v(pred)
                    EI = np.append(EI, pred)
            
            EI_img = EI_product(img_height, img_width, lonlat, EI)
            
            driver = gdal.GetDriverByName("GTiff")
            out_tif = driver.Create('../result/EI_'+task+'/'+y+'/EI_'+tif_name+'.tif', img_width, img_height, 1, gdal.GDT_Float32)
            out_tif.SetGeoTransform(im_geotrans)
            out_tif.SetProjection(im_proj)
            out_tif.GetRasterBand(1).WriteArray(EI_img)
            out_tif.GetRasterBand(1).SetNoDataValue(-9999)
            out_tif.FlushCache()
            out_tif = None
            print(tif_name+'.tif'+' done!')
