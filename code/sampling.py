import pandas as pd
import numpy as np
import os
from osgeo import gdal

source = pd.read_csv("../data/source.csv")
source = source.sort_values(by='date').reset_index (drop=True)

date = source['date'].drop_duplicates ().reset_index (drop=True)
date_np = np.array(date)

#source添加波段列，初始值为-9999
for i in range(1,8):
    source['b0'+str(i)] = [-9999 for i in range(source.shape[0])]

NoData = 0 #波段的NoData值
for date in date_np:
    name = date+'.tif'
    name = name.replace("-", "_")
    if not os.path.exists('../../SRTP/Data/GEE_IMAGE_Nocloud_Qualified/'+name):
        print(name+' does not exist!')
        while source['date'][index]==date:
            source.drop(axis=0, index=index, inplace=True)
            index = (index+1) % source.shape[0] #防止越界
        continue
        # os._exit(0)
    #打开当前天的tif图像
    tif = gdal.Open('../../SRTP/Data/GEE_IMAGE_Nocloud_Qualified/'+name)
    img_bands = tif.RasterCount
    img_height, img_width = tif.RasterYSize, tif.RasterXSize
    im_geotrans = tif.GetGeoTransform()
    img_data = tif.ReadAsArray(0,0,img_width,img_height)

    index_list = source[source['date'] == date].index.tolist()
    #遍历当天的source条目
    for index in index_list :
        p_lon, p_lat = source['lon'][index], source['lat'][index]
        i_lon, i_lat = im_geotrans[0], im_geotrans[3] #图像左上角经纬度
        dx, dy = im_geotrans[1], im_geotrans[5] #像元的尺寸
        #点经纬度转波段数据矩阵i、j下标
        i = int((p_lat - i_lat) // dy)
        j = int((p_lon - i_lon) // dx)
        #判断该点所处像元值是否为NoData
        if np.all(img_data[:,i,j] == NoData):
            source.drop(axis=0, index=index, inplace=True)
            continue
        #点采样
        source.loc[index,'b01':'b07']= img_data[:,i,j]

source = source.reset_index (drop=True)
if -9999 in source.loc[:,'b01':'b07'].values:
    print('WARNING: still having NoData in the point_data after sampling!')
else:
    print('Done!')
source_clean = source[source['EI']<48.0]
source_clean.to_csv('../data/sampling_c_q.csv', index=False, sep=',')