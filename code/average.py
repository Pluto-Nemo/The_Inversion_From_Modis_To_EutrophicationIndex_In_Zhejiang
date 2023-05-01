import numpy as np
import math
from osgeo import gdal
from osgeo.gdalconst import GDT_Float32
import os
import sys
os.environ['PROJ_LIB'] = os.path.dirname(sys.argv[0])
from numba import njit


@njit()#先不加权，尝试正态分布去异常
def Calculate(month_data, coverage_list, rows, cols, nodata):
    result_data = np.full((rows, cols), nodata, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            element = np.empty(0, dtype=np.float32)
            coverage = np.empty(0, dtype=np.float32)
            for day in range(month_data.shape[0]):
                if month_data[day, i, j] != nodata:
                    element = np.append(element, month_data[day, i, j])
                    coverage = np.append(coverage, coverage_list[day])
            # if len(element) != 0:
            #     #正态分布去异常
            #     mean, std = np.mean(element), np.std(element)
            #     preprocess_list = [item for item in element if item <= mean + std]
            #     preprocess_list = np.array(preprocess_list)
            #     result_data[i, j] = sum(preprocess_list) / len(preprocess_list)
            if len(element) != 0:
                #正态分布去异常
                wx = np.empty(0, dtype=np.float32)
                w = np.empty(0, dtype=np.float32)
                mean, std = np.mean(element), np.std(element)
                for k in range(len(element)):
                    if element[k] <= mean + std:
                        #根据覆盖率加权平均
                        wx = np.append(wx, element[k]*coverage[k])
                        w = np.append(w, coverage[k])
                result_data[i, j] = sum(wx) / sum(w)
    return result_data


# E_2015_01_01_0.tif 海域_年份_月份_日期_波段.tif
def Average(nodata, dataPath, resultPath, start_year, end_year):
    cols, rows, geoTransform, projection = 0, 0, None, None
    for year in [str(i) for i in range(start_year, end_year + 1)]:
        for month in [str(i).zfill(2) for i in range(1, 13)]:
            month_data = list()
            coverage_list = list()
            for day in [str(i).zfill(2) for i in range(1, 32)]:
                tif_name = 'EI_' + year + '_' + month + '_' + day + '.tif'
                if not os.path.exists(dataPath+'/'+str(start_year)+'/'+ tif_name):
                    continue
                dataset = gdal.Open(
                    dataPath+'/'+str(start_year)+'/'+ tif_name, gdal.GA_ReadOnly)
                geoTransform = dataset.GetGeoTransform()
                projection = dataset.GetProjection()
                cols = dataset.RasterXSize
                rows = dataset.RasterYSize
                dataAsArray = dataset.ReadAsArray(0, 0, cols, rows)

                coverage = np.count_nonzero(dataAsArray!=-9999) / 252984
                coverage_list.append(coverage)
                month_data.append(dataAsArray)
            if len(month_data) == 0:
                print(year + '_' + month + ' does not exist!')
                continue
            month_data = np.array(month_data)
            coverage_list = np.array(coverage_list)
            coverage_list = coverage_list.astype(np.float32)
            result_data = Calculate(month_data, coverage_list, rows, cols, nodata)
            # 以下是输出图部分
            if not os.path.exists(resultPath+'/'+str(start_year)):
                os.mkdir(resultPath+'/'+str(start_year))
            result_format = "GTiff"
            driver = gdal.GetDriverByName(result_format)
            result_ds = driver.Create(
                resultPath+'/'+str(start_year)+ '/EI_' + year + '_' + month + '.tif', cols, rows, 1, GDT_Float32)
            result_ds.SetGeoTransform(geoTransform)
            result_ds.SetProjection(projection)
            result_ds.GetRasterBand(1).WriteArray(result_data)
            result_ds.GetRasterBand(1).SetNoDataValue(-9999)
            result_ds.FlushCache()
            result_ds = None
            print(year + '_' + month + '.tif'+' done!')

Average(-9999, '../result/EI_sgd10', '../result/average_sgd10', 2019, 2019)