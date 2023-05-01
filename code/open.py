import numpy as np
import cv2
from osgeo import gdal
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
import sys
os.environ['PROJ_LIB'] = os.path.dirname(sys.argv[0])

def opening_operation(data, size):
    # Define the kernel for opening
    kernel = kernel = np.ones((size, size), np.uint8)
    # Apply opening on the input image using the kernel
    opening = cv2.morphologyEx(np.uint8(data), cv2.MORPH_OPEN, kernel)
    return opening


kernel_size = [7,9,11]
year ='2018'
month = [str(i).zfill(2) for i in range(1, 13)]
day = [str(i).zfill(2) for i in range(1, 32)]

for m in month:
    for d in day:
        name = "EI_"+year+'_'+m+'_'+d
        open_path = "../result/EI_sgd10/"+year+'/'+name+'.tif'
        print(open_path)
        if not os.path.exists(open_path):
                print(name+" dose not exist!")
                continue

        tif = gdal.Open(open_path)
        img_height, img_width = tif.RasterYSize, tif.RasterXSize # 获取影像的宽高
        im_geotrans = tif.GetGeoTransform()  #仿射矩阵，左上角像素的大地坐标和像素分辨率
        im_proj = tif.GetProjection() #地图投影信息，字符串表示
        img_data = tif.ReadAsArray(0,0,img_width,img_height)
        img_data=img_data.astype(np.float32)

        mask = img_data != -9999            #图像转黑白

        for size in kernel_size:
            re_mask = opening_operation(mask,size)   #开运算黑白图像

            img_data_opened = np.where(re_mask,img_data,-9999)

            driver = gdal.GetDriverByName("GTiff")
            out_tif = driver.Create("../result"+"/opened_sgd10/"+year+'/'+str(size)+'/'+name+"_opened"+str(size)+".tif", img_width, img_height, 1, gdal.GDT_Float32)
            out_tif.SetGeoTransform(im_geotrans)
            out_tif.SetProjection(im_proj)
            out_tif.GetRasterBand(1).WriteArray(img_data_opened)
            out_tif.GetRasterBand(1).SetNoDataValue(-9999)
            out_tif.FlushCache()
            out_tif = None