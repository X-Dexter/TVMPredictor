import os
import json
import ast
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from create_dataset.datasets.common import *

# 读取配置文件
json_path = "create_dataset/datasets/dataset.json"

datasets = get_database_json(json_path)

model_list = ["inception_v3","mat-multiply","mobilenet","resnet-50","resnet3d-50","squeezenet_v1_1"]
device_list = ["dell03","dell04"]

# 开始画图
plt.rcParams['font.sans-serif'] = ['FangSong']                                              # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False                                                  # 解决保存图像时'-'显示为方块的问题

plt.figure(1)

for i in range(len(model_list)):
    datas_1 = None
    datas_2 = None

    for j in range(len(device_list)):
        if j==0:
            datas_1 = read_data(device_name=device_list[j],device_type=0,shape_dimensionality=((4,),(0,0)),op_name=model_list[i],json_path=json_path)
            
            img = plt.subplot(len(device_list)+1,len(model_list)+1, i+1+j*len(model_list))
            myplot(data=datas_1[0],color=mycolor(i+1),label_type="shape")
            img.set_title("GPU-"+device_list[j]+": "+model_list[i])

            img_sum = plt.subplot(len(device_list)+1,len(model_list)+1, (j+1)*len(model_list))
            myplot(data=datas_1[0],color=mycolor(i+1),label_type="size")
            img_sum.set_title("GPU summay-"+device_list[j]+": "+model_list[i])
        elif j==1:
            datas_2 = read_data(device_name=device_list[j],device_type=0,shape_dimensionality=((4,),(0,0)),op_name=model_list[i],json_path=json_path)
            
            img = plt.subplot(len(device_list)+1,len(model_list)+1, i+1+j*len(model_list))
            myplot(data=datas_2[0],color=mycolor(i+1),label_type="shape")
            img.set_title("GPU-"+device_list[j]+": "+model_list[i])

            img_sum = plt.subplot(len(device_list)+1,len(model_list)+1, (j+1)*len(model_list))
            myplot(data=datas_2[0],color=mycolor(i+1),label_type="size")
            img_sum.set_title("GPU summay-"+device_list[j]+": "+model_list[i])

        # 加速比
        speedup_datas=data_div(datas_1[0],datas_2[0])
        img = plt.subplot(len(device_list)+1,len(model_list)+1, i+1+(len(device_list)-1)*len(model_list))
        myplot(data=speedup_datas,color=mycolor(i+1),label_type="shape")
        img.set_title("GPU-"+device_list[0]+"/"+device_list[1]+": "+model_list[i])

        img_sum = plt.subplot(len(device_list)+1,len(model_list)+1, len(device_list)*len(model_list))
        myplot(data=speedup_datas,color=mycolor(i+1),label_type="size")
        img_sum.set_title("GPU summay-"+device_list[0]+"/"+device_list[1]+": "+model_list[i])

plt.show()