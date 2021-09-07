import os
import json
import ast
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from create_dataset.datasets.read_data import read_data
from create_dataset.datasets.mycolor import mycolor

def calc_mul(shape):
    result = 1
    for s in shape:
        if s!=-1:
            result*=s
    
    return result

# 读取配置文件
json_path = "create_dataset/datasets/dataset.json"
if not os.path.exists(json_path):
    print("loss dataset json config")

log_dict = {}
with open(json_path,'r') as f:
    log_dict = json.load(f)

# name = "add"
name = "conv2d"
shape_dimensionality=((4,4),(0,0))
device_name="dell04"

CPU_count=log_dict[device_name.lower()][name]["-1"][str(shape_dimensionality)]["count"]
GPU_count=log_dict[device_name.lower()][name]["0"][str(shape_dimensionality)]["count"]

# 开始画图
plt.rcParams['font.sans-serif'] = ['FangSong']                                              # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False                                                  # 解决保存图像时'-'显示为方块的问题

plt.figure(1)

index = 1
width = max(CPU_count+1, GPU_count+1)                                                       # 个数

for key in log_dict[device_name][name]["-1"][str(shape_dimensionality)].keys():
    if key=="count":
        continue
    shape = ast.literal_eval(key)[0]
    CPU_data = read_data(log_dict[device_name][name]["-1"][str(shape_dimensionality)][key]["file_path"])
    GPU_data = read_data(log_dict[device_name][name]["0"][str(shape_dimensionality)][key]["file_path"])
    
    # 绘制CPU
    img1 = plt.subplot(3, width, index)
    plt.plot(CPU_data[0],CPU_data[1],color=mycolor(index),label=str(shape))
    plt.legend() # 显示图例
    plt.xlabel("shape value")
    plt.ylabel("runtime(ms)")
    img1.set_title("CPU: "+name)

    # 绘制总图
    img2_=plt.subplot(3, width, width)
    plt.plot(CPU_data[0],CPU_data[1],color=mycolor(index),label=str(calc_mul(shape)))
    plt.legend() # 显示图例
    plt.xlabel("shape value")
    plt.ylabel("runtime(ms)")
    img2_.set_title("CPU-summay: "+name)

    # 绘制GPU
    img2=plt.subplot(3, width, width+index)
    plt.plot(GPU_data[0],GPU_data[1],color=mycolor(index),label=str(calc_mul(shape)))
    plt.legend() # 显示图例
    plt.xlabel("shape value")
    plt.ylabel("runtime(ms)")
    img2.set_title("GPU: "+name)

    img2_=plt.subplot(3, width, 2*width)
    plt.plot(GPU_data[0],GPU_data[1],color=mycolor(index),label=str(calc_mul(shape)))
    plt.legend() # 显示图例
    plt.xlabel("shape value")
    plt.ylabel("runtime(ms)")
    img2_.set_title("GPU: "+name)

    datas=[]
    for i in range(len(CPU_data[1])):
        datas.append(CPU_data[1][i]/GPU_data[1][i])

    # 绘制CPU/GPU
    img3=plt.subplot(3, width, 2*width+index)
    plt.plot(GPU_data[0],datas,color=mycolor(index),label=str(calc_mul(shape)))
    plt.legend() # 显示图例
    plt.xlabel("shape value")
    plt.ylabel("runtime(ms)")
    img3.set_title("CPU/GPU: "+name)

    # 绘制总图
    img3_=plt.subplot(3, width, 3*width)
    plt.plot(GPU_data[0],datas,color=mycolor(index),label=str(calc_mul(shape)))
    plt.legend() # 显示图例
    plt.xlabel("shape value")
    plt.ylabel("runtime(ms)")
    img3_.set_title("CPU/GPU: "+name)

    index += 1

plt.show()