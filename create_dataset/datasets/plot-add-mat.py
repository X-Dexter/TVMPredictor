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
name = "add"
shape_dimensionality=((3,3),(0,0))
device_name="dell04"

CPU_count=log_dict[device_name.lower()][name]["-1"][str(shape_dimensionality)]["count"]
GPU_count=log_dict[device_name.lower()][name]["0"][str(shape_dimensionality)]["count"]

# name = "add"
name_mat = "mat-multiply"
CPU_count=log_dict[device_name.lower()][name]["-1"][str(shape_dimensionality)]["count"]
GPU_count=log_dict[device_name.lower()][name]["0"][str(shape_dimensionality)]["count"]

# 开始画图
plt.rcParams['font.sans-serif'] = ['FangSong']                                              # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False                                                  # 解决保存图像时'-'显示为方块的问题

plt.figure(1)

index = 1
width = max(CPU_count+1, GPU_count+1)                                                       # 个数

for key,value in log_dict[device_name][name]["-1"][str(shape_dimensionality)].items():
    if key=="count":
        continue
    shape = ast.literal_eval(key)[0]
    img1 = plt.subplot(2, width, index)
    
    plot_data = [[],[]]
    mul_data = read_data(log_dict[device_name][name_mat]["-1"][str(shape_dimensionality)][key]["file_path"])
    add_data = read_data(value["file_path"])
    for i in range(len(mul_data[0])):
        plot_data[0].append(mul_data[0][i])
        plot_data[1].append(mul_data[1][i]/add_data[1][i])
    
    plt.plot(tuple(plot_data[0]),tuple(plot_data[1]),color=mycolor(index),label=str(shape))
    
    plt.legend() # 显示图例
    plt.xlabel("shape value")
    plt.ylabel("runtime(ms)")
    img1.set_title("CPU: "+name)
    # 绘制第二个曲线
    img2=plt.subplot(2, width, width)
    plt.plot(*read_data(value["file_path"]),color=mycolor(index),label=str(calc_mul(shape)))
    plt.legend() # 显示图例
    plt.xlabel("shape value")
    plt.ylabel("runtime(ms)")
    img2.set_title("CPU-summay: "+name)
    index += 1

index= CPU_count+2
for key,value in log_dict[device_name][name]["0"][str(shape_dimensionality)].items():
    if key=="count":
        continue
    shape = ast.literal_eval(key)[0]
    img1 = plt.subplot(2, width, index)
    
    plot_data = [[],[]]
    mul_data = read_data(log_dict[device_name][name_mat]["0"][str(shape_dimensionality)][key]["file_path"])
    add_data = read_data(value["file_path"])
    for i in range(len(mul_data)):
        plot_data[0].append(mul_data[0][i])
        plot_data[1].append(mul_data[1][i]/add_data[1][i])
    
    plt.plot(tuple(plot_data[0]),tuple(plot_data[1]),color=mycolor(index),label=str(shape))
     
    plt.legend() # 显示图例
    plt.xlabel("shape value")
    plt.ylabel("runtime(ms)")
    img1.set_title("GPU: "+name)
    # 绘制第二个曲线
    img2=plt.subplot(2, width, width*2)
    plt.plot(*read_data(value["file_path"]),color=mycolor(index-width),label=str(calc_mul(shape)))
    plt.legend() # 显示图例
    plt.xlabel("shape value")
    plt.ylabel("runtime(ms)")
    img2.set_title("GPU-summay: "+name)


    index += 1

plt.show()