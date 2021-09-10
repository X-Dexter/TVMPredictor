import os
import json
import ast
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

database_json = None

def get_database_json(json_path="create_dataset/datasets/dataset.json"):
    global database_json
    if not database_json:
        # 未初始化则尝试从磁盘读取配置文件
        if not os.path.exists(json_path):
            print("loss dataset json config")
            return None

        database_json = {}
        with open(json_path,'r') as f:
            database_json = json.load(f)

    return database_json

def mycolor(index)->str:
    colors = ['red','green','blue','c','m','y','k']

    if index<=len(colors) and index>0:
        return colors[index-1]
    else:
        return 'w'

def read_data_from_path(file_path) ->tuple:
    '''
    read data from file-path that store the datas, return tuple(xs,ys)
    '''
    tmp = []

    with open(file_path,"r") as f:
        line = f.readline()
        while line is not None and len(line)>0 :
            tmp.append(line.split(","))
            line=f.readline()

    xs=[]
    ys=[]

    for data in tmp:
        xs.append(int(data[0]))
        ys.append(float(data[1]))
    return (tuple(xs),tuple(ys))

def read_data(device_name,device_type,shape_dimensionality,op_name, json_path="create_dataset/datasets/dataset.json") ->tuple:
    '''
    read data from datasets, return ((xs,ys,shape,shape_changing,time), ... )
    '''
    # 加载数据库配置文件
    global database_json
    if not database_json:
        # 未初始化则尝试从磁盘读取配置文件
        if not os.path.exists(json_path):
            print("loss dataset json config")
            return None

        database_json = {}
        with open(json_path,'r') as f:
            database_json = json.load(f)

    if device_name.lower() not in database_json.keys() or op_name not in database_json[device_name.lower()].keys() or str(device_type) not in database_json[device_name.lower()][op_name].keys() or str(shape_dimensionality) not in database_json[device_name.lower()][op_name][str(device_type)].keys() or len(database_json[device_name.lower()][op_name][str(device_type)][str(shape_dimensionality)].keys())<=0:                            
        return None

    datas = []
    for shape_str,value_dict in database_json[device_name.lower()][op_name][str(device_type)][str(shape_dimensionality)].items():
        if shape_str=="count":
            continue
        file_path = value_dict["file_path"]
        shape = ast.literal_eval(value_dict["shapes"])
        shape_changing = ast.literal_eval(value_dict["changed_shape"])
        time = value_dict["time"]

        tmp = []
        with open(file_path,"r") as f:
            line = f.readline()
            while line is not None and len(line)>0 :
                tmp.append(line.split(","))
                line=f.readline()

        xs=[]
        ys=[]
        for data in tmp:
            xs.append(int(data[0]))
            ys.append(float(data[1]))

        datas.append((tuple(xs),tuple(ys),shape,shape_changing,time))

    return tuple(datas)

def myplot(data,color,label_type="shape",show_legend=True, show_xy_labels=True):
    if label_type=="shape":
        plt.plot(data[0],data[1],color=color,label=str(data[2][0]))
    elif label_type=="size":
        plt.plot(data[0],data[1],color=color,label=str(calc_mul(data[2][0])))
    else:
        plt.plot(data[0],data[1],color=color)
    
    if show_legend:
        plt.legend() # 显示图例

    if show_xy_labels:
        plt.xlabel("shape value")
        plt.ylabel("runtime(ms)")

def calc_mul(shape):
    result = 1
    for s in shape:
        if s!=-1:
            result*=s
    
    return result

def data_div(a,b):
    result = []

    xs=[]
    ys=[]
    for i in range(len(a[0])):
        xs.append(a[0][i])
        ys.append(a[1][i]/b[1][i])
    result.append((tuple(xs),tuple(ys),a[2],a[3],a[4]))

    return tuple(result)