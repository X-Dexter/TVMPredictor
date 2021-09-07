import time
import hashlib
import json
import os
import tvm
import random
import copy
# from tvm import target
import tvm.relay as relay
import numpy as np
from tvm.contrib import graph_runtime
from tvm.contrib import graph_executor
from tqdm import tqdm
from tvm.runtime.module import Module

class Device:
    device_params_CPU = {"target": "llvm", "device": tvm.cpu(0),"type":-1}
    device_params_GPU0 = {"target": "cuda", "device": tvm.cuda(0),"type":0}
    device_params_GPU1 = {"target": "cuda", "device": tvm.cuda(1),"type":1}

def generate_datasets_with_one_dimensionality_changing(count,shape_dimensionality,range_min,range_max,function_dict,min_shapes,max_shapes,sampling,force_shape_relation=None,dtype="float32",device_parame_array=[{"target": "llvm", "device": tvm.cpu(0),"type":-1}],cycle_times=200,min_repeat_ms=500,opt_level=0,fold_path="create_dataset/datasets/",device_name="dell04",dataset_config_name="dataset.json",show_print=True) ->None:
    '''
    * count: the datasets enum try best to generate.
    * shape_dimensionality: gives the shape info and changing dimensionality. ((a,b,c),(x,y)) means there is three inputs, and the first input is a-d, the second input is b-d, the third input is c-d. The y-d of the x-th input is changing.
    * range_min ~ range_max: gives the range when random generate the unchanging dimensionality.
    * function_dict: < {"func": op-function, "name": "op-name"} > .  In which: op-function(shape_tuple, dtype) -> func
    * min_shapes ~ max_shapes: give the range of the changing demensionality
    * sampling: when ensure the range of the changing demensionality, sampling gives the sampling hit rate.
    * force_shape_relation: gives the relation of every demensionality in every inputs reference to the shape of the changing input.
    * opt_level:   The optimization level of this pass.[0-3?]. opt_level= 0 means disable optimization.
    * dtype: when test the run time, the input data type
    * device_parames_array: [device_parames_1, device_parames_2....]，reference to function<generate_dataset_with_one_dimensionality_changing>
    * cycle_times: when test single-op, minimum test times
    * min_repeat_ms: if time(op)*cycle_times < min_repeat_ms，test will go on until fit.
    * fold_path: the save root fold for datasets
    * device_name: the alias-name of device where you test op.
    * dataset_config_name: the dataset config name, which record and organize the whole dataset. It's a good choice to keep the default value if you are not sure.
    * show_print: show progress-information in terminal or not when generate the datasets.

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)

    # ast.literal_eval("(255, 0, 0)") can get tuple(255,0,0)

    Returns
    ---------
    no return value.
    '''
    log_file=os.path.join(fold_path,dataset_config_name)
    log_dict = {"count":0}
    if os.path.exists(log_file):
        with open(log_file,'r') as f:
            log_dict = json.load(f)
    
    counts=[]
    for device_params in device_parame_array:
        if device_name.lower() not in log_dict or function_dict["name"].lower() not in log_dict[device_name.lower()] or str(device_params["type"]) not in log_dict[device_name.lower()][function_dict["name"].lower()] or str(shape_dimensionality) not in log_dict[device_name.lower()][function_dict["name"].lower()][str(device_params["type"])]:
            counts.append(count)
        else:
            counts.append(count -log_dict[device_name.lower()][function_dict["name"].lower()][str(device_params["type"])][str(shape_dimensionality)]["count"])

    if count<=0:
        if show_print:
            print("count(data) is enough.")
        return

    for i in range(count):
        shapes = create_random_shape(shape_dimensionality,range_min,range_max)
        for device_params,left_count in zip(device_parame_array,counts):
            if i<left_count:
                generate_dataset_with_one_dimensionality_changing(function_dict = function_dict,shapes=shapes,min_shapes=min_shapes,max_shapes=max_shapes,sampling=sampling,force_shape_relation=force_shape_relation,dtype=dtype,device_parames=device_params,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level,fold_path=fold_path,device_name=device_name,dataset_config_name=dataset_config_name,show_print=show_print)

def generate_dataset_with_one_dimensionality_changing(function_dict,shapes,min_shapes,max_shapes,sampling,force_shape_relation=None,dtype="float32",device_parames={"target": "llvm", "device": tvm.cpu(0),"type":-1},cycle_times=200,min_repeat_ms=500,opt_level=0,fold_path="create_dataset/datasets/",device_name="dell04",dataset_config_name="dataset.json",show_print=True) ->None:
    '''
    save the op-time(ms) to disk-file when only a single dimensionality of one input-shape changing.

    Parameters
    ----------
    * function_dict: {"func": (tvm.relay.Function), "name": op-name}, func is a tensor which can describe an op-process.
    * shapes: (python tuple) which describe the tensor-shape of inputs, exp: ((x0,-1,x2), (y0,y1,y2)); -2 means no exist.
    * min_shapes ~ max_shapes: give the range of the changing dimensionality
    * sampling: sampling/100 gives the hit rate
    * opt_level:   The optimization level of this pass.[0-3?]. opt_level= 0 means disable optimization.
    * dtype: when test the run time, the input data type
    * device_parames_array: [device_parames_1, device_parames_2....]
    * cycle_times: when test single-op, minimum test times
    * min_repeat_ms: if time(op)*cycle_times < min_repeat_ms，test will go on until fit.
    * cycle_times: when test single-op, minimum test times
    * min_repeat_ms: if time(op)*cycle_times < min_repeat_ms，test will go on until fit.
    * fold_path: the save root fold for datasets
    * device_name: the alias-name of device where you test op.
    * dataset_config_name: the dataset config name, which record and organize the whole dataset. It's a good choice to keep the default value if you are not sure.
    * show_print: show progress-information in terminal or not when generate the datasets.
    * device_parames: type=-1: CPU, type=n: GPU(n)

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)

    # ast.literal_eval("(255, 0, 0)") can get tuple(255,0,0)

    Returns
    ---------
    no return value.
    '''
    # 生成不冲突的数据集路径
    x,y = search_changing_shape(shapes)
    shape_dimensionality_str = str((get_dimensionality(shapes),(x,y)))
    real_shape = generate_shape(shapes,x,y,-1,force_shape_relation=force_shape_relation)

    data_savepath = os.path.join(ensure_dir_exist(os.path.join(fold_path,device_name,function_dict["name"],str(device_parames["type"]),str(shape_dimensionality_str))),str(real_shape)+".txt")

    if os.path.exists(data_savepath) and show_print:
        print("exists and skip: %s"%(data_savepath))
        return

    dshape = uniform_sampling(min_shapes,max_shapes,sampling)
    f_dataset = open(data_savepath, "a")

    # 写入执行时间
    for i in range(len(dshape)):
        runtime = test_op_time(function_dict["func"](generate_shape(shapes,x,y,dshape[i],force_shape_relation=force_shape_relation),dtype=dtype),device_parames=device_parames,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level)
        f_dataset.write(str(dshape[i])+","+str(runtime*1000)+"\n")
    f_dataset.close()

    if show_print:
        print("create:\n--op: %s\n--device: %s\n--shape: %s\n--file: %s\n\n"%(function_dict["name"].lower(),translate_device_type(device_parames["type"]),str(real_shape),data_savepath))

    # 构建数据集存档信息
    log_file=os.path.join(fold_path,dataset_config_name)

    log_dict = {"count":0}
    if os.path.exists(log_file):
        with open(log_file,'r') as f:
            log_dict = json.load(f)

    # 检查字典树路径存在
    # 确保 设备名-->keys
    if device_name.lower() not in log_dict.keys():
        log_dict[device_name.lower()] = {}
        log_dict[device_name.lower()]["count"] = 0 

    # 确保 算子名-->keys
    if function_dict["name"].lower() not in log_dict[device_name.lower()].keys():
        log_dict[device_name.lower()][function_dict["name"].lower()] = {}
        log_dict[device_name.lower()][function_dict["name"].lower()]["count"] = 0 

    # 确保 硬件类型-->keys
    if str(device_parames["type"]) not in log_dict[device_name.lower()][function_dict["name"].lower()].keys():
        log_dict[device_name.lower()][function_dict["name"].lower()][str(device_parames["type"])] = {}
        log_dict[device_name.lower()][function_dict["name"].lower()][str(device_parames["type"])]["count"] = 0 
    
    # 确保 输入shape_dimensionality-->keys
    if shape_dimensionality_str not in log_dict[device_name.lower()][function_dict["name"].lower()][str(device_parames["type"])].keys():
        log_dict[device_name.lower()][function_dict["name"].lower()][str(device_parames["type"])][shape_dimensionality_str] = {}
        log_dict[device_name.lower()][function_dict["name"].lower()][str(device_parames["type"])][shape_dimensionality_str]["count"] = 0

    # 确保 输入shapes-->keys
    if str(real_shape) not in log_dict[device_name.lower()][function_dict["name"].lower()][str(device_parames["type"])][shape_dimensionality_str].keys():
        log_dict[device_name.lower()][function_dict["name"].lower()][str(device_parames["type"])][shape_dimensionality_str][str(real_shape)] = {}
        
    # 记录数据集文件名，shape形状，开始训练时间  
    log_dict[device_name.lower()][function_dict["name"].lower()][str(device_parames["type"])][shape_dimensionality_str][str(real_shape)]["file_path"]=data_savepath
    log_dict[device_name.lower()][function_dict["name"].lower()][str(device_parames["type"])][shape_dimensionality_str][str(real_shape)]["changed_shape"]=shape_dimensionality_str
    log_dict[device_name.lower()][function_dict["name"].lower()][str(device_parames["type"])][shape_dimensionality_str][str(real_shape)]["shapes"]=str(real_shape)
    log_dict[device_name.lower()][function_dict["name"].lower()][str(device_parames["type"])][shape_dimensionality_str][str(real_shape)]["time"]= time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    # 数量递增
    log_dict["count"] += 1
    log_dict[device_name.lower()]["count"] += 1
    log_dict[device_name.lower()][function_dict["name"].lower()]["count"] +=1
    log_dict[device_name.lower()][function_dict["name"].lower()][str(device_parames["type"])]["count"] +=1
    log_dict[device_name.lower()][function_dict["name"].lower()][str(device_parames["type"])][shape_dimensionality_str]["count"] += 1
    
    with open(log_file,"w") as f:
        json.dump(log_dict,fp=f,indent=4,separators=(',', ': '),sort_keys=True)
    
def uniform_sampling(min,max,sampling=0.1) ->list :
    '''
    uniform sampling on a list, sampling gives the hit rate.
    '''

    result = []
    if min<1 or min>max or sampling<=0.0 or sampling>1.0:
        return result

    interval = int(1.0/sampling) 
    i=min
    while i<=max:
        result.append(i)
        i+=interval

    return result

def get_dimensionality(shape) ->tuple:
    a=[]
    for i in range(len(shape)):
        a.append(len(shape[i]))
    return tuple(a)

def test_op_time(func, device_parames,cycle_times,min_repeat_ms,opt_level=0) -> float :
    '''
    Get the run-time(us) of single op with a certain shape on certain device.

    Parameters
    ----------
    * func_dict:   give the real inputs. exp: { "input_shapes": (shape1,shape2...), "function": function }, function is the output of the module function.
    * opt_level:   The optimization level of this pass.[0-3?]. opt_level= 0 means disable optimization.
    * device_parames: 

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)

    Returns
    ---------
    the avg run time(second) with one kind of shape.
    '''

    model = relay.transform.InferType()(tvm.ir.IRModule.from_expr(relay.Function(relay.analysis.free_vars(func), func)))

    with tvm.transform.PassContext(opt_level=opt_level):   
        lib = relay.build(model, target=device_parames["target"], params={})

    # 给模型赋初始值
    module = graph_executor.GraphModule(lib["default"](device_parames["device"]))

    ftimer = module.module.time_evaluator("run", device_parames["device"], repeat=cycle_times, min_repeat_ms=min_repeat_ms, number=1)
    return np.mean(np.array(ftimer().results))

def redress_dimensionality(dimensions):
    '''
    Correct the dimension with 0
    '''
    result = []
    for dimension in dimensions:
        if dimension==0 and len(result)>0:
            return None                         # 屏蔽中间出现0的shape
        
        if dimension>0:
            result.append(dimension)
    
    if len(result)<1:
        return None
    else:
        return tuple(result)

def ensure_dir_exist(dir_path) ->str:
    '''
    create the dir-tree if not exists.
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def search_changing_shape(shapes) ->tuple:
    '''
    search the changing dimensionality index.
    '''
    for i in range(len(shapes)):
        for j in range(len(shapes[i])):
            if shapes[i][j]==-1:
                return i,j

def generate_shape(shapes,x,y,value,force_shape_relation=None) ->tuple:
    '''
    replace the shapes[x][y] with the given value, and it won't change original shapes.

    Parameters
    ----------
    * x,y: shapes[x][y] is the changing value in shapes
    * force_shape_relation: when the other shapes have an relation with the changing shape, we can use this. if one dimensionality with on relation, you can give that dimensionality with None.

    * exp:

    if the shapes relation we want to have is ((a,b,c),(b,a,d)), and the first dimensionality's a is changing

    force_shape_relation = (None, (lambda x,y,z:y, lambda x,y,z:x, None))

    '''
    # tuple转化为list
    tmp = []
    for i in range(len(shapes)):
        tmp.append(list(shapes[i]))
    tmp[x][y]=value

    if force_shape_relation:
        for i in range(len(shapes)):
            if i != x :
                # 调整第i个shape
                if not force_shape_relation[i]:
                    # 保持不变
                    continue
                for j in range(len(shapes[i])):
                    if not force_shape_relation[i][j]:
                        # tmp[i][j]保持不变
                        continue
                    tmp[i][j]=force_shape_relation[i][j](*tmp[x])
    
    # list转化为tuple
    result=[]
    for i in range(len(tmp)):
        result.append(tuple(tmp[i]))
                               
    return tuple(result)

def create_random_shape(shape_dimensionality,min,max):
    '''
    * shapes[i][j] have range from min[i][j] to max[i][j].
    '''
    # tuple转化为list
    x = shape_dimensionality[1][0]
    y = shape_dimensionality[1][1]

    # 生成shapes list，变化的维度为-1
    shapes = []
    for i in range(len(shape_dimensionality[0])):
        tmp = []
        for j in range(shape_dimensionality[0][i]):
            if x==i and y==j:
                # 变化的维度
                tmp.append(-1)
                continue
            tmp.append(random.randint(min[i][j],max[i][j]))
        shapes.append(tmp)

    # list转化为tuple
    result=[]
    for i in range(len(shapes)):
        result.append(tuple(shapes[i]))
                               
    return tuple(result)

def translate_device_type(type_id)->str:
    if type_id<0:
        return "CPU"

    return "GPU(" + str(type_id) + ")"


def fix_json_config(log_file="create_dataset/datasets/dataset.json") ->None:
    '''
    delete un-exist dataset-items from json config
    '''

    log_dict = {}
    if os.path.exists(log_file):
        with open(log_file,'r') as f:
            log_dict = json.load(f)

    result = copy.deepcopy(log_dict)

    for device_name,device_dict in log_dict.items():
        if device_name=="count":
            continue
        for function_name,function_dict in device_dict.items():
            if function_name=="count":
                continue
            for device_type,device_type_dict in function_dict.items():
                if device_type=="count":
                    continue
                for shape_dimensionality,shape_dimensionality_dict in device_type_dict.items():
                    if shape_dimensionality=="count":
                        continue
                    for shapes_str,value in shape_dimensionality_dict.items():
                        if shapes_str=="count":
                            continue
                        if not os.path.exists(value["file_path"]):
                            del result[device_name][function_name][device_type][shape_dimensionality][shapes_str]           # 删除对应条目

                            result[device_name][function_name][device_type][shape_dimensionality]["count"]-=1
                            # 没有条目时，删除维度信息
                            if result[device_name][function_name][device_type][shape_dimensionality]["count"]<=0:
                                del result[device_name][function_name][device_type][shape_dimensionality] 
       
                            result[device_name][function_name][device_type]["count"]-=1
                            # 没有条目时，删除硬件信息
                            if result[device_name][function_name][device_type]["count"]<=0:
                                del result[device_name][function_name][device_type]

                            result[device_name][function_name]["count"]-=1
                            # 没有条目时，删除算子信息
                            if result[device_name][function_name]["count"]<=0:
                                del result[device_name][function_name]

                            result[device_name]["count"]-=1
                            # 没有条目时，删除设备信息
                            if result[device_name]["count"]<=0:
                                del result[device_name]

                            result["count"] -= 1 
                            if result["count"] <= 0:
                                result={"count": 0}

    with open(log_file,"w") as f:
        json.dump(result,fp=f,indent=4,separators=(',', ': '),sort_keys=True)