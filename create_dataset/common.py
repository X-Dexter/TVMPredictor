import time
import tvm
# from tvm import target
import tvm.relay as relay
import numpy as np
from tvm.contrib import graph_runtime
from tqdm import tqdm

def test_op_time(input_dict,output, target="llvm", device=tvm.cpu(0),cycle_times=200, min_value=-100,max_value=100):
    '''
    Get the run-time(s) of single op. 

    Parameters
    ----------
    * input_dict:   give the real inputs. exp: { var_name_str: (shape,type)}
    * output:   give the real output that is compute by inputs
    * min_value ~ max_value:    when create the random input values, this gives the range.

    Returns
    ---------
    the avg run time(second) with one kind of shape.

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)
    '''

    func = relay.Function(relay.analysis.free_vars(output), output)
    model = relay.transform.InferType()(tvm.ir.IRModule.from_expr(func))

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(model, target=target, params={})

    # 给模型赋初始值
    module = graph_runtime.create(graph, lib, device)

    # 一次性随机生成所有输入矩阵
    input_values={}
    for key,size in input_dict.items():
        input_values[key] = np.random.uniform(min_value, max_value, size=(cycle_times+1,*(size[0]))).astype(size[1])

    total_time=0.0
    for i in range(cycle_times+1):
        for key,values in input_values.items():
            module.set_input(key, values[i])

        # 测试时间,初次测量存在模型加载时间，手动去除
        start = time.time()
        module.run()
        single_t =time.time()-start

        if i>0:
            total_time+=single_t

    return total_time/cycle_times

def uniform_sampling(array,sampling=0.1):
    '''
    uniform sampling on a list/tuple, sampling gives the hit rate.
    '''

    result = []
    if array is None or len(array) == 0 or sampling<=0.0 or sampling>1.0:
        return result

    interval = int(1.0/sampling) 
    i=0
    while i<len(array):
        result.append(array[i])
        i+=interval

    return result

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

def create_dataset_nd(function,shape_relation, max_shapes, sampling, dtype="float32",file_name="dataset.txt",fold_path="create_dataset/datasets/"):
    '''
    The dataset is obtained through uniform sampling. usable range: n->1,  but the n inputs have fixed relationship

    Parameters
    ----------
    * function: < function(shape,dtype)>, which can gives the run-time
    * max_shapes: give the max size of each dimensionality
    * shape_relation: give the shape relation between inputs. ep: y = op(x0, x1, ... , xn), then xn= shape_relation[n](x)
    * sampling: sampling/100 gives the hit rate
    '''

    shapes_in_dimensionality=[]
    for i in range(len(max_shapes)):
        shapes_in_dimensionality.append(uniform_sampling(range(max_shapes[i]), sampling[i]))

    # 为避免循环多层嵌套，以自适应高层维度，将嵌套循环展开
    total_case=1
    len_shape_in_dimensionality=[]
    for shape_in_dimensionality in shapes_in_dimensionality:
        length = len(shape_in_dimensionality)
        len_shape_in_dimensionality.append(length)
        total_case *= length

    # 打开文件
    fo = open(fold_path+file_name, "a")
    print("--total: ",total_case)
    for i in tqdm(range(total_case)):
        shape=[]
        temp=i
        # 计算shape
        for j in range(len(shapes_in_dimensionality)):
            len_shape=len_shape_in_dimensionality[j]
            
            shape.append(shapes_in_dimensionality[j][int(temp%len_shape)])
            temp = int(temp/len_shape)

        shape=redress_dimensionality(shape)
        if shape:
            shapes = []
            for relation in shape_relation:
                shapes.append(relation(shape))
            run_time = function(shapes,dtype)

            # 打开一个文件
            fo.write(" ".join(str(m) for m in shape)+","+str(run_time*1000000)+"\n")
        
    # 关闭打开的文件
    fo.close()

def create_dataset_1d(function,max_shapes, sampling, dtype="float32",file_name="dataset.txt",fold_path="create_dataset/datasets/"):
    '''
    The dataset is obtained through uniform sampling. usable range: y = op(X),  but the n inputs have fixed relationship

    Parameters
    ----------
    * function: < function(shape,dtype)>, which can gives the run-time
    * max_shapes: give the max size of each dimensionality
    * sampling: sampling/100 gives the hit rate
    '''

    create_dataset_nd(function=function,shape_relation=[lambda x:x],max_shapes=max_shapes,sampling=sampling,dtype=dtype,file_name=file_name,fold_path=fold_path)

def create_dataset_2d(function,max_shapes, sampling, dtype="float32",file_name="dataset.txt",fold_path="create_dataset/datasets/",limit = lambda x,y:True):
    '''
    The dataset is obtained through uniform sampling. usable range: z=op(x,y),  but there is nearly no relationship between x and y. You can set the limitation with limit(x,y) -> True/False

    Parameters
    ----------
    * function: < function(shape,dtype)>, which can gives the run-time
    * max_shapes: give the max size of each dimensionality
    * sampling: sampling/100 gives the hit rate
    * limit: when x,y is ok to be the inputs at the same time, limit(x,y) return True, or return False
    '''

    # 针对各输入维度生成采样空间
    shapes_in_dimensionality=[[],[]]
    for j in range(len(shapes_in_dimensionality)):
        for i in range(len(max_shapes[j])):
            shapes_in_dimensionality[j].append(uniform_sampling(range(max_shapes[j][i]), sampling[j][i]))

    # 为避免循环多层嵌套，以自适应多维输入，将 嵌套循环 展开成一维循环
    total_case=[1,1]
    len_shape_in_dimensionality=[[],[]]
    for i in range(len(shapes_in_dimensionality)):
        for shape_in_dimensionality in shapes_in_dimensionality[i]:
            length = len(shape_in_dimensionality)
            len_shape_in_dimensionality[i].append(length)
            total_case[i] *= length

    # 打开文件
    fo = open(fold_path+file_name, "a")
    print("model path: ", fold_path+file_name)

    print("--total: ",total_case)
    
    for i_x in tqdm(range(total_case[0])):
        shape_x = []
        temp_x = i_x
        # 生成shape x
        for j_x in range(len(shapes_in_dimensionality[0])):
            len_shape_x=len_shape_in_dimensionality[0][j_x]
            
            shape_x.append(shapes_in_dimensionality[0][j_x][int(temp_x % len_shape_x)])
            temp_x = int(temp_x/len_shape_x)

        shape_x=redress_dimensionality(shape_x)     # 第一个input的shape
        if not shape_x:   # shape不合理
            continue

        for i_y in range(total_case[1]):
            shape_y=[]
            temp_y=i_y
            # 生成shape y
            for j_y in range(len(shapes_in_dimensionality[1])):
                len_shape_y=len_shape_in_dimensionality[1][j_y]
                
                shape_y.append(shapes_in_dimensionality[1][j_y][int(temp_y % len_shape_y)])
                temp_y = int(temp_y/len_shape_y)

            shape_y=redress_dimensionality(shape_y)     # 第二个input的shape
            if shape_y:
                if limit(shape_x,shape_y):
                    run_time = function([shape_x,shape_y],dtype)

                    # 打开一个文件
                    fo.write(" ".join(str(m) for m in shape_x)+"|"+ " ".join(str(m) for m in shape_y) +","+str(run_time*1000000)+"\n")
            
    # 关闭打开的文件
    fo.close()