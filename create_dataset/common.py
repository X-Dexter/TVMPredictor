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

def create_dataset(function, max_shapes, sampling, dtype="float32",file_name="dataset.txt",fold_path="create_dataset/datasets/"):
    '''
    The dataset is obtained through uniform sampling. 

    Parameters
    ----------
    * function: < function(shape,dtype)>, which can gives the run-time
    * max_shapes: give the max size of each dimensionality
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
        for i in range(len(shapes_in_dimensionality)):
            len_shape=len_shape_in_dimensionality[i]
            
            shape.append(shapes_in_dimensionality[i][int(temp%len_shape)])
            temp = int(temp/len_shape)

        shape=redress_dimensionality(shape)
        if shape:
            run_time = function(shape,dtype)

            # 打开一个文件
            fo.write(" ".join(str(i) for i in shape)+","+str(run_time*1000000)+"\n")
        
    # 关闭打开的文件
    fo.close()

# # test code
# dshape = (2,0,2)
# x = relay.var("input_x", shape=dshape,dtype="float32")
# y = relay.var("input_y", shape=dshape,dtype="float32")
# f = relay.add(x, y)

# print("avg:",test_op_time(input_dict={"input_x": (dshape,"float32"), "input_y":(dshape,"float32")},output=f,cycle_times=1))