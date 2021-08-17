import time
import tvm
# from tvm import target
import tvm.relay as relay
import numpy as np
from tvm.contrib import graph_runtime

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

# # test code
dshape = (100,100,100)
x = relay.var("input_x", shape=dshape,dtype="float32")
x.type_annotation.concrete_shape
y = relay.var("input_y", shape=dshape,dtype="float32")
f = relay.add(x, y)

print("avg:",test_op_time(input_dict={"input_x": (dshape,"float32"), "input_y":(dshape,"float32")},output=f,cycle_times=200))