# 生成add函数的训练集，原始示例代码

import time
import tvm
# from tvm import target
import tvm.relay as relay
import numpy as np
from tvm.contrib import graph_runtime

dshape1 = (4,2,2)
dshape2 = (1,5,2)
dtype="int32"
target_ = "llvm"
device=tvm.cpu(0)

x = relay.var("input_x", shape=dshape1,dtype=dtype)
y = relay.var("input_y", shape=dshape2,dtype=dtype)
# f = relay.add(x, y)
f = relay.nn.batch_matmul(x,y)
# f=relay.const(1.0, "float32")

func = relay.Function(relay.analysis.free_vars(f), f)

model = relay.transform.InferType()(tvm.ir.IRModule.from_expr(func))
shape_dict = {v.name_hint : v.checked_type for v in model["main"].params}
params = {}

lib = relay.build(model, target=target_, params=params)

with relay.build_config(opt_level=0):
    graph, lib, params2 = relay.build(model, target=target_, params=params)

# 给模型赋初始值
module = graph_runtime.create(graph, lib, device)

for i in range(1):
    test_input_x = np.random.uniform(1, 6, size=dshape1).astype(dtype)
    test_input_y = np.random.uniform(1, 6, size=dshape2).astype(dtype)
    # module.set_input('input_x', test_input_x)
    # module.set_input('input_y', test_input_y)

    # print('--input_x:\n', test_input_x)
    # print('--input_y:\n', test_input_y)

    # 测试时间
    t1_start = time.time()
    module.run()
    t1=time.time()-t1_start

    output = module.get_output(0)
    print("--output:\n",output,"\n")

    print(t1)