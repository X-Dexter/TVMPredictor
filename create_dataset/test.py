# 生成add函数的训练集

import time
import tvm
# from tvm import target
import tvm.relay as relay
import numpy as np
from tvm.contrib import graph_runtime

dshape = (1000,100,100)
target_ = "llvm"
device=tvm.cpu(0)

x = relay.var("input_x", shape=dshape,dtype="float32")
y = relay.var("input_y", shape=dshape,dtype="float32")
f = relay.add(x, y)

func = relay.Function(relay.analysis.free_vars(f), f)

model = relay.transform.InferType()(tvm.ir.IRModule.from_expr(func))
shape_dict = {v.name_hint : v.checked_type for v in model["main"].params}
params = {}

with relay.build_config(opt_level=3):
    graph, lib, params2 = relay.build(model, target=target_, params=params)

# 给模型赋初始值
module = graph_runtime.create(graph, lib, device)

for i in range(30):
    test_input_x = np.random.uniform(-1, 1, size=dshape).astype("float32")
    test_input_y = np.random.uniform(-1, 1, size=dshape).astype("float32")
    module.set_input('input_x', test_input_x)
    module.set_input('input_y', test_input_y)

    # 测试时间
    t1_start = time.time()
    module.run()
    t1=time.time()-t1_start

    print(t1)