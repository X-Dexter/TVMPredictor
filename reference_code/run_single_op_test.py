# 运行单个算子测试

import time
import tvm
# from tvm.relay import transform
import tvm.relay as relay
import numpy as np
from tvm.contrib import graph_runtime
# from TVMProfiler.relayIR.relay_graph import construct_op_graph, profile_resource_usage

dshape = (1000,100,100)
test_input = np.random.uniform(-1, 1, size=dshape).astype("float32")

x = relay.var("input_x", shape=dshape,dtype="float32")
f = relay.add(x, relay.const(1.0, "float32"))

func = relay.Function(relay.analysis.free_vars(f), f)
print("--before pass:\n", func,"\n")

model = relay.transform.InferType()(tvm.ir.IRModule.from_expr(func))
shape_dict = {v.name_hint : v.checked_type for v in model["main"].params}

# 打印模型和input参数列表
print("--model:\n",model,"\n")
print("--params shape:\n",shape_dict,"\n")

# 赋值: 一般用作给模型（权重和偏置）赋初始值。此处如果给input赋值则不能再调用set_input赋值
params = {}
# for k, v in shape_dict.items():
#     if k == "data":
#         continue
#     init_value = np.random.uniform(-1, 1, v.concrete_shape).astype(v.concrete_shape.dtype)
#     params[k] = tvm.nd.array(init_value, device=tvm.cpu(0))

# params["input_x"] = tvm.nd.array(np.array((1.0,2.0,3.0),dtype="float32"),device=tvm.cpu(0)) 
# params["input_x"] = tvm.nd.array(test_input, device=tvm.cpu(0))

print("--params:\n",params,"\n")

with relay.build_config(opt_level=3):
    graph, lib, params2 = relay.build(model, "llvm", params=params)

# print("--TVM graph:\n", graph,"\n")

# 给模型赋初始值
device=tvm.cpu(0)
module = graph_runtime.create(graph, lib, device)
# module.set_input('input_x', np.array((1.0,2.0,3.0),dtype="float32"))
module.set_input('input_x', test_input)
module.set_input(**params)

# 测试时间
t1_start = time.time()
module.run()
t1=time.time()-t1_start

output = module.get_output(0)
print("--output:\n",output,"\n")

# 测试方法2，存在问题
entrance_tuple = model.functions.items()[0]
main_function = entrance_tuple[1]

call_body = main_function.body.tuple_value if hasattr(main_function.body,"tuple_value") else main_function.body
temp_body = tvm.relay.Call(call_body.op, call_body.args, attrs=call_body.attrs, type_args=call_body.type_args)
# temp_body1 = tvm.relay.expr.TupleGetItem(temp_body,0)
call_function = tvm.relay.Function(relay.analysis.free_vars(temp_body),temp_body)
call_functions = {"main": call_function}
call_ir_module = tvm.ir.IRModule(functions=call_functions)
with tvm.transform.PassContext(opt_level=3):
    call_interpreter = relay.build_module.create_executor("graph", call_ir_module, device, "llvm")

print("--call_ir_module:\n",call_ir_module,"\n")
input_args = []
input_args.append(test_input)
print("--params.keys():\n",params.keys(),"\n")

# 测试时间
t2_start = time.time()
res = call_interpreter.evaluate()(*input_args, **params)
t2=time.time()-t2_start

print("--output by model:\n    result:--\n",output,"\n    time",t1,"\n")
print("--res by re-op:\n    result:--\n",res,"\n    time:",t2,"\n")

print("time model: %0.3f ms"%(t1*1000))
print("time re-op: %0.3f ms"%(t2*1000))
print("re-op/model: %.2f"%(t2/t1))