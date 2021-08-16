import tvm
# from tvm.relay import transform
import tvm.relay as relay
# import numpy as np
# from tvm.contrib import graph_runtime

dshape = (1, 16, 64, 64)
x = relay.var("x", shape=dshape)
f = relay.add(x, relay.const(1.0, "float32"))

func = relay.Function(relay.analysis.free_vars(f), f)
print("--before pass:\n", func,"\n")

model = relay.transform.InferType()(tvm.ir.IRModule.from_expr(func))
shape_dict = {v.name_hint : v.checked_type for v in model["main"].params}

print("--model:\n",model,"\n")
print("--params shape:\n",shape_dict,"\n")