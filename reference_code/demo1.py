# 实例：生成计算图，并打印其表达式（抽象表达式）

import tvm
from tvm import relay
from tvm.relay.testing import run_opt_pass

# dshape = (1, 16, 64, 64)
# x = relay.var("x", shape=dshape)
# f = relay.add(x, relay.const(1.0, "float32"))
# y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(1, 1), padding=(0, 0), channels=16)
# y1 = relay.add(relay.const(1, "float32"), y)
# y = relay.add(y, y1)
# z2 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=16)
# z3 = relay.nn.conv2d(y, relay.var("w3"), kernel_size=(1, 1), padding=(0, 0), channels=16)
# z = relay.add(z2, z3)
# func = relay.Function(relay.analysis.free_vars(z), z)
# func = relay.Function(relay.analysis.free_vars(f), f)


x = relay.var("input_x", shape=(1,5,4), dtype="float32")
y = relay.var("input_y", shape=(1,6,4), dtype="float32")
f = relay.nn.batch_matmul(x,y)

print("before pass: ", f)

mod = tvm.ir.IRModule.from_expr(f)
mod = relay.transform.InferType()(mod)

# func = run_opt_pass(func, relay.transform.FuseOps(fuse_opt_level=1))
# print("before pass: ", func)

entrance_tuple = mod.functions.items()[0]
main_function = entrance_tuple[1]

call_body = main_function.body.tuple_value if hasattr(main_function.body,"tuple_value") else main_function.body

print("以下是关键信息：")
print("--op type:",call_body.op)
print("--input args[0].shape:",call_body.args[0].checked_type.shape)
print("--input args[1].shape:",call_body.args[1].checked_type.shape)
print("--output shape:",call_body.checked_type.shape)

print()