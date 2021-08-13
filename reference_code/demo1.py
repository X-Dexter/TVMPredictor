# 实例：生成计算图，并打印其表达式（抽象表达式）

from tvm import relay
# from tvm.relay.testing import run_opt_pass

dshape = (1, 16, 64, 64)
x = relay.var("x", shape=dshape)
x = relay.add(x, relay.const(1, "float32"))
y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(1, 1), padding=(0, 0), channels=16)
y1 = relay.add(relay.const(1, "float32"), y)
y = relay.add(y, y1)
z2 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=16)
z3 = relay.nn.conv2d(y, relay.var("w3"), kernel_size=(1, 1), padding=(0, 0), channels=16)
z = relay.add(z2, z3)
func = relay.Function(relay.analysis.free_vars(z), z)
print("before pass: ", func)

# func = run_opt_pass(func, relay.transform.FuseOps(fuse_opt_level=1))
# print("before pass: ", func)