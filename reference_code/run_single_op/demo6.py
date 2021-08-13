# 参考：https://www.pythonf.cn/read/152182

import tvm
n = 1024
A = tvm.te.placeholder((n,), name='A')
B = tvm.te.placeholder((n,), name='B')
C = tvm.te.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))  # <class 'tvm.te.tensor.Tensor'>
print(C)

s = tvm.create_schedule(C.op)
