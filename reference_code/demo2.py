# 示例：自定义算子

import numpy as np
import tvm
import tvm.topi
# import logging
# from numbers import Integral
# from tvm.topi import utils
from tvm.topi.nn.conv2d import conv2d_nchw
# from tvm.topi.nn.utils import  get_pad_tuple
# import sys
# from tvm import autotvm

n, ih, iw = 1, 7, 7
new_ic, kh, kw = 1024, 3, 3
dtype = 'float32'
index_dtype = 'int32'
stride_h, stride_w = (1, 1)
pad_h, pad_w = (1, 1)
dilation_h, dilation_w = (1, 1)
oh = (ih + 2 *pad_h - kh) // stride_h + 1
ow = (iw + 2 * pad_w - kw) // stride_w + 1

def test_topi_conv2d(ic = 1024, oc = 1024):
    
    A = tvm.placeholder(shape=(n, ic, ih, iw), dtype=dtype, name='A')
    B = tvm.placeholder(shape=(oc, ic, kh, kw), dtype=dtype, name='B')
    output = conv2d_nchw(Input = A, Filter = B, stride = (stride_h, stride_w), padding = (pad_h, pad_w), dilation = (dilation_h, dilation_w))
    
    s = tvm.te.create_schedule(output.op)
    s = tvm.te.schedule(s,output)
    # func_cpu = tvm.build(s, [A, B, output], target="llvm")
    # print(tvm.lower(s, [A, B, output], simple_mode=True))
    a_np = np.random.uniform(-1, 1, size=(n, ic, ih, iw)).astype(dtype)
    # b_np = np.random.uniform(-1, 1, size=(oc, ic, kh, kw)).astype(dtype)
    ctx = tvm.context("llvm",0)	
    # d_cpu = tvm.nd.array(np.zeros((n, oc, oh, ow), dtype=dtype), ctx)
    a = tvm.nd.array(a_np, ctx)

    return a