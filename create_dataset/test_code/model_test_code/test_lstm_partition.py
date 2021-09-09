from std_memory_profiler import operation_memory_profile, operation_time_profile,operation_cuda_memory_profile, profile
from onnx_profiler import compile_tvm_mod,run_interpreter
from tvm.relay.testing import lstm,densenet,init
import numpy as np
import onnx
import tvm

target = "llvm"
device = tvm.cpu(0)
onnx_model = onnx.load("lstm.onnx")
shape_dict = {'input': (5, 3, 10), 'h0':(2, 3, 20), 'c0':(2, 3, 20)}
mod, params = tvm.relay.frontend.from_onnx(onnx_model, shape_dict)

