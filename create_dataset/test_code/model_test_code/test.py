from operation_statistics.cnn_workload_generator import get_network, compile_without_log, create_graph_executor_on_single_device, evaluate_time_with_tvm_evaluator
'''
run on TeslaT4:
16:2521.94 ms,32:5152.75 ms,64:10310.36 ms,128:21304.56 ms,256:42367.17 ms,512: 85813.79 ms,1024:175231.19

'''
mod, params, input_shape, output_shape = get_network("mobilenet", (260,3,224,224), dtype="float32")
target = "cuda"

lib = compile_without_log(mod, target, params)
module = create_graph_executor_on_single_device(lib,input_shape,target)
evaluate_time_with_tvm_evaluator(module, target)
