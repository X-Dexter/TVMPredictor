# # create the runtime-dataset for batch-normal-op

# from create_dataset.common import test_op_time,create_dataset_nd
# import tvm.relay as relay
# import tvm

# shape = (100,100,100)

# def calculate_time(dshape,dtype="float32",target="llvm", device=tvm.cpu(0)):
#     '''
#     test add-op in one kind of shape.
#     '''
#     x = relay.var("input_x", shape=dshape[0], dtype=dtype)
#     f = relay.nn.batch_norm(x)

#     return test_op_time(input_dict={"input_x": (dshape[0],dtype)},output=f,cycle_times=50,target=target, device=device)

# create_dataset_nd(function={"body":calculate_time,"params":{"target": "llvm", "device": tvm.cpu(0)}},shape_relation={lambda x:x}, max_shapes=(100,100,100),sampling=(0.15,0.15,0.15),dtype="float32",file_name="batch_normal_float.txt",fold_path="create_dataset/datasets/dell04/")