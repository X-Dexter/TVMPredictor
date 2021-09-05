# create the runtime-dataset for add-op

from create_dataset.common import test_op_time,create_dataset_nd,test_data_copy_time
import tvm.relay as relay
import tvm

shape = (100,100,100)

def calculate_op_time_1(dshape,dtype="float32",target="llvm", device=tvm.cpu(0)):
    '''
    test add-op in one kind of shape.

    Parameters
    ----------

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)
    '''
    x = relay.var("input_x", shape=dshape[0], dtype=dtype)
    # y = relay.var("input_y", shape=dshape[1], dtype=dtype)
    tmp = x

    for i in range(2000):
        tmp = relay.add(tmp,tmp)

    f = tmp

    # return test_op_time(input_dict={"input_x": (dshape[0],dtype)},output=f,cycle_times=50,target=target, device=device,use_tvm_test_function=True,min_repeat_ms=10)

    return test_op_time(input_dict={"input_x": (dshape[0],dtype)},output=f,cycle_times=50,target=target, device=device,use_tvm_test_function=False)

def calculate_op_time_2(dshape,dtype="float32",target="llvm", device=tvm.cpu(0)):
    '''
    test add-op in one kind of shape.

    Parameters
    ----------

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)
    '''
    x = relay.var("input_x", shape=dshape[0], dtype=dtype)
    # y = relay.var("input_y", shape=dshape[1], dtype=dtype)
    f=relay.add(x,x)

    return test_op_time(input_dict={"input_x": (dshape[0],dtype)},output=f,cycle_times=50,target=target, device=device,use_tvm_test_function=True,min_repeat_ms=10)

def calculate_op_time_3(dshape,dtype="float32",target="llvm", device=tvm.cpu(0)):
    '''
    test add-op in one kind of shape.

    Parameters
    ----------

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)
    '''
    x = relay.var("input_x", shape=dshape[0], dtype=dtype)
    # y = relay.var("input_y", shape=dshape[1], dtype=dtype)
    f=relay.add(x,x)

    # return test_op_time(input_dict={"input_x": (dshape[0],dtype)},output=f,cycle_times=50,target=target, device=device,use_tvm_test_function=True,min_repeat_ms=10)

    return test_op_time(input_dict={"input_x": (dshape[0],dtype)},output=f,cycle_times=50,target=target, device=device,use_tvm_test_function=False)

# def calculate_copy_time(dshape,dtype="float32",target="llvm", device=tvm.cpu(0)):
#     '''
#     test add-op in one kind of shape.

#     Parameters
#     ----------

#     exp:
#     * GPU: target = "cuda", device = tvm.cuda(0)
#     * CPU: target = "llvm", device=tvm.cpu(0)
#     '''
#     x = relay.var("input_x", shape=dshape[0], dtype=dtype)
#     y = relay.var("input_y", shape=dshape[1], dtype=dtype)
#     f = relay.add(x, y)

#     return test_data_copy_time(input_dict={"input_x": (dshape[0],dtype), "input_y":(dshape[1],dtype)},output=f,cycle_times=50,target=target, device=device)
create_dataset_nd(function={"body":calculate_op_time_3,"params":{"target": "llvm", "device": tvm.cpu(0)}},shape_relation=[lambda x:x],max_shapes=(100,100,100),sampling=(0.03,0.03,0.03),dtype="float32",file_name="multi-add-3.txt",fold_path="create_dataset/datasets/own_way_abandon/dell04/")

create_dataset_nd(function={"body":calculate_op_time_1,"params":{"target": "llvm", "device": tvm.cpu(0)}},shape_relation=[lambda x:x],max_shapes=(100,100,100),sampling=(0.03,0.03,0.03),dtype="float32",file_name="multi-add-1.txt",fold_path="create_dataset/datasets/own_way_abandon/dell04/")
create_dataset_nd(function={"body":calculate_op_time_2,"params":{"target": "llvm", "device": tvm.cpu(0)}},shape_relation=[lambda x:x],max_shapes=(100,100,100),sampling=(0.03,0.03,0.03),dtype="float32",file_name="multi-add-2.txt",fold_path="create_dataset/datasets/own_way_abandon/dell04/")
# create_dataset_nd(function={"body":calculate_op_time,"params":{"target": "cuda", "device": tvm.cuda(0)}},shape_relation=[lambda x:x, lambda x:x],max_shapes=(100,100,100),sampling=(0.15,0.15,0.15),dtype="float32",file_name="add_float_gpu.txt",fold_path="create_dataset/datasets/dell04/")


# create_dataset_nd(function={"body":calculate_op_time,"params":{"target": "llvm", "device": tvm.cpu(0)}},shape_relation=[lambda x:x, lambda x:x],max_shapes=(100,100,100),sampling=(0.15,0.15,0.15),dtype="float32",file_name="add_float.txt",fold_path="create_dataset/datasets/dell04/")
# create_dataset_nd(function={"body":calculate_op_time,"params":{"target": "cuda", "device": tvm.cuda(0)}},shape_relation=[lambda x:x, lambda x:x],max_shapes=(100,100,100),sampling=(0.15,0.15,0.15),dtype="float32",file_name="add_float_gpu.txt",fold_path="create_dataset/datasets/dell04/")

# create_dataset_nd(function={"body":calculate_copy_time,"params":{"target": "llvm", "device": tvm.cpu(0)}},shape_relation=[lambda x:x, lambda x:x],max_shapes=(100,100,100),sampling=(0.15,0.15,0.15),dtype="float32",file_name="copy_add_float.txt",fold_path="create_dataset/datasets/dell04/")
# create_dataset_nd(function={"body":calculate_copy_time,"params":{"target": "cuda", "device": tvm.cuda(0)}},shape_relation=[lambda x:x, lambda x:x],max_shapes=(100,100,100),sampling=(0.15,0.15,0.15),dtype="float32",file_name="copy_add_float_gpu.txt",fold_path="create_dataset/datasets/dell04/")