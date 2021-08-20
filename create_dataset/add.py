# create the runtime-dataset for add-op

from create_dataset.common import test_op_time,create_dataset_nd
import tvm.relay as relay
import tvm

shape = (100,100,100)

def calculate_time(dshape,dtype="float32",target="llvm", device=tvm.cpu(0)):
    '''
    test add-op in one kind of shape.
    '''
    x = relay.var("input_x", shape=dshape[0], dtype=dtype)
    y = relay.var("input_y", shape=dshape[1], dtype=dtype)
    f = relay.add(x, y)

    return test_op_time(input_dict={"input_x": (dshape[0],dtype), "input_y":(dshape[1],dtype)},output=f,cycle_times=50,target=target, device=device)

create_dataset_nd(function={"body":calculate_time,"params":{"target": "llvm", "device": tvm.cpu(0)}},shape_relation=[lambda x:x, lambda x:x],max_shapes=(100,100,100),sampling=(0.15,0.15,0.15),dtype="float32",file_name="add_float.txt")