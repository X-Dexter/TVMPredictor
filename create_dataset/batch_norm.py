# create the runtime-dataset for batch-normal-op

from create_dataset.common import test_op_time,create_dataset_nd
import tvm.relay as relay

shape = (100,100,100)

def calculate_time(dshape,dtype="float32"):
    '''
    test add-op in one kind of shape.
    '''
    x = relay.var("input_x", shape=dshape[0], dtype=dtype)
    f = relay.nn.batch_norm(x)

    return test_op_time(input_dict={"input_x": (dshape[0],dtype)},output=f,cycle_times=50)

create_dataset_nd(function=calculate_time,shape_relation={lambda x:x}, max_shapes=(100,100,100),sampling=(0.15,0.15,0.15),dtype="float32",file_name="batch_normal_float.txt")