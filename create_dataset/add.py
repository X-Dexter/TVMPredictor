# create the runtime-dataset for add-op

from numpy import random
from create_dataset.common import test_op_time,create_dataset
import tvm
# from tvm import target
import tvm.relay as relay
import numpy as np

shape = (100,100,100)

def calculate_time(dshape,dtype="float32"):
    '''
    test add-op in one kind of shape.
    '''
    x = relay.var("input_x", shape=dshape, dtype=dtype)
    y = relay.var("input_y", shape=dshape, dtype=dtype)
    f = relay.add(x, y)

    return test_op_time(input_dict={"input_x": (dshape,dtype), "input_y":(dshape,dtype)},output=f,cycle_times=50)

create_dataset(function=calculate_time,max_shapes=(100,100,100),sampling=(0.15,0.15,0.15),dtype="float32",file_name="test.txt")