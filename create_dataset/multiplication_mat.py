# create the runtime-dataset for add-op

from create_dataset.common import create_dataset_2d, test_op_time
import tvm.relay as relay

shape = (100,100,100)

def calculate_time(dshape,dtype="float32"):
    '''
    test add-op in one kind of shape.
    '''

    dshape=convert_shape(dshape)
    # print(dshape)

    x = relay.var("input_x", shape=dshape[0], dtype=dtype)
    y = relay.var("input_y", shape=dshape[1], dtype=dtype)
    f = relay.nn.matmul(x,y)

    return test_op_time(input_dict={"input_x": (dshape[0],dtype), "input_y":(dshape[1],dtype)},output=f,cycle_times=25)

def convert_shape(shapes):
    if len(shapes[0])<3:
        shapes=[(1,*(shapes[0])), (1,*(shapes[1]))]

    return shapes

def relation(x,y):
    if len(x)!=len(y) or len(x)<2:
        return False

    if x[-1] != y[-1]:
        return False

    for i in range(len(x)):
        if i>= len(x)-2:
            break

        if x[i]!=y[i]:
            return False
    
    return True

create_dataset_2d(function=calculate_time,max_shapes=((100,100,100),(100,100,100)),sampling=((0.1,0.1,0.1),(0.1,0.1,0.1)),dtype="float32",file_name="mul_mat.txt",limit=relation)