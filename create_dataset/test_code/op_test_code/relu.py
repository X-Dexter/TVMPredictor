from create_dataset.common import generate_datasets_with_one_dimensionality_changing,Device
import tvm.relay as relay

def calculate_op(dshape,dtype="float32"):
    '''
    test add-op in one kind of shape.

    Parameters
    ----------

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)
    '''
    x = relay.var("input_x", shape=dshape[0], dtype=dtype)
    f = relay.nn.relu(x)

    return f

# 定义参数
function_dict = {"func":calculate_op, "name": "relu"}
min_shapes=1
max_shapes=100
sampling=1.0
dtype="float32"

cycle_times=20
min_repeat_ms=30
opt_level=0
fold_path="create_dataset/datasets/"
device_name="dell04"
show_print=True
# log_file默认值.

# 研究二维加法
count2 = 7

force_shape_relation2=(None,)
shapes_dimensionality2=((2,),(0,0))
range_min2 = ((-1,1),)
range_max2 = ((-1,100),)
device_parame_array2 = [Device.device_params_CPU,Device.device_params_GPU0]

generate_datasets_with_one_dimensionality_changing(device_parame_array=device_parame_array2,count=count2,shape_dimensionality=shapes_dimensionality2,range_min=range_min2,range_max=range_max2,function_dict = function_dict,min_shapes=min_shapes,max_shapes=max_shapes,sampling=sampling,force_shape_relation=force_shape_relation2,dtype=dtype,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level,fold_path=fold_path,device_name=device_name,show_print=show_print)

# 研究三维加法
count3 = 7
force_shape_relation3=(None,)
shapes_dimensionality3=((3,),(0,0))
range_min3 = ((-1,1,1),)
range_max3 = ((-1,100,100),)
device_parame_array3 = [Device.device_params_CPU,Device.device_params_GPU0]

generate_datasets_with_one_dimensionality_changing(device_parame_array=device_parame_array3,count=count3,shape_dimensionality=shapes_dimensionality3,range_min=range_min3,range_max=range_max3,function_dict = function_dict,min_shapes=min_shapes,max_shapes=max_shapes,sampling=sampling,force_shape_relation=force_shape_relation3,dtype=dtype,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level,fold_path=fold_path,device_name=device_name,show_print=show_print)