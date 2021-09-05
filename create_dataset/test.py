from create_dataset.common import generate_dataset_with_one_dimensionality_changing,Device
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
    y = relay.var("input_y", shape=dshape[1], dtype=dtype)
    f = relay.add(x, y)

    return f

# 定义参数
function_dict = {"func":calculate_op, "name": "add"}
shapes=((-1,40),(0,0))
min_shapes=1
max_shapes=100
sampling=0.25
# force_shape_relation=(None,(lambda x,y,z:x, lambda x,y,z:y, lambda x,y,z:z))
force_shape_relation=(None,(lambda x,y:x, lambda x,y:y))
dtype="float32"

cycle_times=20
min_repeat_ms=30
opt_level=0
fold_path="create_dataset/datasets/"
device_name="dell04"
show_print=True
# log_file默认值.

generate_dataset_with_one_dimensionality_changing(device_parames=Device.device_params_CPU,function_dict = function_dict,shapes=shapes,min_shapes=min_shapes,max_shapes=max_shapes,sampling=sampling,force_shape_relation=force_shape_relation,dtype=dtype,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level,fold_path=fold_path,device_name=device_name,show_print=show_print)
generate_dataset_with_one_dimensionality_changing(device_parames=Device.device_params_GPU0,function_dict = function_dict,shapes=shapes,min_shapes=min_shapes,max_shapes=max_shapes,sampling=sampling,force_shape_relation=force_shape_relation,dtype=dtype,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level,fold_path=fold_path,device_name=device_name,show_print=show_print)
