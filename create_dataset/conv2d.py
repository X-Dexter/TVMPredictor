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
    data = relay.var("input_x", shape=dshape[0], dtype=dtype)
    weight = relay.var("weight", shape=dshape[1], dtype=dtype)
    f = relay.nn.conv2d(data=data,weight=weight,channels=32, kernel_size=(3, 3), strides=(2, 2),padding=(1, 1))

    return f

# 定义参数
function_dict = {"func":calculate_op, "name": "conv2d"}
min_shapes=10
max_shapes=1200
sampling=0.1
dtype="float32"

cycle_times=20
min_repeat_ms=30
opt_level=0
fold_path="create_dataset/datasets/"
device_name="dell04"
show_print=True
# log_file默认值.

# # 研究二维乘法
# count2 = 7

# force_shape_relation2=(None,(lambda x,y:x, lambda x,y:y))
# shapes_dimensionality2=((2,2),(0,0))
# range_min2 = ((-1,1),(1,1))
# range_max2 = ((-1,100),(100,100))
# device_parame_array2 = [Device.device_params_CPU,Device.device_params_GPU0]

# generate_datasets_with_one_dimensionality_changing(device_parame_array=device_parame_array2,count=count2,shape_dimensionality=shapes_dimensionality2,range_min=range_min2,range_max=range_max2,function_dict = function_dict,min_shapes=min_shapes,max_shapes=max_shapes,sampling=sampling,force_shape_relation=force_shape_relation2,dtype=dtype,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level,fold_path=fold_path,device_name=device_name,show_print=show_print)

# 研究三维卷积
count3 = 7
force_shape_relation3=(None,None)
shapes_dimensionality3=((4,4),(0,0))
device_parame_array3 = [Device.device_params_CPU,Device.device_params_GPU0]

# shape_ = ((28,28),(32,32),(64,64),(128,128),(156,156),(224,224),(256,256))
# for i in range(7):
#     range_min3 = ((-1,3,*shape_[i]),(32,3,3,3))
#     range_max3 = ((-1,3,*shape_[i]),(32,3,3,3))
#     generate_datasets_with_one_dimensionality_changing(device_parame_array=device_parame_array3,count=i+1,shape_dimensionality=shapes_dimensionality3,range_min=range_min3,range_max=range_max3,function_dict = function_dict,min_shapes=min_shapes,max_shapes=max_shapes,sampling=sampling,force_shape_relation=force_shape_relation3,dtype=dtype,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level,fold_path=fold_path,device_name=device_name,show_print=show_print)

range_min3 = ((-1,3,28,28),(32,3,3,3))
range_max3 = ((-1,3,28,28),(32,3,3,3))
generate_datasets_with_one_dimensionality_changing(device_parame_array=device_parame_array3,count=6,shape_dimensionality=shapes_dimensionality3,range_min=range_min3,range_max=range_max3,function_dict = function_dict,min_shapes=min_shapes,max_shapes=max_shapes,sampling=sampling,force_shape_relation=force_shape_relation3,dtype=dtype,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level,fold_path=fold_path,device_name=device_name,show_print=show_print)

range_min3 = ((-1,3,32,32),(32,3,3,3))
range_max3 = ((-1,3,32,32),(32,3,3,3))
generate_datasets_with_one_dimensionality_changing(device_parame_array=device_parame_array3,count=7,shape_dimensionality=shapes_dimensionality3,range_min=range_min3,range_max=range_max3,function_dict = function_dict,min_shapes=min_shapes,max_shapes=max_shapes,sampling=sampling,force_shape_relation=force_shape_relation3,dtype=dtype,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level,fold_path=fold_path,device_name=device_name,show_print=show_print)