from operation_statistics.cnn_workload_generator import get_network
from create_dataset.common import generate_datasets_with_one_dimensionality_changing,Device

def calculate_inception_v3(dshape,dtype="float32"):
    '''
    test add-op in one kind of shape.

    Parameters
    ----------

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)
    '''

    return get_network("inception_v3", input_shape=dshape[0], dtype=dtype)[0]

def calculate_mobilenet(dshape,dtype="float32"):
    '''
    test add-op in one kind of shape.

    Parameters
    ----------

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)
    '''

    return get_network("mobilenet", input_shape=dshape[0], dtype=dtype)[0]

def calculate_squeezenet_v1_1(dshape,dtype="float32"):
    '''
    test add-op in one kind of shape.

    Parameters
    ----------

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)
    '''

    return get_network("squeezenet_v1.1", input_shape=dshape[0], dtype=dtype)[0]

def calculate_resnet_50(dshape,dtype="float32"):
    '''
    test add-op in one kind of shape.

    Parameters
    ----------

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)
    '''

    return get_network("resnet-50", input_shape=dshape[0], dtype=dtype)[0]

def calculate_resnet3d_50(dshape,dtype="float32"):
    '''
    test add-op in one kind of shape.

    Parameters
    ----------

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)
    '''

    return get_network("resnet3d-50", input_shape=dshape[0], dtype=dtype)[0]

min_shapes=1
max_shapes=201
sampling=0.1
dtype="float32"

isModule=True
cycle_times=3
min_repeat_ms=30
opt_level=0
fold_path="create_dataset/datasets/"
device_name="dell04"
show_print=True

count3 = 1
force_shape_relation3=(None,)
shapes_dimensionality3=((4,),(0,0))
device_parame_array3 = [Device.device_params_GPU0,Device.device_params_CPU]

range_min3 = ((-1,3,299,299),)
range_max3 = ((-1,3,299,299),)
function_dict = {"func":calculate_inception_v3, "name": "inception_v3"}
generate_datasets_with_one_dimensionality_changing(device_parame_array=device_parame_array3,count=count3,shape_dimensionality=shapes_dimensionality3,range_min=range_min3,range_max=range_max3,function_dict = function_dict,min_shapes=min_shapes,max_shapes=max_shapes,sampling=sampling,force_shape_relation=force_shape_relation3,dtype=dtype,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level,fold_path=fold_path,device_name=device_name,show_print=show_print,isModule=isModule)

range_min3 = ((-1,3,224,224),)
range_max3 = ((-1,3,224,224),)
function_dict = {"func":calculate_mobilenet, "name": "mobilenet"}
generate_datasets_with_one_dimensionality_changing(device_parame_array=device_parame_array3,count=count3,shape_dimensionality=shapes_dimensionality3,range_min=range_min3,range_max=range_max3,function_dict = function_dict,min_shapes=min_shapes,max_shapes=max_shapes,sampling=sampling,force_shape_relation=force_shape_relation3,dtype=dtype,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level,fold_path=fold_path,device_name=device_name,show_print=show_print,isModule=isModule)

function_dict = {"func":calculate_squeezenet_v1_1, "name": "squeezenet_v1_1"}
generate_datasets_with_one_dimensionality_changing(device_parame_array=device_parame_array3,count=count3,shape_dimensionality=shapes_dimensionality3,range_min=range_min3,range_max=range_max3,function_dict = function_dict,min_shapes=min_shapes,max_shapes=max_shapes,sampling=sampling,force_shape_relation=force_shape_relation3,dtype=dtype,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level,fold_path=fold_path,device_name=device_name,show_print=show_print,isModule=isModule)

function_dict = {"func":calculate_resnet_50, "name": "resnet-50"}
generate_datasets_with_one_dimensionality_changing(device_parame_array=device_parame_array3,count=count3,shape_dimensionality=shapes_dimensionality3,range_min=range_min3,range_max=range_max3,function_dict = function_dict,min_shapes=min_shapes,max_shapes=max_shapes,sampling=sampling,force_shape_relation=force_shape_relation3,dtype=dtype,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level,fold_path=fold_path,device_name=device_name,show_print=show_print,isModule=isModule)

function_dict = {"func":calculate_resnet3d_50, "name": "resnet3d-50"}
generate_datasets_with_one_dimensionality_changing(device_parame_array=device_parame_array3,count=count3,shape_dimensionality=shapes_dimensionality3,range_min=range_min3,range_max=range_max3,function_dict = function_dict,min_shapes=min_shapes,max_shapes=max_shapes,sampling=sampling,force_shape_relation=force_shape_relation3,dtype=dtype,cycle_times=cycle_times,min_repeat_ms=min_repeat_ms,opt_level=opt_level,fold_path=fold_path,device_name=device_name,show_print=show_print,isModule=isModule)
