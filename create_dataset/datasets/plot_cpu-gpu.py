import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fold_name="create_dataset/datasets/dell04/"

# add-op
f=open(fold_name+"add_float.txt","r")

add_datas_cpu=[]
line = f.readline()
while line is not None and len(line)>0 :
    add_datas_cpu.append(float(line.split(",")[1]))
    line=f.readline()

f.close()


f=open(fold_name+"add_float_gpu.txt","r")

add_datas_gpu=[]
line = f.readline()
while line is not None and len(line)>0 :
    add_datas_gpu.append(float(line.split(",")[1]))
    line=f.readline()

f.close()

# mul_mat-op
f=open(fold_name+"mul_mat_float.txt","r")

mul_mat_datas_cpu=[]
line = f.readline()
while line is not None and len(line)>0 :
    mul_mat_datas_cpu.append(float(line.split(",")[1]))
    line=f.readline()

f.close()


f=open(fold_name+"mul_mat_float_gpu.txt","r")

mul_mat_datas_gpu=[]
line = f.readline()
while line is not None and len(line)>0 :
    mul_mat_datas_gpu.append(float(line.split(",")[1]))
    line=f.readline()

f.close()

# 开始画图
plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时'-'显示为方块的问题

plt.figure(1)

add_cpu_div_gpu=[]
for i in range(len(add_datas_cpu)):
    add_cpu_div_gpu.append(add_datas_cpu[i]/add_datas_gpu[i])
add_standard=[1 for i in add_cpu_div_gpu]

mul_mat_cpu_div_gpu=[]
for i in range(len(mul_mat_datas_cpu)):
    mul_mat_cpu_div_gpu.append(mul_mat_datas_cpu[i]/mul_mat_datas_gpu[i])
mul_mat_standard=[1 for i in mul_mat_cpu_div_gpu]

# add-op
add_img = plt.subplot(2, 2, 1)
plt.plot(add_datas_cpu, color='green', label='cpu')
plt.plot(add_datas_gpu, color='red', label='gpu')
plt.legend() # 显示图例
plt.xlabel('input shapes')
plt.ylabel('run-time')
add_img.set_title("add-op")

# add-op-cdg
add_cdg_img = plt.subplot(2, 2, 3)
plt.plot(add_cpu_div_gpu, color='green', label='cpu/gpu')
plt.plot(add_standard, color='red', label='value=1')
plt.legend() # 显示图例
plt.xlabel('input shapes')
plt.ylabel('cpu/gpu')
add_cdg_img.set_title("add-op")


# mat_mul-op
mat_mul_img = plt.subplot(2, 2, 2)
plt.plot(mul_mat_datas_cpu, color='green', label='cpu')
plt.plot(mul_mat_datas_gpu, color='red', label='gpu')
plt.legend() # 显示图例
plt.xlabel('input shapes')
plt.ylabel('run-time')
mat_mul_img.set_title("mat_mul-op")

# mat_mul-op-cdg
mat_mul_cdg_img = plt.subplot(2, 2, 4)
plt.plot(mul_mat_cpu_div_gpu, color='green', label='cpu/gpu')
plt.plot(mul_mat_standard, color='red', label='value=1')
plt.legend() # 显示图例
plt.xlabel('input shapes')
plt.ylabel('cpu/gpu')
mat_mul_cdg_img.set_title("mat_mul-op")

plt.show()