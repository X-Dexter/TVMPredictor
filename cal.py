# 临时脚本，探究一些规律
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

fold_name="create_dataset/datasets/own_way_abandon/dell04/"

# python
f=open(fold_name+"multi-add-1.txt","r")

python_calc=[]
line = f.readline()
while line is not None and len(line)>0 :
    python_calc.append(float(line.split(",")[1])/2000)
    line=f.readline()

f.close()

# C++
f=open(fold_name+"multi-add-2.txt","r")

cpp_calc=[]
line = f.readline()
while line is not None and len(line)>0 :
    cpp_calc.append(float(line.split(",")[1]))
    line=f.readline()

f.close()

loss_calc = []
for i in range(len(python_calc)):
    loss_calc.append(abs(python_calc[i]-cpp_calc[i]))

# 开始画图
plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时'-'显示为方块的问题

plt.figure(1)

# add-op
add_img = plt.subplot(2, 1, 1)
plt.plot(python_calc, color='green', label='python')
plt.plot(cpp_calc, color='red', label='C++')
plt.legend() # 显示图例
plt.xlabel('input shapes')
plt.ylabel('run-time  (us)')
add_img.set_title("add-op")

# add-op-cdg
loss_img = plt.subplot(2, 1, 2)
plt.plot(loss_calc, color='black')
plt.legend() # 显示图例
plt.xlabel('input shapes')
plt.ylabel('(|cpp_time-py_time|  (us)')
loss_img.set_title("loss")

plt.show()