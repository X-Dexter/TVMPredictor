import matplotlib.pyplot as plt

f=open("add_float.txt","r")

datas1=[]
line = f.readline()
while line is not None and len(line)>0 :
    datas1.append(float(line.split(",")[1]))
    line=f.readline()

f.close()


f=open("mul_mat_float.txt","r")

datas2=[]
line = f.readline()
while line is not None and len(line)>0 :
    datas2.append(float(line.split(",")[1]))
    line=f.readline()

f.close()


plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时'-'显示为方块的问题
 
plt.figure(1)
add_img = plt.subplot(1, 2, 1)
plt.plot(datas1)
add_img.set_title("add-op")

mul_img = plt.subplot(1, 2, 2)
plt.plot(datas2)
mul_img.set_title("mul-op")

plt.show()