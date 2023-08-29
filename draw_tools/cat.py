import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import os

filelists = ['average.png', 'OxfordPets.png', 'Flowers102.png', 'FGVCAircraft.png', 'StanfordCars.png', 'DTD.png',
             'EuroSAT.png', 'Food101.png', 'SUN397.png', 'UCF101.png', 'Caltech101.png', 'ImageNet.png']

file_name = "main_curves"  ## 存放图片文件夹的名称(仅需这里修改)
# file_path = os.path.abspath(file_name)
# filelists = os.listdir(file_path)
print(filelists)
m, n = 4, 3
assert m * n == len(filelists)
# while m*n != len(filelists):
#     m,n = map(int,input("请输入您要排版的行数与列数(以空格隔开)：").split(" "))
# img_name = input("请输入要保存图片的名称:")
img_name = 'main'
img_list = []
for i in filelists:
    img_list.append(mping.imread(f"./{file_name}/{i}"))
img_temp = []
for i in range(0, m * n, n):
    img_temp.append(np.concatenate(img_list[i:i + n], axis=1))
img_end = np.concatenate(img_temp, axis=0)

mping.imsave(f"{img_name}.png", img_end)
