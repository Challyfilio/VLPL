import os
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from pylab import *
from matplotlib import rcParams

font_set = FontProperties(fname=r"SongNTR.ttf", size=6)
font_set_1 = FontProperties(fname=r"SongNTR.ttf", size=10)

myfont = matplotlib.font_manager.FontProperties(
    fname=r'SongNTR.ttf')

save_dir = "main_curves"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

path = "Results.xlsx"  # this is the excel file containing the results (like the one we released)
file = pd.read_excel(path, sheet_name="imcls_fewshot")

config = {
    "font.family": 'myfont',
    "font.size": 18,
    "mathtext.fontset": 'stix',
    # "font.serif": ['SongNTR'],
}
rcParams.update(config)


def autolabel(rects):
    for rect in rects:
        w = rect.get_width()
        print(w)
        if w >= 0:
            plt.text(w, rect.get_y() + rect.get_height() / 2, '+%s' % float(w), ha='left', va='center', fontsize=11)
        else:
            plt.text(w, rect.get_y() + rect.get_height() / 2, '%s' % float(w), ha='right', va='center', fontsize=11)


def autolabel_1(rects):
    for rect in rects:
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.16, 1.005 * h, '%s' % float(h))


def Draw_hbar(coop, ptve):
    # font = {'family': 'Times New Roman',
    #         'size': 16,
    #         }
    # # sns.set(font_scale=1.2)
    # # sns.set_theme(style="white")
    # plt.rc('font', family='Times New Roman')

    config = {
        "font.family": 'serif',
        "font.size": 18,
        "mathtext.fontset": 'stix',
        # "font.serif": ['SongNTR'],
    }
    rcParams.update(config)

    plt.figure(figsize=(12, 8))

    autolabel(plt.barh(np.arange(10) + 0.15, coop, color='lightgray', height=0.3, label='CoOp'))
    autolabel(plt.barh(np.arange(10) - 0.15, ptve, color='dimgray', height=0.3, label='PTVE'))
    # plt.barh(range(len(ptve)), ptve, color='dimgray', label='PTVE')

    # plt.xlim((-10, 10))
    plt.xticks((-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50),
               ('-5', '0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'),
               font={'size': 14})
    plt.yticks((9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
               ('EuroSAT', 'Flowers102', 'DTD', 'FGVCAircraft', 'StanfordCars', 'UCF101', 'SUN397', 'Caltech101',
                'OxfordPets', 'Food101'),
               font={'size': 15})
    plt.xlabel('性能提升（%）', fontdict={'size': 20}, fontsize=18, fontfamily='sans-serif', fontstyle='italic',
               fontproperties=font_set)
    plt.ylabel('数据集', fontdict={'size': 20}, fontsize=18, fontfamily='sans-serif', fontstyle='italic',
               fontproperties=font_set)
    plt.title("CLIP + CoOp (M=16 , end)  vs  PTVE",
              font={'size': 20})
    plt.legend(fontsize=16)

    plt.show()
    plt.savefig('vs.png')


def Draw_bar(clipz, ptve):
    config = {
        "font.family": 'serif',
        "font.size": 18,
        "mathtext.fontset": 'stix',
        # "font.serif": ['SongNTR'],
    }
    rcParams.update(config)

    plt.figure(figsize=(12, 8))

    autolabel_1(plt.bar(np.arange(4) - 0.15, clipz, color='lightgray', width=0.3, label='Zero-shot CLIP'))
    autolabel_1(plt.bar(np.arange(4) + 0.15, ptve, color='dimgray', width=0.3, label='PTVE'))
    # plt.barh(range(len(ptve)), ptve, color='dimgray', label='PTVE')

    plt.ylim((50, 80))
    plt.xticks((0, 1, 2, 3),
               ('ResNet-50', 'ResNet-101', 'ViT-B/32', 'ViT-B/16'),
               font={'size': 20})
    plt.yticks((55, 60, 65, 70, 75),
               ('55', '60', '65', '70', '75'),
               font={'size': 20})

    plt.xlabel('视觉主干网络', fontdict={'size': 20}, fontsize=24, fontfamily='sans-serif', fontstyle='italic',
               fontproperties=font_set)
    plt.ylabel('分数（%）', fontdict={'size': 20}, fontsize=24, fontfamily='sans-serif', fontstyle='italic',
               fontproperties=font_set)
    plt.title("Zero-shot CLIP  vs  PTVE",
              font={'size': 20})
    plt.legend(fontsize=20)

    plt.show()
    plt.savefig('vs.png')


if __name__ == "__main__":
    # Draw_hbar(np.array([-2.64, 1.24, 5.54, 10.74, 14.25, 17.75, 13.98, 21.26, 28.37, 45.97]),  # coop
    #           np.array([-4.73, 1.10, 6.07, 10.60, 14.96, 17.94, 20.49, 22.14, 28.89, 46.53])  # ptve
    #           )

    Draw_bar(np.array([58.77, 59.86, 61.88, 65.23]),  # clipz
             np.array([62.46, 63.67, 65.38, 70.51])  # ptve
             )
