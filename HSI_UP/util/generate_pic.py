# -*- coding:utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文

axes = plt.gca()
axes.set_ylim([75, 101])

# 构建数据
width = 0.3
x = np.arange(3) + 1 + width
mode = "AA"
if mode is "OA":
         # None   SE    SPA
    y = [95.75, 96.36, 96.51]
    y2 = [94.73, 94.78, 96.48]
    y3 = [95.21, 95.37, 97.18]
elif mode is "AA":
    y = [88.19, 91.30, 93.30]
    y2 = [91.36, 91.17, 94.25]
    y3 = [97.33, 97.50, 98.44]
else:
    y = [95.16, 95.85, 96.02]
    y2 = [93.01, 93.07, 95.34]
    y3 = [94.70, 94.85, 96.87]

# 绘图
plt.bar(x=x - width, height=y, width=width, label='IP', alpha=0.8)
plt.bar(x=x, height=y2, width=width, label='UP', alpha=0.8)
plt.bar(x=x + width, height=y3, width=width, label='SV', alpha=0.8)
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 15,
         }
plt.xticks(x, ['None', 'SE', 'SPA'])

# plt.tick_params(labelsize=15)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
for x1, yy in zip(x, y):
    plt.text(x1 - width, yy + 1, str(yy), ha='center', va='bottom', fontsize=13,
             rotation=0, weight="extra bold", fontdict=font1)
for x1, yy in zip(x, y2):
    plt.text(x1, yy + 1, str(yy), ha='center', va='bottom', fontsize=13,
             rotation=0, weight="extra bold", fontdict=font1)
for x1, yy in zip(x, y3):
    plt.text(x1 + width, yy + 1, str(yy), ha='center', va='bottom', fontsize=13,
             rotation=0, weight="extra bold", fontdict=font1)
# plt.grid(axis="y")
# 设置标题
plt.title("")
# 为两条坐标轴设置名称
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}
plt.xlabel("Different core layers", fontdict = font2)
plt.ylabel("%s %%" % mode, fontdict = font2)
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xticks(fontproperties = 'Times New Roman', size = 16)
# 显示图例
plt.legend(loc=5, prop=font1)
plt.savefig('./pic/%s.png' % mode, dpi=300, bbox_inches = 'tight')
plt.show()
