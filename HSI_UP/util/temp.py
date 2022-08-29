# --coding:utf-8--
import matplotlib.pyplot as plt

# 数据设置
# IP or UP or SV
mode = "IP"
if mode is "IP":
    x = [5, 10, 15, 20, 25]
    y_DBMA = [93.20, 96.55,	97.59, 98.25, 98.75]
    y_DBDA = [97.33, 98.38, 98.88, 99.29, 99.53]
    y_SSRN = [96.49, 96.83, 99.45, 99.69, 99.72]
    y_FDSSC = [97.80, 98.38, 98.88, 99.53, 99.85]
    y_S3EResBoF = [94.24, 97.17, 97.47, 99.35, 99.42]
    y_SSUN = [94.62, 97.40, 98.78, 99.00, 99.75]
    y_pResNet = [94.08, 98.10, 99.20, 99.21, 99.65]
    y_LSSAN = [96.51, 98.58, 99.39, 99.69, 99.76]
elif mode is "UP":
    x = [1, 3, 5,  7, 10]
    y_DBMA = [92.74, 97.90, 98.14, 98.55, 98.88]
    y_DBDA = [96.50, 99.24, 99.48, 99.73, 99.87]
    y_SSRN = [97.38, 99.60, 99.75, 99.83, 99.90]
    y_FDSSC = [98.21, 99.65, 99.73, 99.90, 99.93]
    y_S3EResBoF = [94.24, 98.69, 99.29, 99.51, 99.56]
    y_SSUN = [94.64, 99.03, 99.26, 99.68, 99.74]
    y_pResNet = [94.79, 99.08, 99.45, 99.59, 99.84]
    y_LSSAN = [96.48, 98.84, 99.34, 99.75, 99.77]
elif mode is "SV":
    x = [1, 3, 5,  7, 10]
    y_DBMA = [93.21, 94.80, 96.09, 96.37, 97.87]
    y_DBDA = [97.62, 98.02, 98.85, 99.13, 99.98]
    y_SSRN = [94.42, 98.46, 99.75, 99.88, 99.93]
    y_FDSSC = [97.92, 99.59, 99.80, 99.91, 99.95]
    y_S3EResBoF = [97.42, 98.87, 99.86, 99.87, 99.91]
    y_SSUN = [96.05, 98.89, 99.76, 99.80, 99.93]
    y_pResNet = [96.66, 99.48, 99.59, 99.88, 99.91]
    y_LSSAN = [97.18, 99.13, 99.67, 99.71, 99.96]
else:
    exit()


# 设置输出的图片大小
figsize = 11, 9
figure, ax = plt.subplots(figsize=figsize)

# 在同一幅图片上画两条折线
DBMA, = plt.plot(x, y_DBMA, color="#1E90FF", linestyle="-", marker="D", markersize=15, label='DBMA', linewidth=5.0)
DBDA, = plt.plot(x, y_DBDA, color="#828282", linestyle="-.", marker="s", markersize=15, label='DBDA', linewidth=5.0)
SSRN, = plt.plot(x, y_SSRN, color="#FF6347", linestyle="--", marker="v", markersize=15, label='SSRN', linewidth=5.0)
FDSSC, = plt.plot(x, y_FDSSC, color="#66CD00", linestyle=":", marker="^", markersize=15, label='FDSSC', linewidth=5.0)
S3EResBoF, = plt.plot(x, y_S3EResBoF, color="#FFDAB9", linestyle="-", marker=">", markersize=15, label='S3EResBoF', linewidth=5.0)
SSUN, = plt.plot(x, y_SSUN, color="#FFA54F", linestyle="--", marker="<", markersize=15, label='SSUN', linewidth=5.0)
pResNet, = plt.plot(x, y_pResNet, color="#8B0000", linestyle=":", marker="d", markersize=15, label='pResNet', linewidth=5.0)
LSSAN, = plt.plot(x, y_LSSAN, color="#CD0000", linestyle="-", marker="o", markersize=15, label='LSSAN', linewidth=5.0)


# 设置图例并且设置图例的字体及大小w
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 25,
         }
legend = plt.legend(handles=[DBMA, DBDA, SSRN, FDSSC, S3EResBoF, SSUN, pResNet, LSSAN], prop=font1)
# legend = plt.legend(handles=[DBMA, DBDA, SSRN, FDSSC, S3EResBoF, SSUN, pResNet, ], prop=font1)

# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=30)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 40,
         }
plt.xlabel('Training percent %', font2)
plt.ylabel('OA %', font2)
plt.grid(axis="y")
# 将文件保存至文件中并且画出图
plt.savefig('./pic/%s_LSSAN_.png' % mode, dpi=300, bbox_inches = 'tight')
plt.show()