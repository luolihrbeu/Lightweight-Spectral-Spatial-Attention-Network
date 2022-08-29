import cv2
import os
def resize():
    image_size_h = 2000  # 设定宽
    image_size_w = 2000  # 设定长
    source_path = "D:\HSI_Classification\HSI_UP\Fig\IP/"  # 源文件路径
    target_path = "D:\HSI_Classification\HSI_UP\Fig\Resize_IP/"  # 输出目标文件路径

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    image_list = os.listdir(source_path)  # 获得文件名

    for file in image_list:
        file_name = os.path.join(source_path, file)
        image_source = cv2.imread(file_name)  # 读取图片
        image = cv2.resize(image_source, (image_size_w, image_size_h), 0, 0)  # 修改尺寸
        cv2.imwrite(target_path + file + ".png", image)  # 重命名并且保存
    print("批量处理完成")

def resize_singal():
    source_path = "D:\HSI_Classification\HSI_UP\Fig\Ground-truth_IP.png"  # 源文件路径
    target_path = "D:\HSI_Classification\HSI_UP\Fig\Resize_IP/"  # 输出目标文件路径
    image_source = cv2.imread(source_path)  # 读取图片
    image = cv2.resize(image_source, (1300, 1300), 0, 0)  # 修改尺寸
    cv2.imwrite(target_path + "Ground-truth_IP" + ".png", image)  # 重命名并且保存

if __name__ == '__main__':
    resize_singal()