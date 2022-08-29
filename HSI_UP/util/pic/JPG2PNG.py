import PIL.Image
import os

i = 0
path = r"D:/temp/Figure/Figure8/"
savepath = r"D:/temp/Figure/Figure8/pngformat/"
filelist = os.listdir(path)
for file in filelist:
    im = PIL.Image.open(path + filelist[i])
    filename = os.path.splitext(file)[0]
    im.save(savepath + filename + '.png')  # or 'test.tif'
    i = i + 1


# path = r"D:/temp/Figure/Figure9/"
# filelist = os.listdir(path)
# for file in filelist:
#     # print(os.path.join(path, file))
#     # print(os.path.join(path, file.split(".")[0] + ".jpg"))
#     os.rename(os.path.join(path, file), os.path.join(path, file.split(".")[0] + ".jpg"))
