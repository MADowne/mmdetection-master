# coding=utf-8
import os.path
import cv2                   #导入opencv库

path = 'data/VOCdevkit/VOC2012/JPEGImages'
# 得到文件夹下所有文件名称
pics = os.listdir(path)
i = 1

for pic_name in pics:  # 遍历文件夹
    #print('修改第' + str(i) + '个pic' + ' 名字是:' + pic_name)
    i = i + 1
    # 得到一个pic完整的路径
    pic_path = os.path.join(path, pic_name)

    img1 = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)  # 读取图片，第二个参数表示以灰度图像读入
    if img1 is None:  # 判断读入的img1是否为空，为空就继续下一轮循环
        print(img1,pic_path,'这是一张坏图')

print('done')