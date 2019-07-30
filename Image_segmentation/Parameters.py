#coding=utf-8
import numpy as np

#一次性训练的个数
batch_size=100

#图片大小
pic_size=512

#输入图片
x_size=pic_size*pic_size*3
#密度图大小
y_size=pic_size*pic_size

#测试集数量
x_num=120

#原图缩小倍数
multriple=8

end_pic_size=int(pic_size/multriple)



