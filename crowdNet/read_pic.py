# -*-coding:utf-8-*-
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from crowdNet import Parameters

#读取数据
def Read_Data(pic_path,density_map_path):
    pics=[]
    for i in range(1,Parameters.x_num+1):
        print(i)
        s=pic_path+str(i)+".jpg"
        pics.append(mpimg.imread(s).reshape(-1))

    density = []
    for i in range(1,Parameters.x_num+1):
        print(i)
        s=density_map_path+str(i)+".txt"
        density.append(np.loadtxt(s, delimiter=",").reshape(-1))
    return np.array(pics), np.array(density)

#读取数据，有范围
def Read_Data2(pic_path,density_map_path,num):
    pics=[]
    for i in range(1,num+1):
        print(i)
        s=pic_path+str(i)+".jpg"
        pics.append(mpimg.imread(s).reshape(-1))

    density = []
    for i in range(1,num+1):
        print(i)
        s=density_map_path+str(i)+".txt"
        density.append(np.loadtxt(s, delimiter=",").reshape(-1))
    return np.array(pics), np.array(density)

#读取数据，有范围
def Read_Data3(pic_path,density_map_path,readlow,readhigh):
    pics=[]
    for i in range(readlow,readhigh+1):
        #print("image:\t" + str(i))
        s=pic_path+str(i)+".jpg"
        pics.append(mpimg.imread(s).reshape(-1))

    density = []
    for i in range(readlow,readhigh+1):
        #print("density:\t" + str(i))
        s=density_map_path+str(i)+".txt"
        density.append(np.loadtxt(s, delimiter=",").reshape(-1))
    return np.array(pics), np.array(density)

#读取数据，有范围
def Read_Data4(pic_path,density_map_path,readlow,readhigh,pics, density):
    index=0
    for i in range(readlow,readhigh+1):
        #print("image:\t" + str(i))
        s=pic_path+str(i)+".jpg"
        pics[index]=np.array(mpimg.imread(s).reshape(-1))
        index+=1

    index=0
    for i in range(readlow,readhigh+1):
        #print("density:\t" + str(i))
        s=density_map_path+str(i)+".txt"
        density[index]=np.array(np.loadtxt(s, delimiter=",").reshape(-1))
        index+=1


def get_batch_data(pics, density):
    images = tf.cast(pics, tf.float64)
    label = tf.cast(density, tf.float64)
    input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
    image_batch, density_batch = tf.train.batch(input_queue, batch_size=64, num_threads=1, capacity=64, allow_smaller_final_batch=False)
    return image_batch, density_batch

#生成batch
def get_batch_buffer(pics, density, iter, batch_size):
    low=(iter*batch_size) % len(pics)
    high=low+batch_size
    if(high>len(pics)):
        rest=high-len(pics)
        restPic=pics[0:rest,:]
        restDen=density[0:rest,:]

        p=pics[low:high,:]
        d=density[low:high,:]

        return np.vstack([p,restPic]),np.vstack([d,restDen])
    #print(low,high, len(pics))
    return pics[low:high,:], density[low:high,:]

#读取测试集
def Read_Data_test(pic_path,density_map_path):
    pics = []
    for i in range(1, 95):
        print(i)
        s = pic_path + str(i) + ".jpg"
        pics.append(mpimg.imread(s).reshape(-1))

    density = []
    for i in range(1, 95):
        print(i)
        s = density_map_path + str(i) + ".txt"
        density.append(np.loadtxt(s, delimiter=",").reshape(-1))
    return np.array(pics), np.array(density)

#读取测试集,有范围
def Read_Data_test2(pic_path,density_map_path,num):
    pics = []
    for i in range(1, num+1):
        print(i)
        s = pic_path + str(i) + ".jpg"
        pics.append(mpimg.imread(s).reshape(-1))

    density = []
    for i in range(1, num+1):
        print(i)
        s = density_map_path + str(i) + ".txt"
        density.append(np.loadtxt(s, delimiter=",").reshape(-1))
    return np.array(pics), np.array(density)


#读取测试集,有范围
def Read_Data_test3(pic_path,num):
    pics = []
    for i in range(1, num+1):
        print(i)
        s = pic_path + str(i) + ".jpg"
        pics.append(mpimg.imread(s).reshape(-1))
    return np.array(pics)


#读取测试集,按索引读取一张
def Read_Data_test4(pic_path,Index):
    pics = []
    print(Index)
    s = pic_path + str(Index) + ".jpg"
    pics.append(mpimg.imread(s).reshape(-1))
    return np.array(pics)

#原始数据切割（无重叠）


"""
sess_config = tf.ConfigProto()
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.90
sess = tf.InteractiveSession(config=sess_config)


pic_path='/home/idiot/PycharmProjects/preProcess_crowdNet/128_pic_set/'
density_map_path='/home/idiot/PycharmProjects/preProcess_crowdNet/128_density_set/'

pics, density=Read_Data(pic_path,density_map_path)
print(type(pics))
print(pics[0])

tempPic=pics[2:6, :]
print(len(tempPic),len(tempPic[0]))

image_batch, label_batch=get_batch_buffer(pics, density, 3, 64)

print(image_batch)
"""

