#coding=utf-8
import tensorflow as tf
from crowdNet import read_pic
from crowdNet import interp
from crowdNet import Parameters


#import train data
path = "/media/public/Flying/segmentation/"
pic_path=path+str(Parameters.pic_size)+'_pic_set/'
density_map_path=path+str(Parameters.pic_size)+'_density_set/'
pics, density=read_pic.Read_Data2(pic_path, density_map_path,1126)


#import test data
pathTest = "/media/public/Flying/segmentation/"
test_pic_path=pathTest+str(Parameters.pic_size)+'_pic_set/'
test_density_map_path=pathTest+str(Parameters.pic_size)+'_density_set/'
test_pics, test_density=read_pic.Read_Data2(test_pic_path, test_density_map_path,100)


print("Data loading completed!!!")

upscale_factor=int(Parameters.pic_size/Parameters.end_pic_size)
kernel_size = 2*upscale_factor - upscale_factor%2
bilinear = interp.get_bilinear_filter([kernel_size,kernel_size,1,1,],upscale_factor)


sess_config = tf.ConfigProto()
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.90
sess = tf.InteractiveSession(config=sess_config)



# 训练数据
x = tf.placeholder(dtype=tf.float32, shape=[None, Parameters.x_size])
# 训练标签数据
y_ = tf.placeholder(dtype=tf.float32, shape=[None, Parameters.y_size])
# 把x更改为4维张量，第1维代表样本数量，第2维和第3维代表图像长宽， 第4维代表图像通道数, 1表示黑白
x_image = tf.reshape(x, [-1, Parameters.pic_size, Parameters.pic_size, 3])



#############
# Deep net  #
#############

#Convolution1_1
# 过滤器大小为3*3, 当前层深度为3， 过滤器的深度为8
conv1_1_weights = tf.get_variable("conv1_1_weights", [7, 7, 3, 8], initializer=tf.truncated_normal_initializer(mean=0.005,stddev=0.005))
conv1_1_biases = tf.get_variable("conv1_1_biases", [8], initializer=tf.constant_initializer(0.0))
# 移动步长为1, 使用全0填充
conv1_1 = tf.nn.conv2d(x_image, conv1_1_weights, strides=[1, 1, 1, 1], padding='SAME')
# 激活函数Relu去线性化
relu1_1 = tf.nn.relu(tf.nn.bias_add(conv1_1, conv1_1_biases))



# max_pool1
# 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
pool1 = tf.nn.avg_pool(relu1_1, ksize=[1, 15, 15, 1], strides=[1, 2, 2, 1], padding='SAME')


#Convolution2_1
# 过滤器大小为3*3, 当前层深度为8， 过滤器的深度为16
conv2_1_weights = tf.get_variable("conv2_1_weights", [7, 7, 8, 16], initializer=tf.truncated_normal_initializer(mean=0.005,stddev=0.005))
conv2_1_biases = tf.get_variable("conv2_1_biases", [16], initializer=tf.constant_initializer(0.0))
# 移动步长为1, 使用全0填充
conv2_1 = tf.nn.conv2d(pool1, conv2_1_weights, strides=[1, 1, 1, 1], padding='SAME')
# 激活函数Relu去线性化
relu2_1 = tf.nn.relu(tf.nn.bias_add(conv2_1, conv2_1_biases))


# max_pool2
# 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
pool2 = tf.nn.avg_pool(relu2_1, ksize=[1, 15, 15, 1], strides=[1, 2, 2, 1], padding='SAME')


#Convolution3_1
# 过滤器大小为3*3, 当前层深度为16， 过滤器的深度为8
conv3_1_weights = tf.get_variable("conv3_1_weights", [7, 7, 16, 8], initializer=tf.truncated_normal_initializer(mean=0.005,stddev=0.005))
conv3_1_biases = tf.get_variable("conv3_1_biases", [8], initializer=tf.constant_initializer(0.0))
# 移动步长为1, 使用全0填充
conv3_1 = tf.nn.conv2d(pool2, conv3_1_weights, strides=[1, 1, 1, 1], padding='SAME')
# 激活函数Relu去线性化
relu3_1 = tf.nn.relu(tf.nn.bias_add(conv3_1, conv3_1_biases))


# max_pool3
# 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
pool3 = tf.nn.avg_pool(relu3_1, ksize=[1, 15, 15, 1], strides=[1, 2, 2, 1], padding='SAME')



#1*1的卷积,第6层
conv6_weights = tf.get_variable("conv6_weights", [1, 1, 8, 1], initializer=tf.truncated_normal_initializer(mean=0.005,stddev=0.005), dtype=tf.float32)
conv6_biases = tf.get_variable("conv6_biases", [1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
# 移动步长为1, 使用全0填充
conv6 = tf.nn.conv2d(pool3, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
# 激活函数Relu去线性化
relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases))

relu6 = tf.cast(relu6, dtype='float64')





#进行双线性插值
bottom = tf.reshape(relu6, [-1, Parameters.end_pic_size, Parameters.end_pic_size, 1])
bottom=tf.cast(bottom,dtype='float32')
weights = tf.reshape(bilinear, [kernel_size, kernel_size, 1, 1, ])
weights=tf.cast(weights,dtype='float32')
stride = upscale_factor
strides = [1, stride, stride, 1]
deconv = tf.nn.conv2d_transpose(bottom, weights, [Parameters.batch_size, Parameters.pic_size, Parameters.pic_size, 1], strides, padding='SAME')


bilinear_reslut = tf.reshape(deconv, [-1, Parameters.y_size], name=None)
bilinear_reslut = tf.cast(bilinear_reslut, dtype='float32')



loss = tf.reduce_sum(y_ - bilinear_reslut)

# 选择优化器，并让优化器最小化损失函数/收敛, 反向传播
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)


#预测
#e=tf.constant(0.00001)
preNum=tf.reduce_sum(bilinear_reslut, 1)
realNum=tf.reduce_sum(y_, 1)
error=tf.abs(tf.subtract(preNum, realNum))
Probability=tf.subtract(1.0, tf.div(error, realNum))
accuracy = tf.reduce_mean(tf.cast(Probability, tf.float64))

MAE=tf.reduce_mean(error)
E=tf.subtract(preNum, realNum)



# 开始训练
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


for i in range(30):
    image_batch, density_batch=read_pic.get_batch_buffer(pics, density, i, Parameters.batch_size)
    Test_image_batch, Test_density_batch = read_pic.get_batch_buffer(test_pics, test_density, i, Parameters.batch_size)
    #batch = mnist.train.next_batch(100)
    if i % 1 == 0:
        l=loss.eval(feed_dict={x: image_batch, y_: density_batch})
        print ("loss:\t%f"%(l))
        print("step %d, \t MEA: %f" % (i, MAE.eval(feed_dict={x: image_batch, y_: density_batch})))
        print("step %d, \t MEA: %f" % (i,MAE.eval(feed_dict={x: Test_image_batch, y_: Test_density_batch})))
        print("step %d, E: %s" % (i, E.eval(feed_dict={x: image_batch, y_: density_batch})))
        print ("---------------------------------------------------------------------------------\n")
    train_step.run(feed_dict={x: image_batch, y_: density_batch})  # 训练阶段使用50%的Dropout

saver_path = saver.save(sess, "save/model.ckpt")  # 将模型保存到save/model.ckpt文件


#Parameters.batch_size=1
# 在测试数据上测试准确率
print("-----------------------------------------------------------------------------------")
