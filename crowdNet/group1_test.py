#coding=utf-8
import tensorflow as tf
from crowdNet import read_pic
from crowdNet import interp
from crowdNet import Parameters
import datetime
import os

def segmentation(rootpath,sess,j,modepath):
    path = rootpath + str(j) + "/"
    saveName = modepath

    fns = [os.path.join(root, fn) for root, dirs, files in os.walk(path) for fn in files]
    Parameters.batch_size = int(len(fns) / 2)

    # import test data
    test_pic_path = path + str(Parameters.pic_size) + '_pic_set/'
    test_pics = read_pic.Read_Data_test3(test_pic_path, Parameters.batch_size)
    print(test_pic_path)
    print("Data loading completed!!!")


    # 上采样准备
    upscale_factor = int(Parameters.pic_size / Parameters.end_pic_size)
    kernel_size = 2 * upscale_factor - upscale_factor % 2
    bilinear = interp.get_bilinear_filter([kernel_size, kernel_size, 1, 1, ], upscale_factor)

    # 训练数据
    x = tf.placeholder(dtype=tf.float32, shape=[None, Parameters.x_size])

    # 把x更改为4维张量，第1维代表样本数量，第2维和第3维代表图像长宽， 第4维代表图像通道数, 1表示黑白
    x_image = tf.reshape(x, [-1, Parameters.pic_size, Parameters.pic_size, 3])

    # Convolution1_1
    # 过滤器大小为3*3, 当前层深度为3， 过滤器的深度为64
    conv1_1_weights = tf.get_variable("conv1_1_weights", [7, 7, 3, 8],
                                      initializer=tf.truncated_normal_initializer(mean=0.005, stddev=0.005))
    conv1_1_biases = tf.get_variable("conv1_1_biases", [8], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv1_1 = tf.nn.conv2d(x_image, conv1_1_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu1_1 = tf.nn.relu(tf.nn.bias_add(conv1_1, conv1_1_biases))

    # max_pool1
    # 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
    pool1 = tf.nn.avg_pool(relu1_1, ksize=[1, 15, 15, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Convolution2_1
    # 过滤器大小为3*3, 当前层深度为64， 过滤器的深度为128
    conv2_1_weights = tf.get_variable("conv2_1_weights", [7, 7, 8, 16],
                                      initializer=tf.truncated_normal_initializer(mean=0.005, stddev=0.005))
    conv2_1_biases = tf.get_variable("conv2_1_biases", [16], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv2_1 = tf.nn.conv2d(pool1, conv2_1_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu2_1 = tf.nn.relu(tf.nn.bias_add(conv2_1, conv2_1_biases))

    # max_pool2
    # 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
    pool2 = tf.nn.avg_pool(relu2_1, ksize=[1, 15, 15, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Convolution3_1
    # 过滤器大小为3*3, 当前层深度为128， 过滤器的深度为256
    conv3_1_weights = tf.get_variable("conv3_1_weights", [7, 7, 16, 8],
                                      initializer=tf.truncated_normal_initializer(mean=0.005, stddev=0.005))
    conv3_1_biases = tf.get_variable("conv3_1_biases", [8], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv3_1 = tf.nn.conv2d(pool2, conv3_1_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu3_1 = tf.nn.relu(tf.nn.bias_add(conv3_1, conv3_1_biases))

    # max_pool3
    # 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
    pool3 = tf.nn.avg_pool(relu3_1, ksize=[1, 15, 15, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 1*1的卷积,第6层
    conv6_weights = tf.get_variable("conv6_weights", [1, 1, 8, 1],
                                    initializer=tf.truncated_normal_initializer(mean=0.005, stddev=0.005),
                                    dtype=tf.float32)
    conv6_biases = tf.get_variable("conv6_biases", [1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    # 移动步长为1, 使用全0填充
    conv6 = tf.nn.conv2d(pool3, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases))

    relu6 = tf.cast(relu6, dtype='float64')

    # 进行双线性插值
    bottom = tf.reshape(relu6, [-1, Parameters.end_pic_size, Parameters.end_pic_size, 1])
    bottom = tf.cast(bottom, dtype='float32')
    weights = tf.reshape(bilinear, [kernel_size, kernel_size, 1, 1, ])
    weights = tf.cast(weights, dtype='float32')
    stride = upscale_factor
    strides = [1, stride, stride, 1]
    deconv = tf.nn.conv2d_transpose(bottom, weights,
                                    [Parameters.batch_size, Parameters.pic_size, Parameters.pic_size, 1], strides,
                                    padding='SAME')

    bilinear_reslut = tf.reshape(deconv, [-1, Parameters.y_size], name=None)
    bilinear_reslut = tf.cast(bilinear_reslut, dtype='float32')

    # 预测
    preNum = tf.reduce_sum(bilinear_reslut)
    realNum = tf.constant(Parameters.realNum[j - 1])
    error = tf.abs(tf.subtract(preNum, realNum))
    Probability = tf.subtract(1.0, tf.div(error, realNum))
    accuracy = tf.reduce_mean(tf.cast(Probability, tf.float64))

    MAE = tf.reduce_mean(error)
    E = tf.subtract(preNum, realNum)

    # 开始训练
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        # load上一步训练的参数
        saver.restore(sess, saveName)
        segmentation_result = bilinear_reslut.eval(feed_dict={x: test_pics})
        return segmentation_result


def test(rootpath, Times,sess,j,modepath):
    path = rootpath +str(j) +"/"
    saveName = modepath + "group1_" + str(Times) + "/model.ckpt"

    starttime = datetime.datetime.now()
    fns = [os.path.join(root, fn) for root, dirs, files in os.walk(path) for fn in files]
    Parameters.batch_size = int(len(fns) / 2)

    # import test data
    test_pic_path = path + str(Parameters.pic_size) + '_pic_set/'
    test_pics = read_pic.Read_Data_test3(test_pic_path, Parameters.batch_size)
    print(test_pic_path)
    print("Data loading completed!!!")
    endtime = datetime.datetime.now()


    filename= 'group1' +'_result'
    with open(filename,'a') as fileobject:
        fileobject.write("---------------------------\t"+str(Times)+"\t--------------------------------\n")
        fileobject.write("the time of load data:\t"+str((endtime - starttime).seconds)+"\n")

    #上采样准备
    upscale_factor = int(Parameters.pic_size / Parameters.end_pic_size)
    kernel_size = 2 * upscale_factor - upscale_factor % 2
    bilinear = interp.get_bilinear_filter([kernel_size, kernel_size, 1, 1, ], upscale_factor)


    # 训练数据
    x = tf.placeholder(dtype=tf.float32, shape=[None, Parameters.x_size])

    # 把x更改为4维张量，第1维代表样本数量，第2维和第3维代表图像长宽， 第4维代表图像通道数, 1表示黑白
    x_image = tf.reshape(x, [-1, Parameters.pic_size, Parameters.pic_size, 3])

    #############
    # Deep net  #
    #############

    # Convolution1_1
    # 过滤器大小为3*3, 当前层深度为3， 过滤器的深度为8
    conv1_1_weights = tf.get_variable("conv1_1_weights", [3, 3, 3, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv1_1_biases = tf.get_variable("conv1_1_biases", [8], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv1_1 = tf.nn.conv2d(x_image, conv1_1_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu1_1 = tf.nn.relu(tf.nn.bias_add(conv1_1, conv1_1_biases))

    # Convolution1_2
    # 过滤器大小为3*3, 当前层深度为8， 过滤器的深度为8
    conv1_2_weights = tf.get_variable("conv1_2_weights", [3, 3, 8, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv1_2_biases = tf.get_variable("conv1_2_biases", [8], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv1_2 = tf.nn.conv2d(relu1_1, conv1_2_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu1_2 = tf.nn.relu(tf.nn.bias_add(conv1_2, conv1_2_biases))

    # max_pool1
    # 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
    pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Convolution2_1
    # 过滤器大小为3*3, 当前层深度为8， 过滤器的深度为16
    conv2_1_weights = tf.get_variable("conv2_1_weights", [3, 3, 8, 16],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv2_1_biases = tf.get_variable("conv2_1_biases", [16], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv2_1 = tf.nn.conv2d(pool1, conv2_1_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu2_1 = tf.nn.relu(tf.nn.bias_add(conv2_1, conv2_1_biases))

    # Convolution2_2
    # 过滤器大小为3*3, 当前层深度为16， 过滤器的深度为16
    conv2_2_weights = tf.get_variable("conv2_2_weights", [3, 3, 16, 16],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv2_2_biases = tf.get_variable("conv2_2_biases", [16], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv2_2 = tf.nn.conv2d(relu2_1, conv2_2_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu2_2 = tf.nn.relu(tf.nn.bias_add(conv2_2, conv2_2_biases))

    # max_pool2
    # 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
    pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Convolution3_1
    # 过滤器大小为3*3, 当前层深度为16， 过滤器的深度为8
    conv3_1_weights = tf.get_variable("conv3_1_weights", [3, 3, 16, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv3_1_biases = tf.get_variable("conv3_1_biases", [8], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv3_1 = tf.nn.conv2d(pool2, conv3_1_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu3_1 = tf.nn.relu(tf.nn.bias_add(conv3_1, conv3_1_biases))

    # Convolution3_2
    # 过滤器大小为3*3, 当前层深度为8， 过滤器的深度为8
    conv3_2_weights = tf.get_variable("conv3_2_weights", [3, 3, 8, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv3_2_biases = tf.get_variable("conv3_2_biases", [8], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv3_2 = tf.nn.conv2d(relu3_1, conv3_2_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu3_2 = tf.nn.relu(tf.nn.bias_add(conv3_2, conv3_2_biases))

    # Convolution3_3
    # 过滤器大小为3*3, 当前层深度为8， 过滤器的深度为8
    conv3_3_weights = tf.get_variable("conv3_3_weights", [3, 3, 8, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv3_3_biases = tf.get_variable("conv3_3_biases", [8], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv3_3 = tf.nn.conv2d(relu3_2, conv3_3_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu3_3 = tf.nn.relu(tf.nn.bias_add(conv3_3, conv3_3_biases))

    # max_pool3
    # 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
    pool3 = tf.nn.max_pool(relu3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Convolution4_1
    # 过滤器大小为3*3, 当前层深度为8， 过滤器的深度为8
    conv4_1_weights = tf.get_variable("conv4_1_weights", [3, 3, 8, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv4_1_biases = tf.get_variable("conv4_1_biases", [8], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv4_1 = tf.nn.conv2d(pool3, conv4_1_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu4_1 = tf.nn.relu(tf.nn.bias_add(conv4_1, conv4_1_biases))

    # Convolution4_2
    # 过滤器大小为3*3, 当前层深度为8， 过滤器的深度为8
    conv4_2_weights = tf.get_variable("conv4_2_weights", [3, 3, 8, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv4_2_biases = tf.get_variable("conv4_2_biases", [8], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv4_2 = tf.nn.conv2d(relu4_1, conv4_2_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu4_2 = tf.nn.relu(tf.nn.bias_add(conv4_2, conv4_2_biases))

    # Convolution4_3
    # 过滤器大小为3*3, 当前层深度为8， 过滤器的深度为8
    conv4_3_weights = tf.get_variable("conv4_3_weights", [3, 3, 8, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv4_3_biases = tf.get_variable("conv4_3_biases", [8], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv4_3 = tf.nn.conv2d(relu4_2, conv4_3_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu4_3 = tf.nn.relu(tf.nn.bias_add(conv4_3, conv4_3_biases))

    # max_pool4
    # 池化层过滤器的大小为3*3, 移动步长为1，使用全0填充
    pool4 = tf.nn.max_pool(relu4_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

    # Convolution5_1
    # 过滤器大小为3*3, 当前层深度为8， 过滤器的深度为8
    conv5_1_weights = tf.get_variable("conv5_1_weights", [3, 3, 8, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv5_1_biases = tf.get_variable("conv5_1_biases", [8], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv5_1 = tf.nn.conv2d(pool4, conv5_1_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu5_1 = tf.nn.relu(tf.nn.bias_add(conv5_1, conv5_1_biases))

    # Convolution5_2
    # 过滤器大小为3*3, 当前层深度为8， 过滤器的深度为8
    conv5_2_weights = tf.get_variable("conv5_2_weights", [3, 3, 8, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv5_2_biases = tf.get_variable("conv5_2_biases", [8], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv5_2 = tf.nn.conv2d(relu5_1, conv5_2_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu5_2 = tf.nn.relu(tf.nn.bias_add(conv5_2, conv5_2_biases))

    # Convolution5_3
    # 过滤器大小为3*3, 当前层深度为8， 过滤器的深度为8
    conv5_3_weights = tf.get_variable("conv5_3_weights", [3, 3, 8, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv5_3_biases = tf.get_variable("conv5_3_biases", [8], initializer=tf.constant_initializer(0.0))
    # 移动步长为1, 使用全0填充
    conv5_3 = tf.nn.conv2d(relu5_2, conv5_3_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu5_3 = tf.nn.relu(tf.nn.bias_add(conv5_3, conv5_3_biases))

    # 1*1的卷积,第6层
    conv6_weights = tf.get_variable("conv6_weights", [1, 1, 8, 1],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)
    conv6_biases = tf.get_variable("conv6_biases", [1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    # 移动步长为1, 使用全0填充
    conv6 = tf.nn.conv2d(relu5_3, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases))

    relu6 = tf.cast(relu6, dtype='float64')

    # 进行双线性插值
    bottom = tf.reshape(relu6, [-1, Parameters.end_pic_size, Parameters.end_pic_size, 1])
    weights = tf.reshape(bilinear, [kernel_size, kernel_size, 1, 1, ])
    stride = upscale_factor
    strides = [1, stride, stride, 1]
    deconv = tf.nn.conv2d_transpose(bottom, weights,
                                    [Parameters.batch_size, Parameters.pic_size, Parameters.pic_size, 1], strides,
                                    padding='SAME')


    bilinear_reslut = tf.reshape(deconv, [-1, Parameters.y_size], name=None)
    bilinear_reslut = tf.cast(bilinear_reslut, dtype='float32')


    # 预测
    preNum = tf.reduce_sum(bilinear_reslut)
    realNum = tf.constant(Parameters.realNum[j-1])
    error = tf.abs(tf.subtract(preNum, realNum))
    Probability = tf.subtract(1.0, tf.div(error, realNum))
    accuracy = tf.reduce_mean(tf.cast(Probability, tf.float64))

    MAE = tf.reduce_mean(error)
    E = tf.subtract(preNum, realNum)

    # 开始训练
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()


    # Launch the graph
    with tf.Session() as sess:
        # load上一步训练的参数
        saver.restore(sess,saveName)

        distribution=bilinear_reslut.eval(feed_dict={x: test_pics})

        # 测试
        t_accuracy=accuracy.eval(feed_dict={x: test_pics})
        t_MEA=MAE.eval(feed_dict={x: test_pics})
        t_E=E.eval(feed_dict={x: test_pics})
        t_preNum=preNum.eval(feed_dict={x: test_pics})
        t_realNum=Parameters.realNum[j-1]

        Parameters.a_accuracy=Parameters.a_accuracy+t_accuracy
        Parameters.a_MEA=Parameters.a_MEA+t_MEA
        Parameters.a_E=Parameters.a_E+t_E
        Parameters.a_preNum=Parameters.a_preNum+t_preNum
        Parameters.a_realNum=Parameters.a_realNum+t_realNum


        with open(filename, 'a') as fileobject:
            fileobject.write("t_accuracy:\t" + str(t_accuracy) + "\n")
            fileobject.write("t_MEA:\t" + str(t_MEA) + "\n")
            fileobject.write("t_E:\t" + str(t_E) + "\n")
            fileobject.write("t_preNum:\t" + str(t_preNum) + "\n")
            fileobject.write("t_realNum:\t" + str(t_realNum) + "\n")

        return distribution

def finalR(distribution_result,segmentation_result,Times,sess,j):
    filename='group1'+"_final"
    distribution_resultT=tf.Variable(tf.constant(distribution_result))
    segmentation_resultT=tf.Variable(tf.constant(segmentation_result))

    low = tf.Variable(tf.constant([90.0]))
    high = tf.Variable(tf.constant([265.0]))

    sem1 = tf.greater_equal(segmentation_resultT, low, name=None)
    sem2 = tf.to_float(sem1, name='to_float')
    sem3 = tf.multiply(segmentation_resultT, sem2)

    sem4 = tf.less_equal(sem3, high, name=None)
    sem5 = tf.to_float(sem4, name='to_float2')
    sem6 = tf.multiply(sem3, sem5)

    sem7 = tf.clip_by_value(sem6, 0, 1)

    # 预测
    preNum = tf.reduce_sum(distribution_resultT*sem7)
    realNum = tf.constant(Parameters.realNum[j - 1])
    error = tf.abs(tf.subtract(preNum, realNum))
    Probability = tf.subtract(1.0, tf.div(error, realNum))
    accuracy = tf.reduce_mean(tf.cast(Probability, tf.float64))

    MAE = tf.reduce_mean(error)
    E = tf.subtract(preNum, realNum)

    # 开始训练
    init = tf.global_variables_initializer()
    sess.run(init)

    print(accuracy.eval())

    Parameters.f_accuracy = Parameters.f_accuracy + accuracy.eval()
    Parameters.f_MEA = Parameters.f_MEA + MAE.eval()
    Parameters.f_E = Parameters.f_E + E.eval()
    Parameters.f_preNum = Parameters.f_preNum + preNum.eval()
    Parameters.f_realNum = Parameters.f_realNum + realNum.eval()

    with open(filename, 'a') as fileobject:
        fileobject.write("t_accuracy:\t" + str(accuracy.eval()) + "\n")
        fileobject.write("t_MEA:\t" + str(MAE.eval()) + "\n")
        fileobject.write("t_E:\t" + str(E.eval()) + "\n")
        fileobject.write("t_preNum:\t" + str(preNum.eval()) + "\n")
        fileobject.write("t_realNum:\t" + str(realNum.eval()) + "\n")
        fileobject.write("-----------------------------------------------------------\n")



#连续运行若干次
for i in range(5):
    rootpath="/media/public/Flying/testdata/group1/"
    #rootpath="/media/idiot/Ubuntu 16.0/testdata/"
    modepath="/home/public/PycharmProjects/raceNum/crowdNet/"

    modepath2="/home/public/PycharmProjects/fengeTest/crowdNet/save/model.ckpt"
    filename = 'group1' + '_result'
    filename2 = 'group1' +  "_final"

    for j in range(1,11):
        # tensorflow参数设置
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.90
        sess = tf.InteractiveSession(config=sess_config)

        starttime = datetime.datetime.now()
        distribution_result=test(rootpath, i + 1, sess,j,modepath)
        print(distribution_result)
        endtime = datetime.datetime.now()
        tf.reset_default_graph()

        starttimeFinal = datetime.datetime.now()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.90
        sess = tf.InteractiveSession(config=sess_config)
        segmentation_result=segmentation(rootpath, sess,j,modepath2)
        print(segmentation_result)
        tf.reset_default_graph()

        sess_config1 = tf.ConfigProto()
        sess_config1.gpu_options.per_process_gpu_memory_fraction = 0.90
        sess1 = tf.InteractiveSession(config=sess_config1)
        finalR(distribution_result,segmentation_result,i+1,sess1,j)
        tf.reset_default_graph()
        endtimeFinal = datetime.datetime.now()

    a_accuracy = Parameters.a_accuracy / 10
    a_MEA = Parameters.a_MEA / 10
    a_E = Parameters.a_E / 10
    a_preNum = Parameters.a_preNum / 10
    a_realNum = Parameters.a_realNum / 10

    f_accuracy = Parameters.f_accuracy / 10
    f_MEA = Parameters.f_MEA / 10
    f_E = Parameters.f_E / 10
    f_preNum = Parameters.f_preNum / 10
    f_realNum = Parameters.f_realNum / 10

    with open(filename, 'a') as fileobject:
        fileobject.write("---------------------\t" + str("AVE") + "\t---------------------\n")
        fileobject.write("a_accuracy:\t" + str(a_accuracy) + "\n")
        fileobject.write("a_MEA:\t" + str(a_MEA) + "\n")
        fileobject.write("a_E:\t" + str(a_E) + "\n")
        fileobject.write("a_preNum:\t" + str(a_preNum) + "\n")
        fileobject.write("a_realNum:\t" + str(a_realNum) + "\n")

    with open(filename2, 'a') as fileobject:
        fileobject.write("---------------------\t" + str("AVE") + "\t---------------------\n")
        fileobject.write("f_accuracy:\t" + str(f_accuracy) + "\n")
        fileobject.write("f_MEA:\t" + str(f_MEA) + "\n")
        fileobject.write("f_E:\t" + str(f_E) + "\n")
        fileobject.write("f_preNum:\t" + str(f_preNum) + "\n")
        fileobject.write("f_realNum:\t" + str(f_realNum) + "\n")

    Parameters.a_accuracy = 0.0
    Parameters.a_MEA = 0.0
    Parameters.a_E = 0.0
    Parameters.a_preNum = 0.0
    Parameters.a_realNum = 0.0

    Parameters.f_accuracy = 0.0
    Parameters.f_MEA = 0.0
    Parameters.f_E = 0.0
    Parameters.f_preNum = 0.0
    Parameters.f_realNum = 0.0

    with open(filename, 'a') as fileobject:
        fileobject.write("total time:\t"+str((endtime - starttime).seconds)+"\n")
        fileobject.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")


    with open(filename2, 'a') as fileobject:
        fileobject.write("total time:\t" + str((endtimeFinal - starttimeFinal).seconds) + "\n")
        fileobject.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
print("---------------------------------------------------------------")
