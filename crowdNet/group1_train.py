#coding=utf-8
import tensorflow as tf
from crowdNet import read_pic
from crowdNet import interp
from crowdNet import Parameters
import datetime


readgap=2000
totalNum=7208


def train(Times,sess):
    saveName="group1_"+str(Times)

    starttime = datetime.datetime.now()
    print("load data.......\t 1,%d" % (readgap))
    # import train data
    path = "/media/public/Flying/traindata/group1/"
    pic_path = path + str(Parameters.pic_size) + '_pic_set/'
    density_map_path = path + str(Parameters.pic_size) + '_density_set/'
    pics,density=read_pic.Read_Data3(pic_path, density_map_path, 1, readgap)


    readTime=1

    # import test data
    pathTest = "/media/public/Flying/testdata/group1/test/"
    test_pic_path = pathTest + str(Parameters.pic_size) + '_pic_set/'
    test_density_map_path = pathTest + str(Parameters.pic_size) + '_density_set/'
    test_pics, test_density = read_pic.Read_Data2(test_pic_path, test_density_map_path, Parameters.batch_size)
    endtime = datetime.datetime.now()

    filename='group1'
    with open(filename,'a') as fileobject:
        fileobject.write("---------------------------\t"+str(Times)+"\t--------------------------------\n")
        fileobject.write(str((endtime - starttime).seconds)+"\n")

    print("Data loading completed!!!")

    #上采样准备
    upscale_factor = int(Parameters.pic_size / Parameters.end_pic_size)
    kernel_size = 2 * upscale_factor - upscale_factor % 2
    bilinear = interp.get_bilinear_filter([kernel_size, kernel_size, 1, 1, ], upscale_factor)



    # 训练数据
    x = tf.placeholder(dtype=tf.float32, shape=[None, Parameters.x_size])
    # 训练标签数据
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, Parameters.y_size])
    # 把x更改为4维张量，第1维代表样本数量，第2维和第3维代表图像长宽， 第4维代表图像通道数, 1表示黑白
    x_image = tf.reshape(x, [-1, Parameters.pic_size, Parameters.pic_size, 3])


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
    # 过滤器大小为3*3, 当前层深度为64， 过滤器的深度为8
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
    # 过滤器大小为3*3, 当前层深度为64， 过滤器的深度为16
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



    #损失函数
    loss = tf.reduce_sum(tf.square(y_ - bilinear_reslut))

    # 选择优化器，并让优化器最小化损失函数/收敛, 反向传播
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

    # 预测
    # e=tf.constant(0.00001)
    preNum = tf.reduce_sum(bilinear_reslut, 1)
    realNum = tf.reduce_sum(y_, 1)
    error = tf.abs(tf.subtract(preNum, realNum))
    Probability = tf.subtract(1.0, tf.div(error, realNum))
    accuracy = tf.reduce_mean(tf.cast(Probability, tf.float64))

    MAE = tf.reduce_mean(error)
    E = tf.subtract(preNum, realNum)

    # 开始训练
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    #迭代训练
    for i in range(1, 400):
        image_batch, density_batch = read_pic.get_batch_buffer(pics, density, i, Parameters.batch_size)
        Test_image_batch, Test_density_batch = read_pic.get_batch_buffer(test_pics, test_density, i,
                                                                         Parameters.batch_size)
        # batch = mnist.train.next_batch(100)
        if i % 1 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: image_batch, y_: density_batch})  # 评估阶段不使用Dropout
            print("step %d, training accuracy: %f, \t MEA: %f" % (i, train_accuracy, MAE.eval(feed_dict={x: image_batch, y_: density_batch})))
            test_train_accuracy = accuracy.eval(feed_dict={x: Test_image_batch, y_: Test_density_batch})  # 评估阶段不使用Dropout
            print("step %d, testing accuracy: %f, \t MEA: %f" % (i, test_train_accuracy, MAE.eval(feed_dict={x: Test_image_batch, y_: Test_density_batch})))
            print("step %d, E: %s" % (i, E.eval(feed_dict={x: image_batch, y_: density_batch})))
            print("---------------------------------------------------------------------------------")
            # print("step %d, E: %s" % (i, E.eval(feed_dict={x: image_batch, y_: density_batch})))
            # print(preNum.eval(feed_dict={x: image_batch}))
            # print(realNum.eval(feed_dict={y_: density_batch}))
            # print(error.eval(feed_dict={x: image_batch, y_: density_batch}))
        train_step.run(feed_dict={x: image_batch, y_: density_batch})  # 训练阶段使用50%的Dropout

        if i % 100==0:
            readlow=readTime*readgap+1
            readhigh=readlow+readgap-1
            readTime = readTime + 1
            if readhigh>totalNum:
                readhigh=totalNum
                readTime=0

            print("load data.....\t %d,%d" % (readlow,readhigh))
            starttime = datetime.datetime.now()
            # import new train data
            read_pic.Read_Data4(pic_path, density_map_path, readlow,readhigh,pics, density)
            endtime = datetime.datetime.now()

            with open(filename, 'a') as fileobject:
                fileobject.write(str((endtime - starttime).seconds)+"\n")

    saver_path = saver.save(sess, saveName+"/model.ckpt")  # 将模型保存到save/model.ckpt文件


#连续运行若干次
for i in range(1,6):
    # tensorflow参数设置
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.90
    sess = tf.InteractiveSession(config=sess_config)

    filename="group1"
    starttime = datetime.datetime.now()
    train(i+1,sess)
    endtime = datetime.datetime.now()
    with open(filename, 'a') as fileobject:
        fileobject.write("total:\t"+str((endtime - starttime).seconds)+"\n")
    tf.reset_default_graph()
    print("---------------------------------------------------------------")

