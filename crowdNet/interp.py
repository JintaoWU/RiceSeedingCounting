import numpy as np
import tensorflow as tf

#获取插值矩阵
def get_bilinear_filter(filter_shape, upscale_factor):
        ##filter_shape is [width, height, num_in_channels, num_out_channels]
        kernel_size = 2*upscale_factor - upscale_factor%2
        ### Centre location of the filter for which value is calculated
        if kernel_size % 2 == 1:
            centre_location = upscale_factor - 1
        else:
            centre_location = upscale_factor - 0.5
        bilinear = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(filter_shape[0]):
            for y in range(filter_shape[1]):
                ##Interpolation Calculation
                value = (1 - abs((x - centre_location)/ upscale_factor)) * (1 - abs((y - centre_location)/ upscale_factor))
                bilinear[x, y] = value
        return bilinear


"""
upscale_factor=8
kernel_size = 2*upscale_factor - upscale_factor%2
bilinear = get_bilinear_filter([kernel_size,kernel_size,1,1,],upscale_factor)

#bottom = tf.constant([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]], dtype='float64')
#bottom = tf.reshape(bottom, [1, 3, 6, 1])

bottom=tf.random_normal([3, 16, 16], stddev=1, seed=1, dtype='float64')
bottom = tf.reshape(bottom, [-1, 16, 16, 1])



weights = tf.reshape(bilinear, [kernel_size,kernel_size,1,1,])
stride = upscale_factor
strides = [1, stride, stride, 1]
deconv = tf.nn.conv2d_transpose(bottom, weights, [3, 128, 128, 1],strides, padding='SAME')

with tf.Session() as sess:
   tt=deconv.eval()
   print("deconv: ", tt)
   print("-------------------------------------")
   print(tt.reshape((3,128,128)))
"""
