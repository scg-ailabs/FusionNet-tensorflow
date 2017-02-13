import tensorflow as tf
import numpy as np
from ops import layers
from ops import acts

class Unet(object):
    def __init__(self):
        self.batch_size = 128
        self.image_size = 28
        self.color_channel = 1
        self.num_kernel = 64
        self.num_linear = 512
        self.leak = 0
        self.trans_kernel = 2
        self.num_class = 1


    def inference(self, images):
        e1_1 = batch_norm(leaky_relu(conv2d(images, self.num_kernel, name='e_conv1_1'), self.leak, name='e_act1_1'), name='e_bn1_1')
        e1_2 = batch_norm(leaky_relu(conv2d(e1_1, self.num_kernel, name='e_conv1_2'), self.leak, name='e_act1_2'), name='e_bn1_2')

        e1_3 = max_pool(e1_2, name="e_pool1")

        e2_1 = batch_norm(leaky_relu(conv2d(e1_3, self.num_kernel * 2, name='e_conv2_1'), self.leak, name='e_act2_1'), name='e_bn2_1')
        e2_2 = batch_norm(leaky_relu(conv2d(e2_1, self.num_kernel * 2, name='e_conv2_2'), self.leak, name='e_act2_2'), name='e_bn2_2')

        e2_3 = max_pool(e2_2, name="e_pool2")

        e3_1 = batch_norm(leaky_relu(conv2d(e2_3, self.num_kernel * 4, name='e_conv3_1'), self.leak, name='e_act3_1'), name='e_bn3_1')
        e3_2 = batch_norm(leaky_relu(conv2d(e3_1, self.num_kernel * 4, name='e_conv3_2'), self.leak, name='e_act3_2'), name='e_bn3_2')

        e3_3 = max_pool(e3_2, name="e_pool3")

        e4_1 = batch_norm(leaky_relu(conv2d(e3_3, self.num_kernel * 8, name='e_conv4_1'), self.leak, name='e_act4_1'), name='e_bn4_1')
        e4_2 = batch_norm(leaky_relu(conv2d(e4_1, self.num_kernel * 8, name='e_conv4_2'), self.leak, name='e_act4_2'), name='e_bn4_2')

        e4_3 = max_pool(e4_2, name="e_pool4")

        e5_1 = batch_norm(leaky_relu(conv2d(e4_3, self.num_kernel * 16, name='e_conv5_1'), self.leak, name='e_act5_1'), name='e_bn5_1')

        d5_1 = batch_norm(leaky_relu(conv2d(e5_1, self.num_kernel * 16, name='d_conv5_1'), self.leak, name='d_act5_1'), name='d_bn5_1')


        d4_1 = conv2d_T(d5_1, e4_2.get_shape(), k_h=self.trans_kernel, k_w=self.trans_kernel, name="d_conv4_1")
        d4_1_concat = tf.concat(3, [d4_1, e4_2])
        d4_2 = batch_norm(leaky_relu(conv2d(d4_1_concat, self.num_kernel * 8, name='d_conv4_2'), self.leak, name='d_act4_2'), name='d_bn4_2')
        d4_3 = batch_norm(leaky_relu(conv2d(d4_2, self.num_kernel * 8, name='d_conv4_3'), self.leak, name='d_act4_3'), name='d_bn4_3')

        d3_1 = conv2d_T(d4_3, e3_2.get_shape(), k_h=self.trans_kernel, k_w=self.trans_kernel, name="d_conv3_1")
        d3_1_concat = tf.concat(3, [d3_1, e3_2])
        d3_2 = batch_norm(leaky_relu(conv2d(d3_1_concat, self.num_kernel * 4, name='d_conv3_2'), self.leak, name='d_act3_2'), name='d_bn3_2')
        d3_3 = batch_norm(leaky_relu(conv2d(d3_2, self.num_kernel * 4, name='d_conv3_3'), self.leak, name='d_act3_3'), name='d_bn3_3')

        d2_1 = conv2d_T(d3_3, e2_2.get_shape(), k_h=self.trans_kernel, k_w=self.trans_kernel, name="d_conv2_1")
        d2_1_concat = tf.concat(3, [d2_1, e2_2])
        d2_2 = batch_norm(leaky_relu(conv2d(d2_1_concat, self.num_kernel * 2, name='d_conv2_2'), self.leak, name='d_act2_2'), name='d_bn2_2')
        d2_3 = batch_norm(leaky_relu(conv2d(d2_2, self.num_kernel * 2, name='d_conv2_3'), self.leak, name='d_act2_3'), name='d_bn2_3')

        d1_1 = conv2d_T(d2_3, e1_2.get_shape(), k_h=self.trans_kernel, k_w=self.trans_kernel, name="d_conv1_1")
        d1_1_concat = tf.concat(3, [d1_1, e1_2])
        d1_2 = batch_norm(leaky_relu(conv2d(d1_1_concat, self.num_kernel, name='d_conv1_2'), self.leak, name='d_act1_2'), name='d_bn1_2')
        d1_3 = batch_norm(leaky_relu(conv2d(d1_2, self.num_kernel, name='d_conv1_3'), self.leak, name='d_act1_3'), name='d_bn1_3')

        output = conv2d(d1_3, self.num_class, k_h=1, k_w=1, name="output")