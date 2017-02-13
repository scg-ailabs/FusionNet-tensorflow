import tensorflow as tf
from FusionNet import FusionNet
import numpy as np
from ops import losses
from glob import glob
import cv2
import time

def train_with_cpu(flag):
    with tf.Graph().as_default():
        data = glob('./dataset/{}/train/*.pgm'.format(flag.dataset_name))

        num_samples_per_epoch = len(data)
        num_batches_per_epoch = num_samples_per_epoch // flag.batch_size

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0), trainable=False)

        decay_steps = int(num_batches_per_epoch * flag.num_epochs_per_decay)

        lr = tf.train.exponential_decay(flag.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        flag.learning_rate_decay_factor,
                                        staircase=True)

        input_placeholder = tf.placeholder(tf.float32,
                                           [flag.batch_size, flag.image_size, flag.image_size*2, flag.channel_dim])

        input_ = input_placeholder[:, :, :flag.image_size, :]
        target_ = input_placeholder[:, :, flag.image_size:flag.image_size*2, :]

        fusionNet = FusionNet()
        output_ = fusionNet.inference(input_)

        loss = losses.pixel_wise_l2_loss(output_, target_)

        var_list = tf.trainable_variables()

        optimizer = tf.train.AdamOptimizer(lr)
        grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

        print("opt complete")
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            print("Learning start!!")
            start = time.time()

            for epoch in range(flag.total_epoch):
                np.random.shuffle(data)
                print("Shuffled")
                for batch_idx in range(num_batches_per_epoch):
                    batch_files = data[batch_idx * flag.batch_size:(batch_idx + 1) * flag.batch_size]
                    batch_image = [cv2.imread(batch_file) for batch_file in batch_files]
                    batch_images = np.array(batch_image).astype(np.float32)
                    feed = {input_placeholder: batch_images}
                    print("Feeded")

                    sess.run(train_op, feed_dict=feed)
                    print("Epoch: %d[%d/%d], time: %d, loss: %f" % (epoch, batch_idx, num_batches_per_epoch, time.time() - start, sess.run(loss, feed)))
                print("Epoch: %d, time: %d, loss: %f" % (epoch, time.time() - start, sess.run(loss)))

def train_with_gpu(flag):
    pass