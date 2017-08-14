import tensorflow as tf
import os.path as osp
import deform_conv3d_op


with tf.Graph().as_default():
    image_size = 5
    image_channel = 3
    video_size = 3
    kernel_length = 3
    kernel_height = 3
    kernel_width = 3
    out_length = 1
    out_height = 3
    out_width = 3
    kernel_channel = 2
    offset_group = 1
    batch_size = 2

    image_shape = (batch_size, image_channel, video_size, image_size, image_size)
    offset_shape = (offset_group, out_length, out_height, out_width, kernel_length, kernel_height, kernel_width, 3)
    kernel_shape = (kernel_channel, kernel_length, kernel_height, kernel_width)
    out_shape = (batch_size, image_channel*kernel_channel, out_length,out_height,out_width)
    images = tf.Variable(tf.random_normal(image_shape,
                                          dtype=tf.float32,
                                          stddev=1e-1))
    offset = tf.Variable(tf.random_normal(offset_shape,
                                          dtype=tf.float32,
                                          stddev=1e-1))
    kernel = tf.Variable(tf.random_normal(kernel_shape,
                                          dtype=tf.float32,
                                          stddev=1e-1))

    last_layer = deform_conv3d_op.deform_conv3d(images, kernel, offset)

    # Build an initialization operation.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()
    sess.run(init)

    err = tf.test.compute_gradient_error([images,kernel,offset],
                                         [image_shape,kernel_shape,offset_shape],
                                         last_layer,
                                         out_shape)

    print(sess.run(err))
