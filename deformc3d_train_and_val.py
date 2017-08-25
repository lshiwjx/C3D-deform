from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from six.moves import xrange
from deformc3d_multi_gpu_tower import *
from deformc3d_read_tfrecord import *
from deformc3d_model import *

FLAGS = tf.app.flags.FLAGS

# Keep 3 decimal place
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
# Print device log before training
tf.app.flags.DEFINE_boolean('log_device_placement', False, """find out which devices are used.""")
# saver
tf.app.flags.DEFINE_string('checkpoint_dir', './checkout_dir', "")
tf.app.flags.DEFINE_string('summaries_dir', './summary_dir', "")
tf.app.flags.DEFINE_string('tf_record_train', './rgb_8_train_uint8.tfrecords', "")
tf.app.flags.DEFINE_string('tf_record_val', './rgb_8_val_uint8.tfrecords', "")

# model load
tf.app.flags.DEFINE_string('pretrain_model_file', './sports1m_finetuning_ucf101.model', "")
tf.app.flags.DEFINE_boolean('use_pretrain_model', True, """.""")
tf.app.flags.DEFINE_string('last_model', './model.ckpt-41400', "")
tf.app.flags.DEFINE_boolean('use_last_model', False, """""")

tf.app.flags.DEFINE_integer('batch_size', 1, "")
tf.app.flags.DEFINE_integer('video_clip_channels', 3, "")
tf.app.flags.DEFINE_integer('video_clip_length', 16, "the number of frame for a clip")
tf.app.flags.DEFINE_integer('video_clip_height', 120, "")
tf.app.flags.DEFINE_integer('video_clip_width', 160, "")
tf.app.flags.DEFINE_integer('crop_size', 112, "")
tf.app.flags.DEFINE_float('crop_mean0', 101.60, "")
tf.app.flags.DEFINE_float('crop_mean1', 97.62, "")
tf.app.flags.DEFINE_float('crop_mean2', 90.34, "")

tf.app.flags.DEFINE_integer('num_classes', 101, "")
tf.app.flags.DEFINE_integer('num_gpus', 8, """How many GPUs to use.""")

tf.app.flags.DEFINE_integer('num_epochs', None, """Number of epochs to run.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
# learning rate schedule
tf.app.flags.DEFINE_integer('learning_rate_decay_step', 10000, "")  # 860
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1, "")  # 0.1
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001, "")  # 0.001

tf.app.flags.DEFINE_float('moving_average_decay', 0.2, "")  # 0.2

tf.app.flags.DEFINE_float('dropout_ratio', 0.6, "")

tf.app.flags.DEFINE_float('weight_decay_ratio', 0.001, "")
tf.app.flags.DEFINE_float('wd',0.01,"")
# shuffle level
tf.app.flags.DEFINE_integer('min_after_dequeue', 4000, "")


def train():
    with tf.Graph().as_default() as _, tf.device('/cpu:0'):
        # The golbal step will update in function apply_gradient
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0), trainable=False)

        # Learning rate schedule.
        learning_rate_basic = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                         global_step,
                                                         FLAGS.learning_rate_decay_step,
                                                         FLAGS.learning_rate_decay_factor,
                                                         staircase=True)
        tf.summary.scalar('learning rate: ', learning_rate_basic)

        # Create an optimizer that performs gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate_basic)

        # diff for train and val, so use the placeholder
        is_training = tf.placeholder(tf.bool)
        dropout_ratio = tf.placeholder(tf.float32)

        # Start reading data queues
        with tf.name_scope('input_pipeline'):
            with tf.name_scope('train'):
                images_batch_train, labels_batch_train = read_data_batch(FLAGS.tf_record_train)
            with tf.name_scope('val'):
                images_batch_val, labels_batch_val = read_data_batch(FLAGS.tf_record_val)

        images_batch, labels_batch = tf.cond(is_training,
                                             lambda: [images_batch_train, labels_batch_train],
                                             lambda: [images_batch_val, labels_batch_val])
        # Start calculate loss and accuracy
        accuracy, losses, grads = [[], [], []]
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        tower_loss, tower_accuracy = tower_loss_accuracy(
                            scope, images_batch, labels_batch, dropout_ratio)
                        losses.append(tower_loss)
                        accuracy.append(tower_accuracy)

                        # The first execution has no variables.
                        tf.get_variable_scope().reuse_variables()

                        tower_grads = optimizer.compute_gradients(tower_loss)
                        grads.append(tower_grads)

        # summary the accuracy and loss
        loss_mean = tf.reduce_mean(losses)
        accuracy_mean = tf.reduce_mean(accuracy)
        tf.summary.scalar('loss', loss_mean)
        tf.summary.scalar('accuracy', accuracy_mean)

        # Calculate the mean of gradients from all tower.
        grads_average = average_gradients(grads)

        # Apply the gradient to update shared variables, this will update the global_step
        apply_gradient_op = optimizer.apply_gradients(grads_average, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_step = tf.group(apply_gradient_op, variables_averages_op)

        # Create a session.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        # Merge summary and create two writer to plot variable in same plot
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        val_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/val')

        # Init all
        sess.run(tf.global_variables_initializer())

        # Initial steps
        step = 0
        # Create a saver.
        saver = tf.train.Saver()
        # Whether load existed model
        if FLAGS.use_last_model:
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                saver_c3d = tf.train.Saver()
                print('using last model')
                saver_c3d.restore(sess, FLAGS.last_model)
                step = int(FLAGS.last_model.split('/')[-1].split('-')[-1])

        elif FLAGS.use_pretrain_model:
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                # Choose which variables to load
                print('using pretrained model')
                variables = {
                    "var_name/wc1": tf.get_variable('conv1/weight'),
                    "var_name/wc2": tf.get_variable('conv2/weight'),
                    "var_name/wc3a": tf.get_variable('conv3/weight_a'),
                    "var_name/wc3b": tf.get_variable('conv3/weight_b'),
                    "var_name/wc4a": tf.get_variable('conv4/weight_a'),
                    "var_name/wc4b": tf.get_variable('conv4/weight_b'),
                    # "var_name/wc5a": tf.get_variable('conv5/weight_a'),
                    # "var_name/wc5b": tf.get_variable('conv5/weight_b'),
                    "var_name/wd1": tf.get_variable('local6/weights'),
                    "var_name/wd2": tf.get_variable('local7/weights'),
                    "var_name/bc1": tf.get_variable('conv1/biases'),
                    "var_name/bc2": tf.get_variable('conv2/biases'),
                    "var_name/bc3a": tf.get_variable('conv3/biases_a'),
                    "var_name/bc3b": tf.get_variable('conv3/biases_b'),
                    "var_name/bc4a": tf.get_variable('conv4/biases_a'),
                    "var_name/bc4b": tf.get_variable('conv4/biases_b'),
                    # "var_name/bc5a": tf.get_variable('conv5/biases_a'),
                    # "var_name/bc5b": tf.get_variable('conv5/biases_b'),
                    "var_name/bd1": tf.get_variable('local6/biases'),
                    "var_name/bd2": tf.get_variable('local7/biases')
                }
                saver_c3d = tf.train.Saver(variables)
                saver_c3d.restore(sess, FLAGS.pretrain_model_file)

        # Start train and val.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            print('Training start')
            while not coord.should_stop():
                if not step % 10 == 9:
                    step += 1
                    # Training
                    if step == 5:
                        # add the runtime statistics
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _, loss_train, acc_train, summary_merged \
                            = sess.run([train_step, loss_mean, accuracy_mean, summary_op],
                                       options=run_options,
                                       run_metadata=run_metadata,
                                       feed_dict={is_training: True,
                                                  dropout_ratio: FLAGS.dropout_ratio})
                        train_writer.add_run_metadata(run_metadata, 'step%d' % step)
                        train_writer.add_summary(summary_merged, step)
                        print('step %d loss %.2f accu %.2f' % (step, loss_train, acc_train))
                        print('Adding run metadata for', step)
                    else:
                        _, loss_train, acc_train, summary_merged \
                            = sess.run([train_step, loss_mean, accuracy_mean, summary_op],
                                       feed_dict={is_training: True,
                                                  dropout_ratio: FLAGS.dropout_ratio})
                        train_writer.add_summary(summary_merged, step)
                        assert not np.isnan(loss_train), 'Model diverged with tower_loss = NaN'
                        print('step %d loss %.2f accu %.2f' % (step, loss_train, acc_train))
                else:
                    step += 1
                    # Validation
                    loss_val, acc_val, summary_merged_val = sess.run(
                        [loss_mean, accuracy_mean, summary_op],
                        feed_dict={is_training: False,
                                   dropout_ratio: 1})
                    val_writer.add_summary(summary_merged_val, step)
                    print('Validation: step %d loss %.2f accu %.2f' % (step, loss_val, acc_val))

                # Save the model checkpoint periodically.
                if step % 1000 == 999 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)

        except tf.errors.OutOfRangeError:
            print('Done training for %d steps.' % (step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


def main(_):
    if not FLAGS.use_last_model:
        if tf.gfile.Exists(FLAGS.checkpoint_dir):
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
        if tf.gfile.Exists(FLAGS.summaries_dir):
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
        tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
