from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from six.moves import xrange
from c3d_multi_gpu_tower import *
from c3d_read_tfrecord import *
from c3d_model import *

FLAGS = tf.app.flags.FLAGS
# train
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 101,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_gpus', 8, """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('num_epochs', 0, """Number of epochs to run.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
# Keep 3 decimal place
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """find out which devices are used.""")
# saver
tf.app.flags.DEFINE_string('checkpoint_dir', './checkout_dir', "")
tf.app.flags.DEFINE_string('summaries_dir', './summary_dir', "")
# model load
tf.app.flags.DEFINE_string('pretrain_model_file', './sports1m_finetuning_ucf101.model', "")
tf.app.flags.DEFINE_boolean('use_pretrain_model', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_string('last_model', FLAGS.checkpoint_dir + '/model.ckpt-1000', "")
tf.app.flags.DEFINE_boolean('use_last_model', False, """Whether to log device placement.""")
# decay
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 4, "")
tf.app.flags.DEFINE_integer('num_img_per_epoch', 10625, "get from pre_convert_image_to_list.sh")  # 2710
tf.app.flags.DEFINE_float('moving_average_decay', 0.2, "")  # 0.2
# learning rate schedule
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5, "")  # 0.1
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01, "")  # 0.01


def train():
    with tf.Graph().as_default() as _, tf.device('/cpu:0'):
        # will update in apply_gradient
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        num_batches_per_epoch = (FLAGS.num_img_per_epoch / FLAGS.batch_size / FLAGS.video_clip_length)
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
        learning_rate_basic = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                         global_step,
                                                         decay_steps,
                                                         FLAGS.learning_rate_decay_factor,
                                                         staircase=True)
        tf.summary.scalar('learning rate: ', learning_rate_basic)

        # Create an optimizer that performs gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate_basic)

        # data for train
        images_batch, labels_batch = read_data_batch(is_training=True, batch_size=FLAGS.batch_size,
                                                     num_epochs=FLAGS.num_epochs)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images_batch, labels_batch], capacity=3 * FLAGS.num_gpus)
        # with tf.name_scope('train'):
        print('----------------------------Training----------------------------------')
        accuracy = []
        losses = []
        grads = []
        summary_train = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (c3d_model.TOWER_NAME, i)) as scope:
                        image_batch, label_batch = batch_queue.dequeue()

                        tower_loss, tower_accuracy = \
                            tower_loss_accuracy(scope, image_batch, label_batch, is_training=True)
                        losses.append(tower_loss)
                        accuracy.append(tower_accuracy)

                        # TODO: Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        tower_grads = optimizer.compute_gradients(tower_loss)
                        grads.append(tower_grads)
        # summary the accuracy and loss
        loss_mean = tf.reduce_mean(losses)
        accuracy_mean = tf.reduce_mean(accuracy)
        summary_train.append(tf.summary.scalar('accuracy', accuracy_mean))
        summary_train.append(tf.summary.scalar('loss', loss_mean))

        # We must calculate the mean of each gradient. grads is a list of grad from all towers
        grads_average = average_gradients(grads)
        # for grad, var in grads_average:
        #     if grad is not None:
        #         tf.summary.histogram(var.op.name + '/gradients', grad)

        # apply the gradient to update shared variables, this will update the global_step
        apply_gradient_op = optimizer.apply_gradients(grads_average, global_step=global_step)
        # grads_b = average_gradients(tower_grads_b)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_step = tf.group(apply_gradient_op, variables_averages_op)

        # with tf.name_scope('val'):
        print('----------------------------validation-------------------------------------')
        summary_val = []
        images_batch_validation, labels_batch_validation \
            = read_data_batch(is_training=False, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
        batch_queue_validation = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images_batch_validation, labels_batch_validation], capacity=3 * FLAGS.num_gpus)
        accuracy_validation = []
        loss_validation = []
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (c3d_model.TOWER_NAME + '_validation', i)) as scope:
                        image_batch_validation, label_batch_validation \
                            = batch_queue_validation.dequeue()
                        tower_loss_validation, tower_accuracy_validation \
                            = tower_loss_accuracy(scope, image_batch_validation, label_batch_validation,
                                                  is_training=False)

                        accuracy_validation.append(tower_accuracy_validation)
                        loss_validation.append(tower_loss_validation)
                        summary_val.append(tf.get_collection(tf.GraphKeys.SUMMARIES,scope))

        accuracy_mean_validation = tf.reduce_mean(accuracy_validation)
        loss_mean_validation = tf.reduce_mean(loss_validation)
        summary_val.append(tf.summary.scalar('accuracy', accuracy_mean_validation))
        summary_val.append(tf.summary.scalar('loss', loss_mean_validation))

        # Create a saver.
        saver = tf.train.Saver()

        # Create a session.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        # whether load existed model
        if FLAGS.use_pretrain_model:
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                saver_c3d = tf.train.Saver()
                saver_c3d.restore(sess, FLAGS.pretrain_model_file)
        elif FLAGS.use_last_model:
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                saver_c3d = tf.train.Saver()
                saver_c3d.restore(sess, FLAGS.last_model)
                # TODO

        # merge summary
        summary_val_op = tf.summary.merge(summary_val)
        # summary_train_op = tf.summary.merge(summary_train)
        summary_train_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        val_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/val')

        # init
        sess.run(tf.global_variables_initializer())
        # 0 or from saver
        step = sess.run(global_step)

        # Start train and val.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            print('Training start')
            while not coord.should_stop():
                step += 1
                epoch = step / num_batches_per_epoch / FLAGS.num_gpus
                if step % 10 == 0:
                    loss_val, acc_val, summary_merged_val = sess.run(
                        [loss_mean_validation, accuracy_mean_validation, summary_val_op])
                    val_writer.add_summary(summary_merged_val, step)
                    print('Validation: epoch %d step %d loss %.2f accu %.2f' % (epoch, step, loss_val, acc_val))
                else:
                    if step % 100 == 99:
                        # add the runtime statistics
                        # choose session runs in tensorboard
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _, loss_train, acc_train, summary_merged \
                            = sess.run([train_step, loss_mean, accuracy_mean, summary_train_op],
                                       options=run_options,
                                       run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'step%d' % step)
                        train_writer.add_summary(summary_merged, step)
                        print('epoch %d step %d loss %.2f accu %.2f' % (epoch, step, loss_train, acc_train))
                        print('Adding run metadata for', step)
                    else:
                        _, loss_train, acc_train, summary_merged \
                            = sess.run([train_step, loss_mean, accuracy_mean, summary_train_op])
                        train_writer.add_summary(summary_merged, step)
                        assert not np.isnan(loss_train), 'Model diverged with tower_loss = NaN'
                        print('epoch %d step %d loss %.2f accu %.2f' % (epoch, step, loss_train, acc_train))

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            epoch = step / num_batches_per_epoch / FLAGS.num_gpus
            print('Done training for %d epochs, %d steps.' % (epoch, step))
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
