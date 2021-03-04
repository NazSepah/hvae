import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import tensorflow as tf
from model2d_pred_det import HierarchicalProbUNet


class VisualizeSamples:
    def __init__(self, config, name, reuse=False):
        super(VisualizeSamples, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.reuse = reuse
        self.filename = 'recon_sample_' + name + '_'

    def on_epoch_end(self, callback_data, nb_imgs, nb_samples, epoch):
        self.callback_data = callback_data
        print("reconstructing...")

        fig_seg = plt.figure(figsize=(3 * 6, 3 * nb_imgs))
        gs_seg_outer = gridspec.GridSpec(1, 6, hspace=-.01, wspace=-0.01)

        gs_x1 = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[0], wspace=0, hspace=0)
        gs_x2 = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[1], wspace=0, hspace=0)
        gs_diff = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[2], wspace=0, hspace=0)
        gs_true_newt2 = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[3], wspace=0, hspace=0)
        gs_true_t2 = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[4], wspace=0, hspace=0)
        gs_pred_newt2 = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[5], wspace=0, hspace=0)

        # set up the model
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        img1_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])
        img2_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])

        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        y_pred_newt2_tf = tf.nn.sigmoid(hpu_net.reconstruct(img1_tf))

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                batch_np = sess.run(next_batch)

                img2_np = (batch_np[1][..., :4] - batch_np[0][..., :4])
                img1_np = batch_np[0][..., :4]
                y_true_newt2 = batch_np[1][0, ..., -1]
                y_true_t2 = batch_np[0][0, ..., 4]

                pred_newt2 = sess.run(y_pred_newt2_tf, feed_dict={img1_tf: img1_np,
                                                                  img2_tf: img2_np})

                print(np.shape(pred_newt2[0, ...]), np.min(pred_newt2[0, ...]), np.max(pred_newt2[0, ...]))
                y_pred_newt2 = pred_newt2[0, ..., 0]

                x1 = batch_np[0][0, ..., :4]
                x2 = batch_np[1][0, ..., :4]

                # modify arrays for plotting
                x1 = x1[::-1, :, 2]
                x2 = x2[::-1, :, 2]
                x2_scaled = x2 * 0.5

                x3 = np.zeros((192, 192, 3))
                x3[..., 0] = x2_scaled
                x3[..., 1] = x2_scaled
                x3[..., 2] = x2_scaled

                x3[..., 0] += 0.5 * (y_true_newt2[::-1, ...]>0).astype(int)
                x3 = (x3 - np.min(x3)) / (np.max(x3) - np.min(x3))

                x5 = np.zeros((192, 192, 3))
                x5[..., 0] = x2_scaled
                x5[..., 1] = x2_scaled
                x5[..., 2] = x2_scaled

                x5[..., 0] += 0.5 * (y_true_t2[::-1, ...]).astype(int)
                x5 = (x5 - np.min(x5)) / (np.max(x5) - np.min(x5))

                x4 = np.zeros((192, 192, 3))
                x4[..., 0] = x2_scaled
                x4[..., 1] = x2_scaled
                x4[..., 2] = x2_scaled

                x4[..., 0] += 0.8 * (y_pred_newt2[::-1, ...])
                x4 = (x4 - np.min(x4)) / (np.max(x4) - np.min(x4))

                ax = fig_seg.add_subplot(gs_x1[i])
                ax.imshow(x1, cmap='Greys_r')
                ax.axis('off')
                ax.set_title("tp1")

                ax = fig_seg.add_subplot(gs_x2[i])
                ax.imshow(x2, cmap='Greys_r')
                ax.axis('off')
                ax.set_title("tp2")

                ax = fig_seg.add_subplot(gs_diff[i])
                ax.imshow(x2-x1, cmap='Greys_r')
                ax.axis('off')
                ax.set_title("diff")

                ax = fig_seg.add_subplot(gs_true_newt2[i])
                ax.imshow(x3, cmap='Greys_r')
                ax.axis('off')
                ax.set_title("true-newt2")

                ax = fig_seg.add_subplot(gs_true_t2[i])
                ax.imshow(x5, cmap='Greys_r')
                ax.axis('off')
                ax.set_title("true-t2")

                ax = fig_seg.add_subplot(gs_pred_newt2[i])
                ax.imshow(x4, cmap='Greys_r')
                ax.axis('off')
                ax.set_title("pred-newt2")

            fig_seg.savefig(os.path.join(self.outdir, self.expt_name, "recons", self.filename + "{}.png".format(epoch)))
            plt.close()


class PlotRocSeg:
    def __init__(self, config, name='valid', reuse=False):
        super(PlotRocSeg, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.save_np = expt_config['save_np']
        self.mode = expt_config['mode']
        self.reuse = reuse
        self.filename = 'roc_seg_' + name + '_'

    def on_epoch_end(self, callback_data, nb_imgs, epoch):

        self.callback_data = callback_data

        # set up the model
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        img1_tf = next_batch[0][..., :4]
        img2_tf = (next_batch[1][..., :4] - next_batch[0][..., :4])

        img1_tf = tf.reshape(img1_tf, [1, 192, 192, 4])
        img2_tf = tf.reshape(img2_tf, [1, 192, 192, 4])

        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        y_pred_newt2_post_tf = tf.nn.sigmoid(hpu_net.reconstruct(img1_tf))
        y_pred_newt2_prior_tf = tf.nn.sigmoid(hpu_net.reconstruct(img1_tf))

        thresholds = [0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.999, 1]
        nb_thrs = len(thresholds)

        fpr_post = np.empty((nb_imgs, nb_thrs))
        tpr_post = np.empty((nb_imgs, nb_thrs))
        fdr_post = np.empty((nb_imgs, nb_thrs))

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                print(i)

                y_pred_post_newt2, y_pred_prior_newt2, batch = sess.run([y_pred_newt2_post_tf, y_pred_newt2_prior_tf, next_batch])

                y_pred_post = y_pred_post_newt2[0][..., 0]
                y_pred_prior = y_pred_prior_newt2[0][..., 0]
                y_true = (batch[1][0, ..., -1] >0).astype(int)

                for j, thr in enumerate(thresholds):
                    y_pred_t = (y_pred_post >= thr).astype(int)
                    total_p = np.sum(y_true)
                    tp = np.sum(y_true * y_pred_t)
                    fp = np.sum((1 - y_true) * y_pred_t)
                    fn = np.sum(y_true * (1 - y_pred_t))
                    tn = np.sum((1 - y_true) * (1 - y_pred_t))
                    tpr_post[i, j] = tp / (tp + fn + .0001)
                    if total_p==0:
                        tpr_post[i, j] = 1
                    fpr_post[i, j] = fp / (fp + tn + .0001)
                    fdr_post[i, j] = fp / (tp + fp + .0001)

        fdr_post_mean = np.mean(fdr_post, axis=0)
        tpr_post_mean = np.mean(tpr_post, axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(fdr_post_mean, tpr_post_mean)
        plt.legend(loc="lower right")
        major_ticks = np.arange(0, 1, 0.1)
        minor_ticks = np.arange(0, 1, 0.02)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        ax.set_xlabel('FDR')
        ax.set_ylabel('TPR')
        fig.savefig(os.path.join(self.outdir, self.expt_name, "rocs", self.filename + "{}.png".format(epoch)))
        plt.close()
