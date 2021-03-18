import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import tensorflow as tf
from model2d_label_pred import HierarchicalProbUNet
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns


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

    def on_epoch_end(self, callback_data, nb_imgs, mode_value, nb_samples, epoch):
        self.callback_data = callback_data
        print("reconstructing...")

        fig_seg = plt.figure(figsize=(3 * (5 + nb_samples + 2), 3 * nb_imgs))
        gs_seg_outer = gridspec.GridSpec(1, 5 + nb_samples + 2, hspace=-.01, wspace=-0.01)

        gs_x1 = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[0], wspace=0, hspace=0)
        gs_x2 = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[1], wspace=0, hspace=0)
        gs_diff = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[2], wspace=0, hspace=0)
        gs_true_newt2 = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[3], wspace=0, hspace=0)
        gs_true_t2 = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[4], wspace=0, hspace=0)

        gs = []
        for itr in range(nb_samples+2):
            gs.append(gridspec.GridSpecFromSubplotSpec(nb_imgs, 1,
                                                       subplot_spec=gs_seg_outer[5 + itr],
                                                       wspace=0,
                                                       hspace=0))

        # set up the model
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        img1_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])
        img2_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])

        name = 'post'
        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        y_pred_newt2_tf = tf.nn.sigmoid(hpu_net.sample(img1_tf, img2_tf, name))

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

                y_pred_newt2_samples = np.zeros((nb_samples, 192, 192))

                for itr in range(nb_samples):

                    print(i, itr)

                    pred_newt2 = sess.run(y_pred_newt2_tf, feed_dict={img1_tf: img1_np,
                                                                      img2_tf: img2_np})

                    print(np.shape(pred_newt2[0, ...]), np.min(pred_newt2[0, ...]), np.max(pred_newt2[0, ...]))
                    y_pred_newt2 = pred_newt2[0, ..., 0]
                    y_pred_newt2_samples[itr, ...] = y_pred_newt2

                    x4 = np.zeros((192, 192, 3))
                    x4[..., 0] = x2_scaled
                    x4[..., 1] = x2_scaled
                    x4[..., 2] = x2_scaled

                    x4[..., 0] += 0.5 * (y_pred_newt2[::-1, ...])
                    x4 = (x4 - np.min(x4)) / (np.max(x4) - np.min(x4))

                    ax = fig_seg.add_subplot(gs[itr][i])
                    ax.imshow(x4, cmap='Greys_r')
                    ax.set_title("sample-{}".format(itr + 1))
                    ax.axis('off')

                y_pred_newt2_avg = np.sum(y_pred_newt2_samples, axis=0) / nb_samples
                x4 = y_pred_newt2_avg[::-1, ...]
                ax = fig_seg.add_subplot(gs[nb_samples][i])
                ax.imshow(x4, cmap='jet')
                ax.axis('off')
                ax.set_title("pred-avg")

                y_pred_newt2_std = np.std(y_pred_newt2_samples, axis=0)
                x4 = y_pred_newt2_std[::-1, ...]
                ax = fig_seg.add_subplot(gs[nb_samples + 1][i])
                ax.imshow(x4, cmap='jet')
                ax.axis('off')
                ax.set_title("pred-std")

                fig_seg.savefig(os.path.join(self.outdir, self.expt_name, "recons", self.filename + name + "_{}.png".format(epoch)))
                plt.close()


class ReplaceZ:
    def __init__(self, config, reuse=False):
        super(ReplaceZ, self).__init__()
        self.callback_data = None
        self.config = config
        self.reuse = reuse
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config["callback"]['mode']
        self.save_np = expt_config['save_np']
        self.filename = 'roc_replaced_'

    def on_epoch_end(self, callback_data, nb_imgs, mode_value, epoch):

        self.callback_data = callback_data

        # set up the model
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        img1_tf = next_batch[0][..., :4]
        img2_tf = (next_batch[1][..., :4] - next_batch[0][..., :4])
        img1_tf = tf.reshape(img1_tf, [1, 192, 192, 4])
        img2_tf = tf.reshape(img2_tf, [1, 192, 192, 4])

        name = 'post'
        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        y_pred_newt2_tf = tf.nn.sigmoid(hpu_net.sample(img1_tf, img2_tf, name))
        y_pred_newt2_rep_tf_1, y_pred_newt2_rep_tf_2, y_pred_newt2_rep_tf_3, y_pred_newt2_rep_tf_4 = hpu_net.replace_latent(img1_tf, img2_tf, name)

        y_pred_newt2_rep_tf_1 = tf.nn.sigmoid(y_pred_newt2_rep_tf_1)
        y_pred_newt2_rep_tf_2 = tf.nn.sigmoid(y_pred_newt2_rep_tf_2)
        y_pred_newt2_rep_tf_3 = tf.nn.sigmoid(y_pred_newt2_rep_tf_3)
        y_pred_newt2_rep_tf_4 = tf.nn.sigmoid(y_pred_newt2_rep_tf_4)

        thresholds = [0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99,
                      0.999, 1]
        nb_thrs = len(thresholds)

        fpr = np.empty((nb_imgs, nb_thrs))
        tpr = np.empty((nb_imgs, nb_thrs))
        fdr = np.empty((nb_imgs, nb_thrs))

        fpr_rep_1 = np.empty((nb_imgs, nb_thrs))
        tpr_rep_1 = np.empty((nb_imgs, nb_thrs))
        fdr_rep_1 = np.empty((nb_imgs, nb_thrs))

        fpr_rep_2 = np.empty((nb_imgs, nb_thrs))
        tpr_rep_2 = np.empty((nb_imgs, nb_thrs))
        fdr_rep_2 = np.empty((nb_imgs, nb_thrs))

        fpr_rep_3 = np.empty((nb_imgs, nb_thrs))
        tpr_rep_3 = np.empty((nb_imgs, nb_thrs))
        fdr_rep_3 = np.empty((nb_imgs, nb_thrs))

        fpr_rep_4 = np.empty((nb_imgs, nb_thrs))
        tpr_rep_4 = np.empty((nb_imgs, nb_thrs))
        fdr_rep_4 = np.empty((nb_imgs, nb_thrs))

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                print(i)

                y_pred_newt2, y_pred_newt2_rep_1, y_pred_newt2_rep_2, y_pred_newt2_rep_3, y_pred_newt2_rep_4, batch = \
                    sess.run([y_pred_newt2_tf, y_pred_newt2_rep_tf_1, y_pred_newt2_rep_tf_2, y_pred_newt2_rep_tf_3,
                              y_pred_newt2_rep_tf_4, next_batch])

                y_pred = y_pred_newt2[0][..., 0]
                y_pred_rep_1 = y_pred_newt2_rep_1[0][..., 0]
                y_pred_rep_2 = y_pred_newt2_rep_2[0][..., 0]
                y_pred_rep_3 = y_pred_newt2_rep_3[0][..., 0]
                y_pred_rep_4 = y_pred_newt2_rep_4[0][..., 0]
                y_true = (batch[1][0, ..., -1] > 0).astype(int)

                for j, thr in enumerate(thresholds):
                    y_pred_t = (y_pred >= thr).astype(int)
                    total_p = np.sum(y_true)
                    tp = np.sum(y_true * y_pred_t)
                    fp = np.sum((1 - y_true) * y_pred_t)
                    fn = np.sum(y_true * (1 - y_pred_t))
                    tn = np.sum((1 - y_true) * (1 - y_pred_t))
                    tpr[i, j] = tp / (tp + fn + .0001)
                    if total_p == 0:
                        tpr[i, j] = 1
                    fpr[i, j] = fp / (fp + tn + .0001)
                    fdr[i, j] = fp / (tp + fp + .0001)

                    ### rep_1
                    y_pred_t = (y_pred_rep_1 >= thr).astype(int)
                    total_p = np.sum(y_true)
                    tp = np.sum(y_true * y_pred_t)
                    fp = np.sum((1 - y_true) * y_pred_t)
                    fn = np.sum(y_true * (1 - y_pred_t))
                    tn = np.sum((1 - y_true) * (1 - y_pred_t))
                    tpr_rep_1[i, j] = tp / (tp + fn + .0001)
                    if total_p == 0:
                        tpr_rep_1[i, j] = 1
                    fpr_rep_1[i, j] = fp / (fp + tn + .0001)
                    fdr_rep_1[i, j] = fp / (tp + fp + .0001)

                    ### rep_2
                    y_pred_t = (y_pred_rep_2 >= thr).astype(int)
                    total_p = np.sum(y_true)
                    tp = np.sum(y_true * y_pred_t)
                    fp = np.sum((1 - y_true) * y_pred_t)
                    fn = np.sum(y_true * (1 - y_pred_t))
                    tn = np.sum((1 - y_true) * (1 - y_pred_t))
                    tpr_rep_2[i, j] = tp / (tp + fn + .0001)
                    if total_p == 0:
                        tpr_rep_2[i, j] = 1
                    fpr_rep_2[i, j] = fp / (fp + tn + .0001)
                    fdr_rep_2[i, j] = fp / (tp + fp + .0001)

                    ### rep_3
                    y_pred_t = (y_pred_rep_3 >= thr).astype(int)
                    total_p = np.sum(y_true)
                    tp = np.sum(y_true * y_pred_t)
                    fp = np.sum((1 - y_true) * y_pred_t)
                    fn = np.sum(y_true * (1 - y_pred_t))
                    tn = np.sum((1 - y_true) * (1 - y_pred_t))
                    tpr_rep_3[i, j] = tp / (tp + fn + .0001)
                    if total_p == 0:
                        tpr_rep_3[i, j] = 1
                    fpr_rep_3[i, j] = fp / (fp + tn + .0001)
                    fdr_rep_3[i, j] = fp / (tp + fp + .0001)

                    ### rep_4
                    y_pred_t = (y_pred_rep_4 >= thr).astype(int)
                    total_p = np.sum(y_true)
                    tp = np.sum(y_true * y_pred_t)
                    fp = np.sum((1 - y_true) * y_pred_t)
                    fn = np.sum(y_true * (1 - y_pred_t))
                    tn = np.sum((1 - y_true) * (1 - y_pred_t))
                    tpr_rep_4[i, j] = tp / (tp + fn + .0001)
                    if total_p == 0:
                        tpr_rep_4[i, j] = 1
                    fpr_rep_4[i, j] = fp / (fp + tn + .0001)
                    fdr_rep_4[i, j] = fp / (tp + fp + .0001)

        fdr_mean = np.mean(fdr, axis=0)
        tpr_mean = np.mean(tpr, axis=0)

        fdr_rep_mean_1 = np.mean(fdr_rep_1, axis=0)
        tpr_rep_mean_1 = np.mean(tpr_rep_1, axis=0)

        fdr_rep_mean_2 = np.mean(fdr_rep_2, axis=0)
        tpr_rep_mean_2 = np.mean(tpr_rep_2, axis=0)

        fdr_rep_mean_3 = np.mean(fdr_rep_3, axis=0)
        tpr_rep_mean_3 = np.mean(tpr_rep_3, axis=0)

        fdr_rep_mean_4 = np.mean(fdr_rep_4, axis=0)
        tpr_rep_mean_4 = np.mean(tpr_rep_4, axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(fdr_mean, tpr_mean, label='fd-tp')
        plt.plot(fdr_rep_mean_1, tpr_rep_mean_1, label='fd-tp(zero-ed)/scale-1')
        plt.plot(fdr_rep_mean_2, tpr_rep_mean_2, label='fd-tp(zero-ed)/scale-2')
        plt.plot(fdr_rep_mean_3, tpr_rep_mean_3, label='fd-tp(zero-ed)/scale-3')
        plt.plot(fdr_rep_mean_4, tpr_rep_mean_4, label='fd-tp(zero-ed)/scale-4')
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
        # ax.set_title("auc:{}".format(auc(fdr_mean, tpr_mean)))
        fig.savefig(os.path.join(self.outdir, self.expt_name, "rocs", self.filename + name + "_{}.png".format(epoch)))
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

    def on_epoch_end(self, callback_data, nb_imgs, mode_value, epoch):

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
        y_pred_newt2_post_tf = tf.nn.sigmoid(hpu_net.sample(img1_tf, img2_tf, 'post'))
        y_pred_newt2_prior_tf = tf.nn.sigmoid(hpu_net.sample(img1_tf, img2_tf))

        thresholds = [0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.999, 1]
        nb_thrs = len(thresholds)

        fpr_post = np.empty((nb_imgs, nb_thrs))
        tpr_post = np.empty((nb_imgs, nb_thrs))
        fdr_post = np.empty((nb_imgs, nb_thrs))

        fpr_prior = np.empty((nb_imgs, nb_thrs))
        tpr_prior = np.empty((nb_imgs, nb_thrs))
        fdr_prior = np.empty((nb_imgs, nb_thrs))

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

                    y_pred_t = (y_pred_prior >= thr).astype(int)
                    total_p = np.sum(y_true)
                    tp = np.sum(y_true * y_pred_t)
                    fp = np.sum((1 - y_true) * y_pred_t)
                    fn = np.sum(y_true * (1 - y_pred_t))
                    tn = np.sum((1 - y_true) * (1 - y_pred_t))
                    tpr_prior[i, j] = tp / (tp + fn + .0001)
                    if total_p == 0:
                        tpr_prior[i, j] = 1
                    fpr_prior[i, j] = fp / (fp + tn + .0001)
                    fdr_prior[i, j] = fp / (tp + fp + .0001)

        fdr_post_mean = np.mean(fdr_post, axis=0)
        tpr_post_mean = np.mean(tpr_post, axis=0)

        if self.save_np:
            np.save(os.path.join(self.outdir, self.expt_name, "rocs", self.filename + "_tpr_{:03d}_1.npy".format(epoch)),
                tpr_post_mean)
            np.save(os.path.join(self.outdir, self.expt_name, "rocs", self.filename + "_fdr_{:03d}_1.npy".format(epoch)),
                fdr_post_mean)

        indx = np.argmin(abs(fdr_post_mean - 0.2))
        print("HERE IS THE THRESHOLD:", thresholds[indx])

        fdr_prior_mean = np.mean(fdr_prior, axis=0)
        tpr_prior_mean = np.mean(tpr_prior, axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # plt.plot(fpr_mean, tpr_mean, label='fp-tp')
        plt.plot(fdr_post_mean, tpr_post_mean, label='post')
        plt.plot(fdr_prior_mean, tpr_prior_mean, label='prior')
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
        #ax.set_title("auc:{}".format(auc(fdr_mean, tpr_mean)))
        fig.savefig(os.path.join(self.outdir, self.expt_name, "rocs", self.filename + "{}.png".format(epoch)))
        plt.close()


class ActvieUnits:
    def __init__(self, config, name='valid', reuse=False):
        super(ActvieUnits, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.save_np = expt_config['save_np']
        self.mode = expt_config['mode']
        self.reuse = reuse
        self.filename = 'unit_activity_' + name + '_'#+ self.mode

    def on_epoch_end(self, callback_data, nb_imgs, mode_value, epoch):

        self.callback_data = callback_data

        # set up the model
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        img2_tf = (next_batch[1][..., :4] - next_batch[0][..., :4])
        img1_tf = next_batch[0][..., :4]

        img1_tf = tf.reshape(img1_tf, [1, 192, 192, 4])
        img2_tf = tf.reshape(img2_tf, [1, 192, 192, 4])

        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        post, prior = hpu_net.dist(img1_tf, img2_tf)

        nb_prob_scale = len(post)
        post_mean_tf = [tf.reshape(post[i].loc, [-1]) for i in range(nb_prob_scale)]
        prior_mean_tf = [tf.reshape(prior[i].loc, [-1]) for i in range(nb_prob_scale)]

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        post_mean_np = {}
        prior_mean_np = {}
        tot_nb_units = 0
        for level in range(nb_prob_scale):
            nb_units = post_mean_tf[level].get_shape().as_list()[0]
            post_mean_np['scale-{}'.format(level)] = np.zeros((nb_imgs, nb_units))
            prior_mean_np['scale-{}'.format(level)] = np.zeros((nb_imgs, nb_units))
            tot_nb_units += nb_units

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):
                print(i)
                post_mean, prior_mean = sess.run([post_mean_tf, prior_mean_tf])

                for level in range(nb_prob_scale):
                    post_mean_np['scale-{}'.format(level)][i, ...] = post_mean[level]
                    prior_mean_np['scale-{}'.format(level)][i, ...] = prior_mean[level]

        std_post_all_units = np.zeros(tot_nb_units)
        std_prior_all_units = np.zeros(tot_nb_units)
        x_labels = []
        i = 0
        for level in range(nb_prob_scale):
            nb_units = np.shape(post_mean_np['scale-{}'.format(level)])[1]
            std_post_all_units[i: i + nb_units] = np.std(post_mean_np['scale-{}'.format(level)], axis=0)
            std_prior_all_units[i: i + nb_units] = np.std(prior_mean_np['scale-{}'.format(level)], axis=0)
            x_labels += [str(level)] + [''] * (nb_units - 1)
            i += nb_units

        fig = plt.figure(figsize=(16, 4))
        plt.scatter(np.arange(len(x_labels)), std_post_all_units, label='post', alpha=.5, color='red')
        plt.scatter(np.arange(len(x_labels)), std_prior_all_units, label='prior', alpha=.5, color='blue')
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
        plt.ylim([0, 2.5])
        plt.legend()
        fig.savefig(os.path.join(self.outdir, self.expt_name, "covs", self.filename + "{}_one.png".format(epoch)))
        plt.close()


class ZEntropy:
    def __init__(self, config, name='valid', reuse=False):
        super(ZEntropy, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.save_np = expt_config['save_np']
        self.mode = expt_config['mode']
        self.reuse = reuse
        self.filename = 'code_entropy_' + name + '_'

    def on_epoch_end(self, callback_data, nb_imgs, mode_value, epoch):

        self.callback_data = callback_data

        # set up the model
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        img2_tf = (next_batch[1][..., :4] - next_batch[0][..., :4])
        img1_tf = next_batch[0][..., :4]

        img1_tf = tf.reshape(img1_tf, [1, 192, 192, 4])
        img2_tf = tf.reshape(img2_tf, [1, 192, 192, 4])

        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        post, prior = hpu_net.dist(img1_tf, img2_tf)

        nb_prob_scale = len(post)
        post_mean_tf = [tf.reshape(post[i].entropy(), [-1]) for i in range(nb_prob_scale)]
        prior_mean_tf = [tf.reshape(prior[i].entropy(), [-1]) for i in range(nb_prob_scale)]

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        post_mean_np = {}
        prior_mean_np = {}
        tot_nb_units = 0
        for level in range(nb_prob_scale):
            nb_units = post_mean_tf[level].get_shape().as_list()[0]
            post_mean_np['scale-{}'.format(level)] = np.zeros((nb_imgs, nb_units))
            prior_mean_np['scale-{}'.format(level)] = np.zeros((nb_imgs, nb_units))
            tot_nb_units += nb_units

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):
                print(i)
                post_mean, prior_mean = sess.run([post_mean_tf, prior_mean_tf])

                for level in range(nb_prob_scale):
                    post_mean_np['scale-{}'.format(level)][i, ...] = post_mean[level]
                    prior_mean_np['scale-{}'.format(level)][i, ...] = prior_mean[level]

        entropy_post_all_units = np.zeros(tot_nb_units)
        entropy_prior_all_units = np.zeros(tot_nb_units)
        x_labels = []
        i = 0
        for level in range(nb_prob_scale):
            nb_units = np.shape(post_mean_np['scale-{}'.format(level)])[1]
            entropy_post_all_units[i: i + nb_units] = np.mean(post_mean_np['scale-{}'.format(level)], axis=0)
            entropy_prior_all_units[i: i + nb_units] = np.mean(prior_mean_np['scale-{}'.format(level)], axis=0)
            x_labels += [str(level)] + [''] * (nb_units - 1)
            i += nb_units

        fig = plt.figure(figsize=(16, 4))
        plt.scatter(np.arange(len(x_labels)), entropy_post_all_units, label='post', alpha=.5, color='red')
        plt.scatter(np.arange(len(x_labels)), entropy_prior_all_units, label='prior', alpha=.5, color='blue')
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
        plt.ylim([-0.5, 2.5])
        plt.legend()
        fig.savefig(os.path.join(self.outdir, self.expt_name, "covs", self.filename + "{}_one.png".format(epoch)))
        plt.close()


class Dist:
    def __init__(self, config, name='valid', reuse=False):
        super(Dist, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.save_np = expt_config['save_np']
        self.mode = expt_config['mode']
        self.reuse = reuse
        self.filename = 'dist_' + name + '_'

    def _resize_down2d(self, input_features, scale=2):
        assert scale >= 1
        return tf.nn.max_pool2d(input_features,
                                ksize=(1, scale, scale, 1),
                                strides=(1, scale, scale, 1),
                                padding='VALID')

    def _get_labels(self, labels):
        labels_level = {}
        for i in range(7):
            scale = 3 if i == 6 else 2
            labels = self._resize_down2d(labels, scale)
            labels_level[6 - i] = tf.reshape(labels, [-1])
        return labels_level

    def on_epoch_end(self, callback_data, nb_imgs, mode_value, epoch):

        self.callback_data = callback_data

        # set up the model
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        img2_tf = (next_batch[1][..., :4] - next_batch[0][..., :4])
        img1_tf = next_batch[0][..., :4]
        y_true_newt2_tf = next_batch[1][..., -1]

        img1_tf = tf.reshape(img1_tf, [1, 192, 192, 4])
        img2_tf = tf.reshape(img2_tf, [1, 192, 192, 4])
        y_true_newt2_tf = tf.reshape(y_true_newt2_tf, [1, 192, 192, 1])
        y_true_newt2_tf = self._get_labels(y_true_newt2_tf)

        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        post, prior = hpu_net.dist(img1_tf, img2_tf)

        nb_prob_scale = len(post)
        post_mean_tf = [tf.reshape(post[i].loc, [-1]) for i in range(nb_prob_scale)]
        prior_mean_tf = [tf.reshape(prior[i].loc, [-1]) for i in range(nb_prob_scale)]
        post_std_tf = [tf.reshape(post[i].scale.diag, [-1]) for i in range(nb_prob_scale)]
        prior_std_tf = [tf.reshape(prior[i].scale.diag, [-1]) for i in range(nb_prob_scale)]

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):
                [batch, post_mean, post_std, prior_mean, prior_std, y_true_newt2] = sess.run([next_batch, post_mean_tf, post_std_tf, prior_mean_tf, prior_std_tf, y_true_newt2_tf])

                y_true_label = len(np.unique(batch[1][0, ..., -1])) - 1

                for level in range(nb_prob_scale):

                    print("here:", np.shape(y_true_newt2[level]))
                    y_true_newt2_level = y_true_newt2[level]

                    nb_units = len(post_mean[level])
                    fig = plt.figure(figsize=(4 * (level + 1), 4 * (level + 1)))
                    for indx in range(nb_units):

                        samples_post = np.random.normal(post_mean[level][indx], post_std[level][indx], 1000)
                        samples_prior = np.random.normal(prior_mean[level][indx], prior_std[level][indx], 1000)
                        samples_post_noise = samples_post + np.random.normal(np.zeros_like(post_mean[level][indx]), 3 * post_std[level][indx], 1000)

                        samples_post = np.reshape(samples_post, [-1])
                        samples_prior = np.reshape(samples_prior, [-1])

                        plt.subplot(int(np.sqrt(nb_units)), int(np.sqrt(nb_units)), indx+1)
                        plt.hist(samples_prior, bins=100, color='blue', label='prior', alpha=.5)
                        plt.hist(samples_post, bins=100, color='red', label='post', alpha=.5)
                        #plt.hist(samples_post_noise, bins=100, color='green', label='post-noise', alpha=.5)
                        plt.axvline(x=post_mean[level][indx], color='red', linewidth=1, linestyle='dashed')
                        plt.axvline(x=prior_mean[level][indx], color='blue', linewidth=1, linestyle='dashed')
                        min_ylim, max_ylim = plt.ylim()
                        min_xlim, max_xlim = plt.xlim()
                        plt.text(post_mean[level][indx] * 1.1, max_ylim * 0.95, 'Mean: {:.2f}'.format(post_mean[level][indx]), color='red')
                        plt.text(prior_mean[level][indx] * 1.1, max_ylim * 0.8, 'Mean: {:.2f}'.format(prior_mean[level][indx]), color='blue')
                        plt.axhline(y=max_ylim * 0.3, color='red', linewidth=1, linestyle='dashed')
                        plt.axhline(y=max_ylim * 0.45, color='blue', linewidth=1, linestyle='dashed')
                        plt.text(max_xlim * 0.3, max_ylim * 0.33, 'Std: {:.2f}'.format(post_std[level][indx]), color='red')
                        plt.text(max_xlim * 0.45, max_ylim * 0.48, 'Std: {:.2f}'.format(prior_std[level][indx]), color='blue')
                        plt.title(y_true_newt2_level[indx])
                        plt.axis('off')

                    plt.legend()
                    #plt.title("label:{}".format(y_true_label))

                    fig.savefig(
                        os.path.join(self.outdir, self.expt_name, "dists", self.filename + "{}_{}_{}.png".format(level, i, epoch)))
                    plt.close()


class GEDImages:
    def __init__(self, config, name, reuse=False):
        super(GEDImages, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.reuse = reuse
        self.filename = 'ged_image_' + name + '_'

    def IoU(self, y1, y2):
        intersection = np.sum(y1 * y2, axis=(1, 2))
        union = np.sum(y1, axis=(1, 2)) + np.sum(y2, axis=(1, 2))
        dice = 2. * intersection / (union + 0.0001)
        dice[union == 0] = 1.
        d = 1 - dice
        return d

    def on_epoch_end(self, callback_data, nb_imgs, mode_value, nb_samples, epoch):
        self.callback_data = callback_data
        print("reconstructing...")

        # set up the model
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        img1_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])
        img2_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])

        name = 'prior'

        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        y_pred_newt2_tf = tf.nn.sigmoid(hpu_net.sample(img1_tf, img2_tf, name))

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        colormap = {0:'blue', 1: 'red'}
        labels = {0: 'inactive', 1: 'active'}

        GED = np.zeros(nb_imgs)
        S = np.zeros(nb_imgs)
        y_true_labels = np.zeros(nb_imgs)
        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                batch_np = sess.run(next_batch)
                img2_np = (batch_np[1][..., :4] - batch_np[0][..., :4])
                img1_np = batch_np[0][..., :4]
                y_true_newt2 = batch_np[1][..., -1]
                y_true_label = len(np.unique(y_true_newt2)) - 1
                y_true_labels[i] = y_true_label

                y_pred_newt2_samples = np.zeros((nb_samples, 192, 192))
                for itr in range(nb_samples):
                    pred_newt2 = sess.run(y_pred_newt2_tf, feed_dict={img1_tf: img1_np,
                                                                      img2_tf: img2_np})
                    y_pred_newt2 = pred_newt2[0, ..., 0]
                    y_pred_newt2_samples[itr, ...] = (y_pred_newt2 > 0.58).astype(np.float32)
                y_pred_newt2_samples_1 = np.repeat(y_pred_newt2_samples, nb_samples, axis=0)
                y_pred_newt2_samples_2 = np.tile(y_pred_newt2_samples, [nb_samples, 1, 1])
                Iou_per_sample = self.IoU(y_pred_newt2_samples_1, y_pred_newt2_samples_2)
                Iou_samples = np.sum(Iou_per_sample)/(nb_samples ** 2)
                y_true_newt2_per_sample = np.repeat(y_true_newt2, nb_samples, axis=0)
                Iou_gt = np.sum(self.IoU(y_pred_newt2_samples, y_true_newt2_per_sample))/nb_samples
                GED[i] = 2 * Iou_gt - Iou_samples
                S[i] = Iou_gt
                print(i, S[i], GED[i])

        GED_active = GED[y_true_labels == 1]
        GED_inactive = GED[y_true_labels == 0]
        GED_all = {0: GED_inactive, 1: GED_active}

        S_active = S[y_true_labels == 1]
        S_inactive = S[y_true_labels == 0]
        S_all = {0: S_inactive, 1: S_active}
        sample_indices = np.arange(nb_imgs)

        fig = plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        for i in range(2):
            ax1.scatter(sample_indices[y_true_labels == i], GED_all[i], alpha=.5, color=colormap[i], label=labels[i])
            ax1.set_ylabel('GED')
            ax1.set_title('avg GED:{:.2f}'.format(np.mean(GED)))
            ax2.scatter(sample_indices[y_true_labels == i], S_all[i], alpha=.5, color=colormap[i], label=labels[i])
            ax2.set_ylabel('d(S, Y)')
            ax2.set_title('avg d:{:.2f}'.format(np.mean(S)))
            ax2.set_xlabel('sample')
            plt.legend()
        fig.savefig(os.path.join(self.outdir, self.expt_name, "covs", self.filename + name + "_{}.png".format(epoch)))
        plt.close()


class GEDSamples:
    def __init__(self, config, name, reuse=False):
        super(GEDSamples, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.reuse = reuse
        self.filename = 'ged_sample_' + name + '_'

    def IoU(self, y1, y2):
        intersection = np.sum(y1 * y2, axis=(1, 2))
        union = np.sum(y1, axis=(1, 2)) + np.sum(y2, axis=(1, 2))
        dice = 2. * intersection / (union + 0.0001)
        dice[union == 0] = 1.
        d = 1 - dice
        return d

    def on_epoch_end(self, callback_data, nb_imgs, mode_value, nb_samples, epoch):
        self.callback_data = callback_data
        print("reconstructing...")

        # set up the model
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        img1_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])
        img2_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])

        name = 'prior'

        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        y_pred_newt2_tf = tf.nn.sigmoid(hpu_net.sample(img1_tf, img2_tf, name))

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            GED = np.zeros(nb_samples)
            S = np.zeros(nb_samples)

            for j in range(nb_samples):

                GED_th = np.zeros(nb_imgs)
                S_th = np.zeros(nb_imgs)

                for i in range(nb_imgs):

                    print("nb samples:{}, img:{}".format(j, i))

                    batch_np = sess.run(next_batch)
                    img2_np = (batch_np[1][..., :4] - batch_np[0][..., :4])
                    img1_np = batch_np[0][..., :4]
                    y_true_newt2 = batch_np[1][..., -1]

                    y_pred_newt2_samples = np.zeros((nb_samples, 192, 192))

                    for itr in range(j):
                        pred_newt2 = sess.run(y_pred_newt2_tf, feed_dict={img1_tf: img1_np,
                                                                          img2_tf: img2_np})

                        y_pred_newt2 = pred_newt2[0, ..., 0]
                        y_pred_newt2_samples[itr, ...] = (y_pred_newt2>0.7).astype(np.float32)

                    y_pred_newt2_samples_1 = np.repeat(y_pred_newt2_samples, nb_samples, axis=0)
                    y_pred_newt2_samples_2 = np.tile(y_pred_newt2_samples, [nb_samples, 1, 1])
                    Iou_per_sample = self.IoU(y_pred_newt2_samples_1, y_pred_newt2_samples_2)
                    Iou_samples = np.sum(Iou_per_sample)/(nb_samples ** 2)
                    y_true_newt2_per_sample = np.repeat(y_true_newt2, nb_samples, axis=0)
                    Iou_gt = np.sum(self.IoU(y_pred_newt2_samples, y_true_newt2_per_sample))/nb_samples
                    GED_th[i] = 2 * Iou_gt - Iou_samples
                    S_th[i] = Iou_gt

                GED[j] = np.mean(GED_th)
                S[j] = np.mean(S_th)

        fig = plt.figure(figsize=(16, 4))
        plt.scatter(np.arange(1, nb_samples + 1), GED, alpha=.5, color='blue', label='GED')
        plt.scatter(np.arange(1, nb_samples + 1), S, alpha=.5, color='red', label='d(S, Y)')
        plt.xlabel('#sample')
        plt.legend()
        fig.savefig(os.path.join(self.outdir, self.expt_name, "covs", self.filename + name +"_{}.png".format(epoch)))
        plt.close()


class PlotRoc:
    def __init__(self, config, name='valid', reuse=False):
        super(PlotRoc, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.save_np = expt_config['save_np']
        self.mode = expt_config['mode']
        self.reuse = reuse
        self.filename = 'roc_class_' + name + '_'

    def on_epoch_end(self, callback_data, nb_imgs, mode_value, epoch):

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
        y_pred_post_tf = tf.nn.sigmoid(hpu_net.classify(img1_tf, img2_tf, 'post'))
        y_pred_prior_tf = tf.nn.sigmoid(hpu_net.classify(img1_tf, img2_tf))

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))

        #from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        #print_tensors_in_checkpoint_file(checkpoint, all_tensors=False, tensor_name='', all_tensor_names=True)

        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        y_true_labels = np.zeros(nb_imgs)
        y_pred_post_labels = np.zeros(nb_imgs)
        y_pred_prior_labels = np.zeros(nb_imgs)
        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):
                y_pred_post, y_pred_prior, batch = sess.run([y_pred_post_tf, y_pred_prior_tf, next_batch])
                y_pred_post = y_pred_post[0][0]
                y_pred_prior = y_pred_prior[0][0]
                y_true = batch[2][0]
                y_true_labels[i] = y_true
                y_pred_post_labels[i] = y_pred_post
                y_pred_prior_labels[i] = y_pred_prior

        fpr_post, tpr_post, thr = roc_curve(y_true_labels, y_pred_post_labels)
        fpr_prior, tpr_prior, _ = roc_curve(y_true_labels, y_pred_prior_labels)
        indx = np.argmin(abs(fpr_post - 0.2))
        print("HERE IS THE THRESHOLD:", thr[indx], fpr_post[indx])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # plt.plot(fpr_mean, tpr_mean, label='fp-tp')
        plt.plot(fpr_post, tpr_post, label='post-auc:{:.2f}'.format(auc(fpr_post, tpr_post)))
        plt.plot(fpr_prior, tpr_prior, label='prior-auc:{:.2f}'.format(auc(fpr_prior, tpr_prior)))
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
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        #ax.set_title("auc:{}".format(auc(fdr_mean, tpr_mean)))
        fig.savefig(os.path.join(self.outdir, self.expt_name, "rocs", self.filename + "{}.png".format(epoch)))
        plt.close()


class PlotRocProb:
    def __init__(self, config, name='valid', reuse=False):
        super(PlotRocProb, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.save_np = expt_config['save_np']
        self.mode = expt_config['mode']
        self.reuse = reuse
        self.filename = 'roc_class_' + name + '_'

    def on_epoch_end(self, callback_data, nb_imgs, nb_samples, mode_value, epoch):

        self.callback_data = callback_data

        # set up the model
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        img1_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])
        img2_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])

        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        y_pred_post_tf = tf.nn.sigmoid(hpu_net.classify(img1_tf, img2_tf, 'post'))
        y_pred_prior_tf = tf.nn.sigmoid(hpu_net.classify(img1_tf, img2_tf))

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))

        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        y_true_labels = np.zeros(nb_imgs)
        y_pred_post_labels = np.zeros(nb_imgs)
        y_pred_prior_labels = np.zeros(nb_imgs)

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                print(i)
                batch = sess.run(next_batch)

                img1_np = batch[0][..., :4]
                img2_np = (batch[1][..., :4] - batch[0][..., :4])

                y_pred_post = 0.
                y_pred_prior = 0.

                for _ in range(nb_samples):
                    y_pred_post, y_pred_prior = sess.run([y_pred_post_tf, y_pred_prior_tf],
                                                                feed_dict={img1_tf: img1_np,
                                                                           img2_tf: img2_np})
                    y_pred_post += int(y_pred_post[0][0] > 0.7)
                    y_pred_prior += int(y_pred_prior[0][0] > 0.7)
                    y_true = batch[2][0]

                y_true_labels[i] = y_true
                y_pred_post_labels[i] = y_pred_post/ nb_samples
                y_pred_prior_labels[i] = y_pred_prior/ nb_samples

        fpr_post, tpr_post, _ = roc_curve(y_true_labels, y_pred_post_labels)
        fpr_prior, tpr_prior, _ = roc_curve(y_true_labels, y_pred_prior_labels)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # plt.plot(fpr_mean, tpr_mean, label='fp-tp')
        plt.plot(fpr_post, tpr_post, label='post-auc:{:.2f}'.format(auc(fpr_post, tpr_post)))
        plt.plot(fpr_prior, tpr_prior, label='prior-auc:{:.2f}'.format(auc(fpr_prior, tpr_prior)))
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
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        #ax.set_title("auc:{}".format(auc(fdr_mean, tpr_mean)))
        fig.savefig(os.path.join(self.outdir, self.expt_name, "rocs", self.filename + "avg_GEDImages{}.png".format(epoch)))
        plt.close()


class PlotRocSegProb:
    def __init__(self, config, name='valid', reuse=False):
        super(PlotRocSegProb, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.save_np = expt_config['save_np']
        self.mode = expt_config['mode']
        self.reuse = reuse
        self.filename = 'roc_class_' + name + '_'

    def on_epoch_end(self, callback_data, nb_imgs, nb_samples, mode_value, epoch):

        self.callback_data = callback_data

        # set up the model
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        thresholds = [0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99,
                      0.999, 1]
        nb_thrs = len(thresholds)

        fpr_post = np.empty((nb_imgs, nb_thrs))
        tpr_post = np.empty((nb_imgs, nb_thrs))
        fdr_post = np.empty((nb_imgs, nb_thrs))

        fpr_prior = np.empty((nb_imgs, nb_thrs))
        tpr_prior = np.empty((nb_imgs, nb_thrs))
        fdr_prior = np.empty((nb_imgs, nb_thrs))

        img1_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])
        img2_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])

        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        y_pred_newt2_post_tf = tf.nn.sigmoid(hpu_net.sample(img1_tf, img2_tf, 'post'))
        y_pred_newt2_prior_tf = tf.nn.sigmoid(hpu_net.sample(img1_tf, img2_tf))

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))

        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        y_true_labels = np.zeros((nb_imgs, 192, 192))
        y_pred_post_segs = np.zeros((nb_imgs, 192, 192))
        y_pred_prior_segs = np.zeros((nb_imgs, 192, 192))
        thr_fdr = .99

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                print(i)
                batch = sess.run(next_batch)

                img1_np = batch[0][..., :4]
                img2_np = (batch[1][..., :4] - batch[0][..., :4])

                y_pred_post = 0.
                y_pred_prior = 0.

                for _ in range(nb_samples):
                    y_pred_post_newt2, y_pred_prior_newt2 = sess.run([y_pred_newt2_post_tf, y_pred_newt2_prior_tf],
                                                                     feed_dict={img1_tf: img1_np,
                                                                                img2_tf: img2_np})

                    y_pred_post += (y_pred_post_newt2[0][..., 0] > thr_fdr).astype(np.float32)
                    y_pred_prior += (y_pred_prior_newt2[0][..., 0] > thr_fdr).astype(np.float32)
                    y_true = (batch[1][0, ..., -1] > 0).astype(int)

                y_true_labels[i, ...] = y_true
                y_pred_post_segs[i, ...] = y_pred_post/ nb_samples
                y_pred_prior_segs[i, ...] = y_pred_prior/ nb_samples

        for i in range(nb_imgs):
            for j, thr in enumerate(thresholds):
                y_pred_t = (y_pred_post_segs[i, ...] >= thr).astype(int)
                total_p = np.sum(y_true)
                tp = np.sum(y_true * y_pred_t)
                fp = np.sum((1 - y_true) * y_pred_t)
                fn = np.sum(y_true * (1 - y_pred_t))
                tn = np.sum((1 - y_true) * (1 - y_pred_t))
                tpr_post[i, j] = tp / (tp + fn + .0001)
                if total_p == 0:
                    tpr_post[i, j] = 1
                fpr_post[i, j] = fp / (fp + tn + .0001)
                fdr_post[i, j] = fp / (tp + fp + .0001)

                y_pred_t = (y_pred_prior_segs[i, ...] >= thr).astype(int)
                total_p = np.sum(y_true)
                tp = np.sum(y_true * y_pred_t)
                fp = np.sum((1 - y_true) * y_pred_t)
                fn = np.sum(y_true * (1 - y_pred_t))
                tn = np.sum((1 - y_true) * (1 - y_pred_t))
                tpr_prior[i, j] = tp / (tp + fn + .0001)
                if total_p == 0:
                    tpr_prior[i, j] = 1
                fpr_prior[i, j] = fp / (fp + tn + .0001)
                fdr_prior[i, j] = fp / (tp + fp + .0001)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # plt.plot(fpr_mean, tpr_mean, label='fp-tp')
        plt.plot(fpr_post, tpr_post, label='post-auc:{:.2f}'.format(auc(fpr_post, tpr_post)))
        plt.plot(fpr_prior, tpr_prior, label='prior-auc:{:.2f}'.format(auc(fpr_prior, tpr_prior)))
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
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        #ax.set_title("auc:{}".format(auc(fdr_mean, tpr_mean)))
        fig.savefig(os.path.join(self.outdir, self.expt_name, "rocs", self.filename + "avg_GEDImages{}.png".format(epoch)))
        plt.close()


class PlotRocLevels:
    def __init__(self, config, reuse=False):
        super(PlotRocLevels, self).__init__()
        self.callback_data = None
        self.config = config
        self.reuse = reuse
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config["callback"]['mode']
        self.save_np = expt_config['save_np']
        self.filename = 'roc_levels_'

    def _resize_down2d(self, input_features, scale=2):
        assert scale >= 1
        return tf.nn.avg_pool2d(input_features,
                                ksize=(1, scale, scale, 1),
                                strides=(1, scale, scale, 1),
                                padding='VALID')

    def _get_labels(self, labels):
        labels_level = {}
        for i in range(1, 8):
            scale = 3 if i == 7 else 2
            labels_level[8 - i] = labels
            labels = self._resize_down2d(labels, scale)
        return labels_level

    def on_epoch_end(self, callback_data, nb_imgs, mode_value, epoch):

        self.callback_data = callback_data

        # set up the model
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        img1_tf = next_batch[0][..., :4]
        img2_tf = (next_batch[1][..., :4] - next_batch[0][..., :4])
        y_true_newt2_tf = next_batch[1][..., -1]
        img1_tf = tf.reshape(img1_tf, [1, 192, 192, 4])
        img2_tf = tf.reshape(img2_tf, [1, 192, 192, 4])
        y_true_newt2_tf = tf.reshape(y_true_newt2_tf, [1, 192, 192, 1])

        name = 'post'
        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        y_pred_newt2_tf = tf.nn.sigmoid(hpu_net.sample(img1_tf, img2_tf, name))
        segmentations = hpu_net.reconstruct_latent(img1_tf, img2_tf, name)
        y_true_newt2_tf_scales = self._get_labels(y_true_newt2_tf)

        y_pred_newt2_tf_1 = tf.nn.sigmoid(segmentations[0])
        y_pred_newt2_tf_2 = tf.nn.sigmoid(segmentations[1])
        y_pred_newt2_tf_3 = tf.nn.sigmoid(segmentations[2])

        y_true_newt2_tf_1 = y_true_newt2_tf_scales[1]
        y_true_newt2_tf_2 = y_true_newt2_tf_scales[2]
        y_true_newt2_tf_3 = y_true_newt2_tf_scales[3]

        thresholds = [0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99,
                      0.999, 1]
        nb_thrs = len(thresholds)

        fpr = np.empty((nb_imgs, nb_thrs))
        tpr = np.empty((nb_imgs, nb_thrs))
        fdr = np.empty((nb_imgs, nb_thrs))

        fpr_rep_1 = np.empty((nb_imgs, nb_thrs))
        tpr_rep_1 = np.empty((nb_imgs, nb_thrs))
        fdr_rep_1 = np.empty((nb_imgs, nb_thrs))

        fpr_rep_2 = np.empty((nb_imgs, nb_thrs))
        tpr_rep_2 = np.empty((nb_imgs, nb_thrs))
        fdr_rep_2 = np.empty((nb_imgs, nb_thrs))

        fpr_rep_3 = np.empty((nb_imgs, nb_thrs))
        tpr_rep_3 = np.empty((nb_imgs, nb_thrs))
        fdr_rep_3 = np.empty((nb_imgs, nb_thrs))

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                print(i)

                y_pred_newt2, y_pred_newt2_1, y_pred_newt2_2, y_pred_newt2_3, y_true_newt2, y_true_newt2_1, y_true_newt2_2, y_true_newt2_3 = \
                    sess.run([y_pred_newt2_tf, y_pred_newt2_tf_1, y_pred_newt2_tf_2, y_pred_newt2_tf_3,
                              y_true_newt2_tf, y_true_newt2_tf_1, y_true_newt2_tf_2, y_true_newt2_tf_3])

                y_pred_newt2 = y_pred_newt2[0][..., 0]
                y_pred_newt2_1 = y_pred_newt2_1[0][..., 0]
                y_pred_newt2_2 = y_pred_newt2_2[0][..., 0]
                y_pred_newt2_3 = y_pred_newt2_3[0][..., 0]
                y_true_newt2 = y_true_newt2[0][..., 0]
                y_true_newt2_1 = y_true_newt2_1[0][..., 0]
                y_true_newt2_2 = y_true_newt2_2[0][..., 0]
                y_true_newt2_3 = y_true_newt2_3[0][..., 0]

                for j, thr in enumerate(thresholds):
                    y_pred_t = (y_pred_newt2 >= thr).astype(int)
                    total_p = np.sum(y_true_newt2)
                    tp = np.sum(y_true_newt2 * y_pred_t)
                    fp = np.sum((1 - y_true_newt2) * y_pred_t)
                    fn = np.sum(y_true_newt2 * (1 - y_pred_t))
                    tn = np.sum((1 - y_true_newt2) * (1 - y_pred_t))
                    tpr[i, j] = tp / (tp + fn + .0001)
                    if total_p == 0:
                        tpr[i, j] = 1
                    fpr[i, j] = fp / (fp + tn + .0001)
                    fdr[i, j] = fp / (tp + fp + .0001)

                    ### rep_1
                    y_pred_t = (y_pred_newt2_1 >= thr).astype(int)
                    total_p = np.sum(y_true_newt2_1)
                    tp = np.sum(y_true_newt2_1 * y_pred_t)
                    fp = np.sum((1 - y_true_newt2_1) * y_pred_t)
                    fn = np.sum(y_true_newt2_1 * (1 - y_pred_t))
                    tn = np.sum((1 - y_true_newt2_1) * (1 - y_pred_t))
                    tpr_rep_1[i, j] = tp / (tp + fn + .0001)
                    if total_p == 0:
                        tpr_rep_1[i, j] = 1
                    fpr_rep_1[i, j] = fp / (fp + tn + .0001)
                    fdr_rep_1[i, j] = fp / (tp + fp + .0001)

                    ### rep_2
                    y_pred_t = (y_pred_newt2_2 >= thr).astype(int)
                    total_p = np.sum(y_true_newt2_2)
                    tp = np.sum(y_true_newt2_2 * y_pred_t)
                    fp = np.sum((1 - y_true_newt2_2) * y_pred_t)
                    fn = np.sum(y_true_newt2_2 * (1 - y_pred_t))
                    tn = np.sum((1 - y_true_newt2_2) * (1 - y_pred_t))
                    tpr_rep_2[i, j] = tp / (tp + fn + .0001)
                    if total_p == 0:
                        tpr_rep_2[i, j] = 1
                    fpr_rep_2[i, j] = fp / (fp + tn + .0001)
                    fdr_rep_2[i, j] = fp / (tp + fp + .0001)

                    ### rep_3
                    y_pred_t = (y_pred_newt2_3 >= thr).astype(int)
                    total_p = np.sum(y_true_newt2_3)
                    tp = np.sum(y_true_newt2_3 * y_pred_t)
                    fp = np.sum((1 - y_true_newt2_3) * y_pred_t)
                    fn = np.sum(y_true_newt2_3 * (1 - y_pred_t))
                    tn = np.sum((1 - y_true_newt2_3) * (1 - y_pred_t))
                    tpr_rep_3[i, j] = tp / (tp + fn + .0001)
                    if total_p == 0:
                        tpr_rep_3[i, j] = 1
                    fpr_rep_3[i, j] = fp / (fp + tn + .0001)
                    fdr_rep_3[i, j] = fp / (tp + fp + .0001)

        fdr_mean = np.mean(fdr, axis=0)
        tpr_mean = np.mean(tpr, axis=0)

        fdr_rep_mean_1 = np.mean(fdr_rep_1, axis=0)
        tpr_rep_mean_1 = np.mean(tpr_rep_1, axis=0)

        fdr_rep_mean_2 = np.mean(fdr_rep_2, axis=0)
        tpr_rep_mean_2 = np.mean(tpr_rep_2, axis=0)

        fdr_rep_mean_3 = np.mean(fdr_rep_3, axis=0)
        tpr_rep_mean_3 = np.mean(tpr_rep_3, axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(fdr_mean, tpr_mean, label='fd-tp')
        plt.plot(fdr_rep_mean_1, tpr_rep_mean_1, label='fd-tp/scale-1')
        plt.plot(fdr_rep_mean_2, tpr_rep_mean_2, label='fd-tp/scale-2')
        plt.plot(fdr_rep_mean_3, tpr_rep_mean_3, label='fd-tp/scale-3')
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
        # ax.set_title("auc:{}".format(auc(fdr_mean, tpr_mean)))
        fig.savefig(os.path.join(self.outdir, self.expt_name, "rocs", self.filename + name + "_{}.png".format(epoch)))
        plt.close()


class VisualizeSamplesLevels:
    def __init__(self, config, name, reuse=False):
        super(VisualizeSamplesLevels, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.reuse = reuse
        self.filename = 'recon_sample_' + name + '_'

    def _get_labels(self, labels):
        labels_level = {}
        for i in range(1, 8):
            scale = 3 if i == 7 else 2
            labels_level[8 - i] = labels
            labels = self._resize_down2d(labels, scale)
        return labels_level

    def _resize_down2d(self, input_features, scale=2):
        """Average pooling rescaling-operation for the input features.

        Args:
          input_features: A tensor of shape (b, h, w, c).
          scale: An integer specifying the scaling factor.
        Returns: A tensor of shape (b, h / scale, w / scale, c).
        """
        assert scale >= 1
        return tf.nn.avg_pool2d(input_features,
                                ksize=(1, scale, scale, 1),
                                strides=(1, scale, scale, 1),
                                padding='VALID')

    def on_epoch_end(self, callback_data, nb_imgs, nb_samples, level, epoch):
        self.callback_data = callback_data
        print("reconstructing...")

        fig_seg = plt.figure(figsize=(3 * (4 + nb_samples + 2), 3 * nb_imgs))
        gs_seg_outer = gridspec.GridSpec(1, 4 + nb_samples + 2, hspace=-.01, wspace=-0.01)

        gs_x1 = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[0], wspace=0, hspace=0)
        gs_x2 = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[1], wspace=0, hspace=0)
        gs_diff = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[2], wspace=0, hspace=0)
        gs_true_newt2 = gridspec.GridSpecFromSubplotSpec(nb_imgs, 1, subplot_spec=gs_seg_outer[3], wspace=0, hspace=0)

        gs = []
        for itr in range(nb_samples+2):
            gs.append(gridspec.GridSpecFromSubplotSpec(nb_imgs, 1,
                                                       subplot_spec=gs_seg_outer[4 + itr],
                                                       wspace=0,
                                                       hspace=0))

        # set up the model
        img_res = 3 * (2**(level - 1))
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()
        y_true_newt2_tf = self._get_labels(tf.reshape(next_batch[1][..., -1], [1, 192, 192, 1]))
        y_true_newt2_tf = y_true_newt2_tf[level]

        name = 'post'
        img1_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])
        img2_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])
        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        y_pred_newt2_tf = tf.nn.sigmoid(hpu_net.reconstruct_latent(img1_tf, img2_tf, name)[level - 1])

        x1_tf = self._get_labels(tf.reshape(next_batch[0][..., 2], [1, 192, 192, 1]))[level]
        x2_tf = self._get_labels(tf.reshape(next_batch[1][..., 2], [1, 192, 192, 1]))[level]

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                batch_np, y_true_newt2, x1, x2 = sess.run([next_batch, y_true_newt2_tf, x1_tf, x2_tf])

                img2_np = (batch_np[1][..., :4] - batch_np[0][..., :4])
                img1_np = batch_np[0][..., :4]

                y_true_newt2 = y_true_newt2[0, ..., 0]

                x1 = x1[0, ..., 0]
                x2 = x2[0, ..., 0]

                # modify arrays for plotting
                x1 = x1[::-1, :]
                x2 = x2[::-1, :]
                x2_scaled = x2 * 0.5

                x3 = np.zeros((img_res, img_res, 3))
                x3[..., 0] = x2_scaled
                x3[..., 1] = x2_scaled
                x3[..., 2] = x2_scaled

                print("HEreee:", np.shape(y_true_newt2[::-1, ...]), np.shape(x3))

                x3[..., 0] += 0.5 * (y_true_newt2[::-1, ...]>0).astype(int)
                x3 = (x3 - np.min(x3)) / (np.max(x3) - np.min(x3))

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

                y_pred_newt2_samples = np.zeros((nb_samples, img_res, img_res))

                for itr in range(nb_samples):

                    print(i, itr)

                    pred_newt2 = sess.run(y_pred_newt2_tf, feed_dict={img1_tf: img1_np,
                                                                      img2_tf: img2_np})

                    print(np.shape(pred_newt2[0, ...]), np.min(pred_newt2[0, ...]), np.max(pred_newt2[0, ...]))
                    y_pred_newt2 = pred_newt2[0, ..., 0]
                    y_pred_newt2_samples[itr, ...] = y_pred_newt2

                    x4 = np.zeros((img_res, img_res, 3))
                    x4[..., 0] = x2_scaled
                    x4[..., 1] = x2_scaled
                    x4[..., 2] = x2_scaled

                    x4[..., 0] += 0.5 * (y_pred_newt2[::-1, ...])
                    x4 = (x4 - np.min(x4)) / (np.max(x4) - np.min(x4))

                    ax = fig_seg.add_subplot(gs[itr][i])
                    ax.imshow(x4, cmap='Greys_r')
                    ax.set_title("sample-{}".format(itr + 1))
                    ax.axis('off')

                y_pred_newt2_avg = np.sum(y_pred_newt2_samples, axis=0) / nb_samples
                x4 = y_pred_newt2_avg[::-1, ...]
                ax = fig_seg.add_subplot(gs[nb_samples][i])
                ax.imshow(x4, cmap='jet')
                ax.axis('off')
                ax.set_title("pred-avg")

                y_pred_newt2_std = np.std(y_pred_newt2_samples, axis=0)
                x4 = y_pred_newt2_std[::-1, ...]
                ax = fig_seg.add_subplot(gs[nb_samples + 1][i])
                ax.imshow(x4, cmap='jet')
                ax.axis('off')
                ax.set_title("pred-std")

            fig_seg.savefig(os.path.join(self.outdir, self.expt_name, "recons", self.filename + name + "_level_{}_{}.png".format(level, epoch)))
            plt.close()


