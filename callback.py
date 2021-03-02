import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import tensorflow as tf
from model2d_pred import HierarchicalProbUNet


class VisualizeSamples3d:
    def __init__(self, config, reuse=False):
        super(VisualizeSamples3d, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.prob = expt_config['prob']
        self.slice = 30
        self.reuse = reuse
        self.nb_latent = self.config["train"]["nb_latent"]
        self.filename = 'recon_sample_'

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def on_epoch_end(self, callback_data, nb_imgs, mode_value, nb_samples, epoch):
        slice_ = self.slice
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
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        img_tf = tf.placeholder(tf.float32, [1, 192, 192, 64, 4])
        seg_tf = tf.placeholder(tf.float32, [1, 192, 192, 64, 2])

        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        y_pred_newt2_tf = hpu_net.sample(seg_tf, img_tf)

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                batch_np = sess.run(next_batch)

                img_np = (batch_np[1][..., :4] - batch_np[0][..., :4]) * batch_np[0][..., :4]
                seg_np = (batch_np[1][..., -1]>0).astype(np.int)
                seg_np = np.stack([1 - seg_np, seg_np], axis=-1)
                y_true_newt2 = batch_np[1][0, ..., -1]

                # find the best slice to display
                sum_pixels = y_true_newt2.sum(axis=0).sum(axis=0)
                if np.max(sum_pixels) > 0:
                    slice_ = np.argmax(sum_pixels)

                x1 = batch_np[0][0, ..., :4]
                x2 = batch_np[1][0, ..., :4]

                # modify arrays for plotting
                x1 = x1[::-1, :, slice_, 2]
                x2 = x2[::-1, :, slice_, 2]
                x2_scaled = x2 * 0.5

                x3 = np.zeros((192, 192, 3))
                x3[..., 0] = x2_scaled
                x3[..., 1] = x2_scaled
                x3[..., 2] = x2_scaled

                x3[..., 0] += 0.5 * (y_true_newt2[::-1, ..., slice_]>0).astype(int)
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

                y_pred_newt2_samples = np.zeros((nb_samples, 192, 192, 64))

                for itr in range(nb_samples):

                    print(i, itr)

                    pred_newt2 = sess.run(y_pred_newt2_tf, feed_dict={img_tf: img_np,
                                                                   seg_tf: seg_np})

                    y_pred_newt2 = pred_newt2[0, ...]
                    y_pred_newt2 = np.argmax(y_pred_newt2, axis=-1)
                    y_pred_newt2_samples[itr, ...] = y_pred_newt2

                    y_pred_newt2 = y_pred_newt2[..., 1]
                    y_pred_newt2_samples[itr, ...] = y_pred_newt2

                    x4 = np.zeros((192, 192, 3))
                    x4[..., 0] = x2_scaled
                    x4[..., 1] = x2_scaled
                    x4[..., 2] = x2_scaled
                    x4[..., 0] += 0.5 * (y_pred_newt2[::-1, ..., slice_])
                    x4 = (x4 - np.min(x4)) / (np.max(x4) - np.min(x4))

                    ax = fig_seg.add_subplot(gs[itr][i])
                    ax.imshow(x4, cmap='Greys_r')
                    ax.axis('off')

                y_pred_newt2_avg = np.sum(y_pred_newt2_samples, axis=0)/nb_samples
                x4 = y_pred_newt2_avg[::-1, ..., slice_]
                ax = fig_seg.add_subplot(gs[nb_samples][i])
                ax.imshow(x4, cmap='jet')
                ax.axis('off')
                ax.set_title("pred-newt2-avg")

                y_pred_newt2_std = np.std(y_pred_newt2_samples, axis=0)
                x4 = y_pred_newt2_std[::-1, ..., slice_]
                ax = fig_seg.add_subplot(gs[nb_samples+1][i])
                ax.imshow(x4, cmap='jet')
                ax.axis('off')
                ax.set_title("pred-newt2-std")

        fig_seg.savefig(os.path.join(self.outdir, self.expt_name, "recons", self.filename + mode_value +"_{}.png".format(epoch)))
        plt.close()


class VisualizeSamples:
    def __init__(self, config, reuse=False):
        super(VisualizeSamples, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.prob = expt_config['prob']
        self.slice = 30
        self.reuse = reuse
        self.nb_latent = self.config["train"]["nb_latent"]
        self.filename = 'recon_sample_'

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def on_epoch_end(self, callback_data, nb_imgs, mode_value, nb_samples, epoch):
        slice_ = self.slice
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
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        img1_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])
        img2_tf = tf.placeholder(tf.float32, [1, 192, 192, 4])

        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        y_pred_newt2_tf = tf.nn.sigmoid(hpu_net.sample(img1_tf, img2_tf))

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                batch_np = sess.run(next_batch)

                img2_np = (batch_np[1][..., :4] - batch_np[0][..., :4])# * batch_np[0][..., :4]
                #seg_np = (batch_np[1][..., -1]>0).astype(np.int)
                img1_np = batch_np[0][..., :4]#np.stack([1 - seg_np, seg_np], axis=-1)
                y_true_newt2 = batch_np[1][0, ..., -1]

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

                y_pred_newt2_samples = np.zeros((nb_samples, 192, 192))

                for itr in range(nb_samples):

                    print(i, itr)

                    pred_newt2 = sess.run(y_pred_newt2_tf, feed_dict={img1_tf: img1_np,
                                                                      img2_tf: img2_np})

                    print(np.shape(pred_newt2[0, ...]), np.min(pred_newt2[0, ...]), np.max(pred_newt2[0, ...]))
                    y_pred_newt2 = pred_newt2[0, ..., 0]
                    #y_pred_newt2 = np.argmax(y_pred_newt2, axis=-1)
                    y_pred_newt2_samples[itr, ...] = y_pred_newt2

                    x4 = np.zeros((192, 192, 3))
                    x4[..., 0] = x2_scaled
                    x4[..., 1] = x2_scaled
                    x4[..., 2] = x2_scaled

                    x4[..., 0] += 0.8 * (y_pred_newt2[::-1, ...])  # * (1 - mask_np[::-1, ...]))
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

            fig_seg.savefig(os.path.join(self.outdir, self.expt_name, "recons", self.filename + mode_value + "_{}_prior.png".format(epoch)))
            plt.close()


class ReplaceZ:
    def __init__(self, config, reuse=False, name='post'):
        super(ReplaceZ, self).__init__()
        self.callback_data = None
        self.config = config
        self.reuse = reuse
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config["callback"]['mode']
        self.save_np = expt_config['save_np']
        self.nb_latent = self.config["train"]["nb_latent"]
        self.nb_class = self.config["train"]["nb_class"]
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

        hpu_net = HierarchicalProbUNet(name='model/HPUNet')
        y_pred_newt2_tf = tf.nn.sigmoid(hpu_net.sample(img1_tf, img2_tf, 'prior'))
        y_pred_newt2_rep_tf = tf.nn.sigmoid(hpu_net.replace_latent(img1_tf, img2_tf))

        thresholds = [0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99,
                      0.999, 1]
        nb_thrs = len(thresholds)

        fpr = np.empty((nb_imgs, nb_thrs))
        tpr = np.empty((nb_imgs, nb_thrs))
        fdr = np.empty((nb_imgs, nb_thrs))

        fpr_rep = np.empty((nb_imgs, nb_thrs))
        tpr_rep = np.empty((nb_imgs, nb_thrs))
        fdr_rep = np.empty((nb_imgs, nb_thrs))

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                print(i)

                y_pred_newt2, y_pred_newt2_rep, batch = sess.run([y_pred_newt2_tf, y_pred_newt2_rep_tf, next_batch])

                y_pred = y_pred_newt2[0][..., 0]
                y_pred_rep = y_pred_newt2_rep[0][..., 0]
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

                    y_pred_t = (y_pred_rep >= thr).astype(int)
                    total_p = np.sum(y_true)
                    tp = np.sum(y_true * y_pred_t)
                    fp = np.sum((1 - y_true) * y_pred_t)
                    fn = np.sum(y_true * (1 - y_pred_t))
                    tn = np.sum((1 - y_true) * (1 - y_pred_t))
                    tpr_rep[i, j] = tp / (tp + fn + .0001)
                    if total_p == 0:
                        tpr_rep[i, j] = 1
                    fpr_rep[i, j] = fp / (fp + tn + .0001)
                    fdr_rep[i, j] = fp / (tp + fp + .0001)

        fdr_mean = np.mean(fdr, axis=0)
        tpr_mean = np.mean(tpr, axis=0)

        fdr_rep_mean = np.mean(fdr_rep, axis=0)
        tpr_rep_mean = np.mean(tpr_rep, axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(fdr_mean, tpr_mean, label='fd-tp')
        plt.plot(fdr_rep_mean, tpr_rep_mean, label='fd-tp(zero-ed)')
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
        fig.savefig(os.path.join(self.outdir, self.expt_name, "rocs", self.filename + "{}_prior.png".format(epoch)))
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
        self.filename = 'roc_seg_' + name + '_'#+ self.mode

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

        fdr_prior_mean = np.mean(fdr_prior, axis=0)
        tpr_prior_mean = np.mean(tpr_prior, axis=0)

        #indx = np.argmin(abs(fdr_mean-0.2))
        #print("I AM THE THREHSOLD:", indx, fdr_mean[indx], thresholds[indx])
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
        fig.savefig(os.path.join(self.outdir, self.expt_name, "covs", self.filename + "{}.png".format(epoch)))
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
        self.filename = 'code_entropy_' + name + '_'#+ self.mode

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
        fig.savefig(os.path.join(self.outdir, self.expt_name, "covs", self.filename + "{}.png".format(epoch)))
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
        self.filename = 'dist_' + name + '_'#+ self.mode

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
                [post_mean, post_std, prior_mean, prior_std] = sess.run([post_mean_tf, post_std_tf, prior_mean_tf, prior_std_tf])

                for level in range(nb_prob_scale):

                    nb_units = len(post_mean[level])

                    samples_post = np.random.normal(post_mean[level], post_std[level], (1000, nb_units))
                    samples_prior = np.random.normal(prior_mean[level], prior_std[level], (1000, nb_units))

                    samples_post = np.reshape(samples_post, [-1])
                    samples_prior = np.reshape(samples_prior, [-1])

                    fig = plt.figure()
                    for indx in range(nb_units):
                        print(i, level, indx)
                        plt.subplot(int(np.sqrt(nb_units)), int(np.sqrt(nb_units)), indx+1)
                        plt.hist(samples_post, bins=100, color='red', label='post', alpha=.5)#,  edgecolor='red'
                        plt.hist(samples_prior, bins=100, color='blue', label='prior', alpha=.5)#, edgecolor='blue'
                        plt.axis('off')

                    fig.savefig(
                        os.path.join(self.outdir, self.expt_name, "covs", self.filename + "{}_{}_{}.png".format(level, i, epoch)))
                    plt.close()

