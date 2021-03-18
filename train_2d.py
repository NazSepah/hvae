import tensorflow as tf
import os, json, argparse
from os.path import join
import numpy as np
from shutil import copy
from data_source2d import BrainDataProvider as tfDataProvider
from model2d_label_pred import HierarchicalProbUNet

_DECAY_RATE = 1.


def _get_cfg():
    parser = argparse.ArgumentParser(description="Main handler for training",
                                     usage="./script/train.sh -j configs/train.json -g 02")
    parser.add_argument("-j", "--json", help="configuration json file", required=True)
    parser.add_argument('-g', '--gpu', help='Cuda Visible Devices', required=True)
    args = parser.parse_args()
    return args


def _main(args):
    with open(args.json, 'r') as f:
        cfg = json.loads(f.read())

    expt_cfg = cfg['experiment']
    expt_name = expt_cfg['name']
    outdir = expt_cfg['outdir']
    tfdir = expt_cfg['tfdir']
    nb_epochs = expt_cfg['nb_epochs']
    nb_prob_scale = 4
    add_label_loss = True

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(join(outdir, expt_name), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'graphs'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'checkpoints'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'rocs'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'recons'), exist_ok=True)
    copy(args.json, join(outdir, expt_name))
    copy("/cim/nazsepah/projects/hprob-deepmind-2d/model2d_label_pred.py", join(outdir, expt_name))

    gen_train = tfDataProvider(tfdir, cfg['train'])
    gen_valid = tfDataProvider(tfdir, cfg['valid'])

    train_data = gen_train.data_generator()
    valid_data = gen_valid.data_generator()

    nb_samples_train = gen_train.get_nb_samples()
    nb_samples_valid = gen_valid.get_nb_samples()

    print("total number of training samples:", nb_samples_train)
    print("total number of validation samples:", nb_samples_valid)

    nb_batches_train = int(np.ceil(nb_samples_train / cfg['train']['batch_size']))
    nb_batches_valid = int(np.ceil(nb_samples_valid / cfg['valid']['batch_size']))

    iterator = tf.data.Iterator.from_structure(train_data.output_types)
    train_init_op = iterator.make_initializer(train_data)
    valid_init_op = iterator.make_initializer(valid_data)
    next_batch = iterator.get_next()

    sample_weights = tf.placeholder(dtype=tf.float32, shape=(), name='sample_weight')
    hpu_net = HierarchicalProbUNet(name='model/HPUNet')
    loss_train = hpu_net.loss(next_batch, sample_weights, mode=True, bs=cfg['train']['batch_size'])
    loss_valid = hpu_net.loss(next_batch, sample_weights, mode=False, bs=cfg['valid']['batch_size'])

    # set up the optimizer
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate_kwargs = {'values': [1e-4, 0.5e-4, 1e-5, 0.5e-6],
                            'boundaries': [80000., 160000., 240000.],
                            'name': 'piecewise_constant_lr_decay'}
    learning_rate = tf.train.piecewise_constant(x=global_step, **learning_rate_kwargs)
    solver = hpu_net.optimizer(loss_train, learning_rate, global_step)

    loss_all = tf.placeholder(dtype=tf.float32, shape=(), name='loss_all')
    loss_seg = tf.placeholder(dtype=tf.float32, shape=(), name='loss_seg')
    loss_ma = tf.placeholder(dtype=tf.float32, shape=(), name='loss_ma')
    loss_label = tf.placeholder(dtype=tf.float32, shape=(), name='loss_label')
    lambd = tf.placeholder(dtype=tf.float32, shape=(), name='lambda')
    kappa = tf.placeholder(dtype=tf.float32, shape=(), name='kappa')
    loss_kl = []
    for i in range(nb_prob_scale):
        loss_kl.append(tf.placeholder(dtype=tf.float32, shape=(), name='loss_kl-{}'.format(i + 1)))
    loss_recon = []
    for i in range(nb_prob_scale - 1):
        loss_recon.append(tf.placeholder(dtype=tf.float32, shape=(), name='loss_recon-{}'.format(i + 1)))

    tf.summary.scalar('weight_decay', sample_weights)
    tf.summary.scalar('learning_rate', learning_rate)
    train_summary = tf.summary.merge_all()
    lambd_tf = tf.summary.scalar('lambda', lambd)
    loss_all_tf = tf.summary.scalar('loss_all', loss_all)
    loss_seg_tf = tf.summary.scalar('loss_seg', loss_seg)
    loss_ma_tf = tf.summary.scalar('loss_ma', loss_ma)
    loss_label_tf = tf.summary.scalar('loss_label', loss_label)
    kappa_tf = tf.summary.scalar('kappa', kappa)
    loss_kl_tf = []
    for i in range(nb_prob_scale):
        loss_kl_tf.append(tf.summary.scalar('loss_kl_{}'.format(i), loss_kl[i]))
    loss_recon_tf = []
    for i in range(nb_prob_scale - 1):
        loss_recon_tf.append(tf.summary.scalar('loss_recon_{}'.format(i), loss_recon[i]))
    loss_summary = tf.summary.merge([loss_all_tf, loss_seg_tf, loss_label_tf, lambd_tf, loss_ma_tf, kappa_tf] + loss_kl_tf + loss_recon_tf)

    #checkpoint = r'/cim/nazsepah/projects/hprob-deepmind-2d/results/unet-prob-label-hw-400/checkpoints/model.ckpt-1000'
    init_op_global = tf.global_variables_initializer()
    init_op_local = tf.local_variables_initializer()
    saver = tf.train.Saver(max_to_keep=50)

    sess_config = tf.ConfigProto(intra_op_parallelism_threads=8,
                                 inter_op_parallelism_threads=8,
                                 device_count={'CPU': 8})

    with tf.Session(config=sess_config) as sess:
        sess.run(init_op_global)
        sess.run(init_op_local)
        #saver.restore(sess, checkpoint)

        train_writer = tf.summary.FileWriter(join(outdir, expt_name, 'graphs', 'train'), sess.graph)
        valid_writer = tf.summary.FileWriter(join(outdir, expt_name, 'graphs', 'valid'), sess.graph)

        print("training starts!!!")
        for epoch in range(nb_epochs):
            print("iteration {}/{}".format(epoch, nb_epochs), flush=True)
            ##### training, post ###########
            sample_weights_value = 800. * (_DECAY_RATE ** epoch)
            loss_seg_train_np = 0.
            loss_all_train_np = 0.
            loss_label_train_np = 0.
            loss_ma_train_np = 0.
            loss_kl_train_np = [0.] * nb_prob_scale
            loss_recon_train_np = [0.] * (nb_prob_scale - 1)
            sess.run(train_init_op)
            for itr in range(nb_batches_train):
                b, l, s, _ = sess.run([next_batch, loss_train, train_summary] + solver,
                                      feed_dict={sample_weights: sample_weights_value})
                loss_all_train_np += l['supervised_loss']
                loss_seg_train_np += l['summaries']['rec_loss_mean']
                loss_ma_train_np += l['summaries']['rec_constraint']
                loss_label_train_np += l['summaries']['label_loss_mean'] if add_label_loss else 0.
                for i in range(nb_prob_scale):
                    loss_kl_train_np[i] += l['summaries']['kl_{}'.format(i)]
                for i in range(nb_prob_scale - 1):
                    loss_recon_train_np[i] += l['summaries']['recon_{}'.format(i)]

            loss_all_train_np = loss_all_train_np/nb_batches_train
            loss_seg_train_np = loss_seg_train_np / nb_batches_train
            loss_label_train_np = loss_label_train_np / nb_batches_train
            loss_ma_train_np = loss_ma_train_np/ nb_batches_train

            feed_dict_kl = {}
            for i in range(nb_prob_scale):
                feed_dict_kl[loss_kl[i]] = loss_kl_train_np[i]/nb_batches_train

            feed_dict_recon = {}
            for i in range(nb_prob_scale - 1):
                feed_dict_recon[loss_recon[i]] = loss_recon_train_np[i] / nb_batches_train

            summary = sess.run(loss_summary, feed_dict={**feed_dict_kl,
                                                        **feed_dict_recon,
                                                        loss_all: loss_all_train_np,
                                                        loss_seg: loss_seg_train_np,
                                                        loss_ma: loss_ma_train_np,
                                                        loss_label: loss_label_train_np,
                                                        lambd: l['summaries']['lagmul'],
                                                        kappa: l['summaries']['kappa']
                                                        })
            train_writer.add_summary(s, global_step=epoch)
            train_writer.add_summary(summary, global_step=epoch)
            #### save a checkpoint ####
            if epoch % 20 == 0:
                saver.save(sess, join(outdir, expt_name, 'checkpoints', 'model.ckpt-{}'.format(epoch)))

            ##### validation, prior ##########
            sess.run(valid_init_op)
            sample_weights_value = 1.0
            loss_all_valid_np = 0.0
            loss_seg_valid_np = 0.0
            loss_label_valid_np = 0.
            loss_ma_valid_np = 0.
            loss_recon_valid_np = [0.] * (nb_prob_scale - 1)
            loss_kl_valid_np = [0.] * nb_prob_scale
            for itr in range(nb_batches_valid):
                b, l = sess.run([next_batch, loss_valid], feed_dict={sample_weights: sample_weights_value})
                loss_all_valid_np += l['supervised_loss']
                loss_seg_valid_np += l['summaries']['rec_loss_mean']
                loss_ma_valid_np += l['summaries']['rec_constraint']
                loss_label_valid_np += l['summaries']['label_loss_mean'] if add_label_loss else 0.
                for i in range(nb_prob_scale):
                    loss_kl_valid_np[i] += l['summaries']['kl_{}'.format(i)]
                for i in range(nb_prob_scale - 1):
                    loss_recon_valid_np[i] += l['summaries']['recon_{}'.format(i)]

                loss_all_valid_np = loss_all_valid_np / nb_batches_valid
                loss_seg_valid_np = loss_seg_valid_np / nb_batches_valid
                loss_label_valid_np = loss_label_valid_np / nb_batches_valid
                loss_ma_valid_np = loss_ma_valid_np/ nb_batches_valid

            feed_dict_kl = {}
            for i in range(nb_prob_scale):
                feed_dict_kl[loss_kl[i]] = loss_kl_valid_np[i]/nb_batches_valid

            feed_dict_recon = {}
            for i in range(nb_prob_scale - 1):
                feed_dict_recon[loss_recon[i]] = loss_recon_valid_np[i] / nb_batches_valid

            summary = sess.run(loss_summary, feed_dict={**feed_dict_kl,
                                                        **feed_dict_recon,
                                                        loss_all: loss_all_valid_np,
                                                        loss_seg: loss_seg_valid_np,
                                                        loss_ma: loss_ma_valid_np,
                                                        loss_label: loss_label_valid_np,
                                                        lambd: l['summaries']['lagmul'],
                                                        kappa: l['summaries']['kappa']
                                                         })
            valid_writer.add_summary(summary, global_step=epoch)


if __name__ == "__main__":
    _main(_get_cfg())
