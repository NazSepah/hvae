import tensorflow as tf
import os, json, argparse
from os.path import join
import numpy as np
from shutil import copy

from data_source2d import BrainDataProvider as tfDataProvider
from model2d_pred import HierarchicalProbUNet

_DECAY_RATE = .9999


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

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.makedirs(join(outdir, expt_name), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'graphs'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'checkpoints'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'rocs'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'recons'), exist_ok=True)
    copy(args.json, join(outdir, expt_name))
    copy("/cim/nazsepah/projects/hprob-deepmind/model2d_pred.py", join(outdir, expt_name))

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
    #entropy_post, entropy_prior = hpu_net.entropy(next_batch, bs=cfg['train']['batch_size'])
    #active_unit_post, active_unit_prior = hpu_net.active_units(next_batch, bs=cfg['train']['batch_size'])

    # set up the optimizer
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate_kwargs = {'values': [1e-4, 0.5e-4, 1e-5, 0.5e-6],
                            'boundaries': [80000., 160000., 240000.],
                            'name': 'piecewise_constant_lr_decay'}
    learning_rate = tf.train.piecewise_constant(x=global_step, **learning_rate_kwargs)
    solver = hpu_net.optimizer(loss_train, learning_rate, global_step)

    lambd = tf.placeholder(dtype=tf.float32, shape=(), name='lambda')
    loss_all = tf.placeholder(dtype=tf.float32, shape=(), name='loss_all')
    loss_seg = tf.placeholder(dtype=tf.float32, shape=(), name='loss_seg')
    loss_kl = []
    ent_prior = []
    ent_post = []
    act_prior = []
    act_post = []
    for i in range(nb_prob_scale):
        #nb_units = active_unit_post[i].get_shape().as_list()[0]
        loss_kl.append(tf.placeholder(dtype=tf.float32, shape=(), name='loss_kl-{}'.format(i + 1)))
        '''
        ent_prior.append([tf.placeholder(dtype=tf.float32, shape=(), name='unit_entropy_prior-{}'.format(i + 1))
                          for j in range(nb_units)])
        ent_post.append([tf.placeholder(dtype=tf.float32, shape=(), name='unit_entropy_post-{}'.format(i + 1))
                         for j in range(nb_units)])
        act_prior.append([tf.placeholder(dtype=tf.float32, shape=(), name='unit_activity_prior-{}'.format(i + 1))
                          for j in range(nb_units)])
        act_post.append([tf.placeholder(dtype=tf.float32, shape=(), name='unit_activity_post-{}'.format(i + 1))
                         for j in range(nb_units)])
                         '''

    tf.summary.scalar('weight_decay', sample_weights)
    tf.summary.scalar('learning_rate', learning_rate)
    train_summary = tf.summary.merge_all()
    lambd_tf = tf.summary.scalar('lambda', lambd)
    loss_all_tf = tf.summary.scalar('loss_all', loss_all)
    loss_seg_tf = tf.summary.scalar('loss_seg', loss_seg)

    loss_kl_tf = []
    ent_prior_tf = []
    ent_post_tf = []
    au_prior_tf = []
    au_post_tf = []
    for i in range(nb_prob_scale):
        #nb_units = active_unit_post[i].get_shape().as_list()[0]
        loss_kl_tf.append(tf.summary.scalar('loss_kl_{}'.format(i), loss_kl[i]))
        '''
        ent_prior_tf.append([tf.summary.scalar('unit_entropy_prior_{}_{}'.format(i, j), ent_prior[i][j])
                             for j in range(nb_units)])
        ent_post_tf.append([tf.summary.scalar('unit_entropy_post_{}_{}'.format(i, j), ent_post[i][j])
                            for j in range(nb_units)])
        au_prior_tf.append([tf.summary.scalar('unit_activity_prior_{}_{}'.format(i, j), act_prior[i][j])
                            for j in range(nb_units)])
        au_post_tf.append([tf.summary.scalar('unit_activity_post_{}_{}'.format(i, j), act_post[i][j])
                           for j in range(nb_units)])
        '''

    loss_summary = tf.summary.merge([loss_all_tf, loss_seg_tf, lambd_tf] + ent_post_tf + ent_prior_tf +
                                    au_prior_tf + au_post_tf + loss_kl_tf)

    init_op_global = tf.global_variables_initializer()
    init_op_local = tf.local_variables_initializer()
    saver = tf.train.Saver(max_to_keep=50)

    sess_config = tf.ConfigProto(intra_op_parallelism_threads=8,
                                 inter_op_parallelism_threads=8,
                                 device_count={'CPU': 8})

    with tf.Session(config=sess_config) as sess:
        sess.run(init_op_global)
        sess.run(init_op_local)
        sess.run(train_init_op)
        # saver.restore(sess, checkpoint)

        train_writer = tf.summary.FileWriter(join(outdir, expt_name, 'graphs', 'train'), sess.graph)
        valid_writer = tf.summary.FileWriter(join(outdir, expt_name, 'graphs', 'valid'), sess.graph)

        print("training starts!!!")

        for itr in range(nb_epochs * nb_batches_train):
            print("iteration {}/{}".format(itr, nb_epochs * nb_batches_train), flush=True)

            ##### training, post ###########
            if itr % nb_batches_train == 0:
                sess.run(train_init_op)

            sample_weights_value = 800. * (_DECAY_RATE ** itr)

            b, l, _, s = sess.run([next_batch, loss_train, solver, train_summary],
                                                                    feed_dict={sample_weights: sample_weights_value})

            loss_all_train_np = l['supervised_loss']
            loss_seg_train_np = l['summaries']['rec_loss_mean']

            feed_dict_kl = {}
            feed_dict_ent_prior = {}
            feed_dict_ent_post = {}
            feed_dict_act_prior = {}
            feed_dict_act_post = {}
            for i in range(nb_prob_scale):
                feed_dict_kl[loss_kl[i]] = l['summaries']['kl_{}'.format(i)]
                '''
                nb_units = len(a_post[i])
                
                for j in range(nb_units):
                    feed_dict_ent_post[ent_post[i][j]] = e_post[i][j]
                    feed_dict_ent_prior[ent_prior[i][j]] = e_prior[i][j]
                    feed_dict_act_post[act_post[i][j]] = a_post[i][j]
                    feed_dict_act_prior[act_prior[i][j]] = a_prior[i][j]
                    '''

            summary = sess.run(loss_summary, feed_dict={**feed_dict_kl,
                                                        #**feed_dict_ent_post,
                                                        #**feed_dict_ent_prior,
                                                        #**feed_dict_act_post,
                                                        #**feed_dict_act_prior,
                                                        loss_all: loss_all_train_np,
                                                        loss_seg: loss_seg_train_np,
                                                        sample_weights: sample_weights_value,
                                                        lambd: l['summaries']['lagmul']
                                                        })

            if itr % 10 == 0:
                train_writer.add_summary(s, global_step=itr)
                train_writer.add_summary(summary, global_step=itr)


            #### save a checkpoint ####
            if itr % 1000 == 0:
                saver.save(sess, join(outdir, expt_name, 'checkpoints', 'model.ckpt-{}'.format(itr)))

            ##### validation, prior ##########
            if itr % nb_batches_valid == 0:
                sess.run(valid_init_op)

            sample_weights_value = 1.0

            b, l = sess.run([next_batch, loss_valid], feed_dict={sample_weights: sample_weights_value})

            loss_all_valid_np = l['supervised_loss']
            loss_seg_valid_np = l['summaries']['rec_loss_mean']

            feed_dict_kl = {}
            feed_dict_ent_prior = {}
            feed_dict_ent_post = {}
            feed_dict_act_prior = {}
            feed_dict_act_post = {}
            for i in range(nb_prob_scale):
                feed_dict_kl[loss_kl[i]] = l['summaries']['kl_{}'.format(i)]
                '''
                nb_units = len(a_post[i])
                
                for j in range(nb_units):
                    feed_dict_ent_post[ent_post[i][j]] = e_post[i][j]
                    feed_dict_ent_prior[ent_prior[i][j]] = e_prior[i][j]
                    feed_dict_act_post[act_post[i][j]] = a_post[i][j]
                    feed_dict_act_prior[act_prior[i][j]] = a_prior[i][j]
                    '''

            summary = sess.run(loss_summary, feed_dict={**feed_dict_kl,
                                                        #**feed_dict_ent_post,
                                                        #**feed_dict_ent_prior,
                                                        #**feed_dict_act_post,
                                                        #**feed_dict_act_prior,
                                                        loss_all: loss_all_valid_np,
                                                        loss_seg: loss_seg_valid_np,
                                                        sample_weights: sample_weights_value,
                                                        lambd: l['summaries']['lagmul']
                                                        })

            if itr % 10 == 0:
                valid_writer.add_summary(summary, global_step=itr)


if __name__ == "__main__":
    _main(_get_cfg())
