import tensorflow as tf
import os, json, argparse
from os.path import join
import numpy as np
from shutil import copy

from data_source2d import BrainDataProvider as tfDataProvider
from callback import VisualizeSamples, PlotRocSeg, ActvieUnits, ZEntropy, Dist, ReplaceZ, GEDImages, GEDSamples, \
    PlotRoc, PlotRocProb, PlotRocSegProb, PlotRocLevels, VisualizeSamplesLevels


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

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.makedirs(join(outdir, expt_name), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'graphs'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'checkpoints'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'rocs'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'recons'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'covs'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'dists'), exist_ok=True)

    cfg['callback']['data_augment'] = False
    gen_callback_valid = tfDataProvider(tfdir, cfg['callback'])
    callback_valid_data = gen_callback_valid.data_generator()
    nb_samples_callback_valid = gen_callback_valid.get_nb_samples()

    print("total number of callback samples:", nb_samples_callback_valid)

    code_entropy_callback = ZEntropy(cfg, reuse=True)
    code_entropy_callback.on_epoch_end(callback_data=callback_valid_data,
                                       nb_imgs=5,
                                       mode_value='valid',
                                       epoch=1000)

    dist_callback = Dist(cfg, reuse=True)
    dist_callback.on_epoch_end(callback_data=callback_valid_data,
                               nb_imgs=5,
                               mode_value='valid',
                               epoch=1000)

    roc_callback = PlotRocLevels(cfg, reuse=True)
    roc_callback.on_epoch_end(callback_data=callback_valid_data,
                              nb_imgs=nb_samples_callback_valid,
                              mode_value='valid',
                              epoch=1500)

    recon_callback = VisualizeSamplesLevels(cfg, name='valid', reuse=True)
    recon_callback.on_epoch_end(callback_data=callback_valid_data,
                                nb_imgs=20,
                                nb_samples=10,
                                level=3,
                                epoch=1000)

    recon_callback = VisualizeSamples(cfg, name='valid', reuse=True)
    recon_callback.on_epoch_end(callback_data=callback_valid_data,
                                nb_imgs=5,
                                mode_value='valid',
                                nb_samples=10,
                                epoch=1000)

    unit_activity_callback = ActvieUnits(cfg, reuse=True)
    unit_activity_callback.on_epoch_end(callback_data=callback_valid_data,
                                        nb_imgs=1,
                                        mode_value='valid',
                                        epoch=500)

    roc_callback = PlotRoc(cfg, name='valid', reuse=True)
    roc_callback.on_epoch_end(callback_data=callback_valid_data,
                              nb_imgs=nb_samples_callback_valid,
                              mode_value='train',
                              epoch=20)

    ged_callback = GEDImages(cfg, name='valid', reuse=True)
    ged_callback.on_epoch_end(callback_data=callback_valid_data,
                              nb_imgs=nb_samples_callback_valid,
                              mode_value='valid',
                              nb_samples=10,
                              epoch=1000)

    roc_callback = ReplaceZ(cfg, reuse=True)
    roc_callback.on_epoch_end(callback_data=callback_valid_data,
                              nb_imgs=nb_samples_callback_valid,
                              mode_value='valid',
                              epoch=400)

    roc_callback = PlotRocSeg(cfg, name='valid', reuse=True)
    roc_callback.on_epoch_end(callback_data=callback_valid_data,
                              nb_imgs=nb_samples_callback_valid,
                              mode_value='train',
                              epoch=200)

    roc_callback = PlotRocSegProb(cfg, name='valid', reuse=True)
    roc_callback.on_epoch_end(callback_data=callback_valid_data,
                              nb_imgs=nb_samples_callback_valid,
                              mode_value='valid',
                              nb_samples=100,
                              epoch=1000)

    ged_callback = GEDSamples(cfg, name='valid', reuse=True)
    ged_callback.on_epoch_end(callback_data=callback_valid_data,
                              nb_imgs=nb_samples_callback_valid,
                              mode_value='valid',
                              nb_samples=16,
                              epoch=800)

    roc_callback = PlotRocProb(cfg, name='valid', reuse=True)
    roc_callback.on_epoch_end(callback_data=callback_valid_data,
                                nb_imgs=nb_samples_callback_valid,
                                mode_value='valid',
                                nb_samples=100,
                                epoch=600)


if __name__ == "__main__":
    _main(_get_cfg())
