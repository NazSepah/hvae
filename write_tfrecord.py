import tensorflow as tf
from os.path import join
import numpy as np
import nibabel as nib
from data_reader_placebo import Reader
import argparse
import json
import random


_MASK_PATH = {'MS-LAQ-302-STX': r'/cim/data/neurorx/MS-LAQ-302-STX/extra/anewt2_75/',
              '101MS326': r'/cim/data/neurorx/lesion_files/'}


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _get_slices(input_, output_):
    _, _, s = np.where(output_[..., -1] > 0)
    total_lesion_slices = np.unique(s)
    nb_lesion_slices = len(total_lesion_slices)
    nb_healthy_slices = (45 - 15) - nb_lesion_slices
    nb_healthy_slices = min(nb_lesion_slices, nb_healthy_slices)
    nb_healthy_slices = nb_healthy_slices if nb_healthy_slices>1 else 1
    random.shuffle(total_lesion_slices)
    lesion_slices = total_lesion_slices[0:nb_lesion_slices]
    slice_indices = np.array(np.arange(15, 45))
    total_healthy_slices = np.setdiff1d(slice_indices, lesion_slices)
    nb_healthy_slices = np.minimum(len(total_healthy_slices), nb_healthy_slices)
    random.shuffle(total_healthy_slices)
    healthy_slices = total_healthy_slices[0:nb_healthy_slices]
    selected_slices = list(healthy_slices) + list(lesion_slices)
    random.shuffle(selected_slices)
    input_selected = input_[:, :, selected_slices, :]
    output_selected = output_[:, :, selected_slices, :]
    return input_selected, output_selected, nb_healthy_slices, nb_lesion_slices


def _load_data(_id, writer_params, time_points, dtype=np.float32):
    input_mods = writer_params['input_mods']
    output_mods = writer_params['output_mods']
    dim = writer_params['dim']
    bbox = writer_params['bbox']

    nb_mod_input = len(list(input_mods.keys()))
    nb_mod_output = len(list(output_mods.keys()))

    input_imgs = np.empty((nb_mod_input, *dim), dtype=dtype)
    output_imgs = np.empty((nb_mod_output + 1, *dim), dtype=dtype)

    subj, tp, subject_folder, dataset, site, img_tag, mask_tag = _id
    tp_indx = time_points.index(tp)
    next_tp = time_points[tp_indx + 1]
    imgs_folder_path = join(subject_folder, tp)
    next_imgs_folder_path = join(subject_folder, next_tp)
    mask_folder_path = join(subject_folder, 'stx152lsq6')

    mask_name = '{}_{}_{}_{}'.format(dataset, site, subj, mask_tag)
    mask_path = join(mask_folder_path, mask_name)
    mask = _centre(nib.load(mask_path).get_data(), bbox)

    # get all the input modalities by which I mean the image sequences belonging to the first time point
    for mod in input_mods.keys():
        mod_indx = input_mods[mod]
        img_name = '{}_{}_{}_{}_{}_{}'.format(dataset, site, subj, tp, mod, img_tag)
        img_path = join(imgs_folder_path, img_name)
        img = nib.load(img_path).get_data()
        if mod == 'ct2f':
            img = _centre(img, bbox) * mask
        elif mod == 'gvf':
            img = _centre(img, bbox) * mask
            img = (img > 0).astype(int)
        else:
            img = _centre(_normalize(img), bbox) * mask
        input_imgs[mod_indx, ...] = img

    # get all the output modalities by which I mean the image sequences belonging to the second time point
    for mod in output_mods.keys():
        mod_indx = output_mods[mod]
        img_name = '{}_{}_{}_{}_{}_{}'.format(dataset, site, subj, next_tp, mod, img_tag)
        img_path = join(next_imgs_folder_path, img_name)
        img = nib.load(img_path).get_data()
        img = _centre(_normalize(img), bbox) * mask
        output_imgs[mod_indx, ...] = img

    # get NE lesion labels
    if dataset == 'MS-LAQ-302-STX':
        img_name = '{}_{}_{}_{}_anewt2_75_TREF-{}.mnc.gz'.format(dataset, site, subj, next_tp, tp)
        img_path = join(_MASK_PATH[dataset], img_name)
    elif dataset == '101MS326':
        img_name = '{}_{}_{}_{}_newt2f_TREF-{}_ISPC-stx152lsq6.mnc.gz'.format(dataset, site, subj, next_tp, tp)
        img_path = join(_MASK_PATH[dataset], subject_folder, next_tp, img_name)
    img = nib.load(img_path).get_data()
    img = _centre(img, bbox) * mask
    img_newt2 = (img > 0).astype(int)
    output_imgs[nb_mod_output, ...] = img_newt2

    input_imgs = _process(input_imgs)
    output_imgs = _process(output_imgs)

    return input_imgs, output_imgs


def _process(x, clip=True):
    x = x.transpose(2, 3, 1, 0)
    x = x[0:-6, 0:-6, :, :]
    if clip:
        x = np.clip(x, 0., 1.)
    x = np.pad(x, ((0, 0), (0, 0), (6, 7), (0, 0)), 'constant', constant_values=0)
    return x


def _normalize(raw_data):
    if np.sum(raw_data) > 0:
        mask = raw_data > 0
        mu = raw_data[mask].mean()
        sigma = raw_data[mask].std()
        data = (raw_data - mu) / (sigma + 0.0001)
        data = np.clip(data, np.min(data), 3)
        data = (data + (-np.min(data))) / (np.minimum(3, np.max(data)) - np.min(data))
        return data
    else:
        return raw_data


def _centre(img, BBOX):
    l = BBOX['max_r'] - BBOX['min_r']
    w = BBOX['max_c'] - BBOX['min_c']
    s = BBOX['max_s'] - BBOX['min_s']
    d = (l - w) // 2
    img_brain = img[BBOX['min_s']: BBOX['max_s'], BBOX['min_r']: BBOX['max_r'], BBOX['min_c']:BBOX['max_c']]
    img_brain_pad = np.zeros((s, l, l))
    img_brain_pad[:, :, d:w + d] = img_brain
    return img_brain_pad


def main(args):
    # reade the json file containing all the params
    with open(args.json, 'r') as f:
        cfg = json.loads(f.read())

    reader_params = cfg['reader']
    writer_params = cfg['writer']
    time_points = cfg['writer']['time_points']

    # get subjects information from the directory
    reader = Reader(reader_params, writer_params['mode'], time_points)
    ids = reader.get_ids()
    if writer_params['debug']:
        ids = ids[0:4]
    print("total number of subjects:", len(ids))

    # save the ids in a text file
    ids_id = [ids[i][0] + "_" + ids[i][1] for i in range(len(ids))]
    with open(join(writer_params['outdir'], 'bravo_placebo_{}_ids.txt'.format(writer_params['mode'])), 'w') as data_file:
        json.dump(ids_id, data_file, indent=4)

    # get the total number of lesionous and healthy slices
    tot_nb_slices = 0
    tot_nb_healthy_slices = 0
    tot_nb_lesion_slices = 0
    for _id in ids:
        # read the image and pre prcoess it
        input_images, output_images = _load_data(_id, writer_params, time_points)
        # get all the lesionous slices + equal number of healthy slices
        _, _, nb_healthy_slices, nb_lesion_slices = _get_slices(input_images, output_images)
        tot_nb_healthy_slices += nb_healthy_slices
        tot_nb_lesion_slices += nb_lesion_slices
        tot_nb_slices += nb_healthy_slices + nb_lesion_slices
    print("total_nb_slices:", tot_nb_healthy_slices, tot_nb_lesion_slices, tot_nb_slices)

    # get all slices + image ids
    inputs_selected = np.zeros((192, 192, tot_nb_slices, 6))
    outputs_selected = np.zeros((192, 192, tot_nb_slices, 5))
    slices_id = []
    start_indx = 0
    for _id in ids:
        input_images, output_images = _load_data(_id, writer_params, time_points)
        input_selected_slices, output_selected_slices, _, _ = _get_slices(input_images, output_images)
        nb_slice = np.shape(input_selected_slices)[-2]
        inputs_selected[:, :, start_indx:start_indx + nb_slice, :] = input_selected_slices
        outputs_selected[:, :, start_indx:start_indx + nb_slice, :] = output_selected_slices
        slices_id += [_id[0] + "_" + _id[1]] * nb_slice
        start_indx += nb_slice

    # shuffle the images and the ids
    indices = np.arange(tot_nb_slices)
    random.seed(40)
    random.shuffle(indices)
    inputs_selected = inputs_selected[:, :, indices, :]
    slices_id = np.array(slices_id)
    outputs_selected = outputs_selected[:, :, indices, :]
    slices_id = slices_id[indices]

    # split and save the data into 10 separate tfrecord files
    nb_folds = 10
    fold_size = tot_nb_slices // nb_folds
    folds_ids = []
    for i in range(nb_folds):
        folds_ids.append(i * fold_size)
    folds_ids.append(tot_nb_slices)

    for i in range(nb_folds - 1):
        writer = tf.python_io.TFRecordWriter(
            join(writer_params['outdir'], 'bravo_placebo_' + str(writer_params['mode']) + '_'+ str(i)+'.tfrecords'))

        start_indx = folds_ids[i]
        end_indx = folds_ids[i + 1]
        for slice_indx in range(start_indx, end_indx):
            input_selected = inputs_selected[:, :, slice_indx, :]
            output_selected = outputs_selected[:, :, slice_indx, :]
            # create features
            feature = {'tp1': _bytes_feature(tf.compat.as_bytes(input_selected.tostring())),
                       'tp2': _bytes_feature(tf.compat.as_bytes(output_selected.tostring())),
                       'id': _bytes_feature(tf.compat.as_bytes(slices_id.tostring()))
                       }
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()


def _get_cfg():
    parser = argparse.ArgumentParser(description="handler for writing the tfrecord flies",
                                     usage="python write_tfrecord.py -j configs/tfrecord.json")
    parser.add_argument("-j", "--json", help="configuration json file", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(_get_cfg())






