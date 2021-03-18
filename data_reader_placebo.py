import numpy as np
from os.path import join, exists, isdir
from os import listdir
from random import shuffle, seed

_MASK_PATH = {'MS-LAQ-302-STX': r'/cim/data/neurorx/MS-LAQ-302-STX/extra/anewt2_75/',
              '101MS326': r'/cim/data/neurorx/lesion_files/'}


class Reader:
    def __init__(self, config, mode, time_points):
        self._data_dir = config.get("image_dir", '/cim/data_raw/preproc/')
        self._dataset_name = config.get("dataset_name", 'MS-LAQ-302-STX')
        self._image_tag = config.get("image_tag", "icbm_N3_VP.mnc")
        self._mask_tag = config.get("mask_tag", "icbm_Beast.mnc")
        self._clinical_file = config.get("clinical_file")
        self._time_points = time_points
        self._image_mods = config.get("image_mods", {'t1p': 0, 'pdw': 1, 'flr': 2, 't2w': 3})
        self._seed = config.get("seed", 1333)
        self._break_tfrecords = config.get("break_tfrecords", False)
        self._mode = mode
        self._nb_image_mods = len(list(self._image_mods.keys()))

    def get_ids(self):
        mri_paths_per_subject = self._get_all_subjects()
        subj_ids = self._get_subject_ids(mri_paths_per_subject)
        subj_ids = self._get_partitions(subj_ids)
        treatment_arm_ids = self._get_treatment_arm_ids(subj_ids)
        mri_paths_per_timepoint = self._get_subjects_tps(mri_paths_per_subject, subj_ids, treatment_arm_ids)
        return mri_paths_per_timepoint

    def _get_all_subjects(self):
        '''
        This function is written for two neurorx trials: Bravo and Ascend.
        It gets the directory information as input and looks inside each subject folder for patients with all required
        modalities available for two consecutive timepoints within the set of interested timepoints.
        :return: a dictionary with said subjects as keys.
        '''

        dataset_name = self._dataset_name
        data_dir = self._data_dir
        ids = {}
        path_info = [[data_dir, site, subj] for site in listdir(data_dir) for subj in
                     listdir(join(data_dir, site)) if '.scannerdb' not in subj]

        nb_subjects = 0
        for data_dir, site, subj in path_info:
            # set the subject_folder
            subject_folder = join(data_dir, site, subj)

            for j in range(len(self._time_points) - 1):

                subject_tp_folder = join(subject_folder, self._time_points[j])
                next_subject_tp_folder = join(subject_folder, self._time_points[j+1])

                if all(isdir(path) for path in [subject_tp_folder, next_subject_tp_folder]):

                    # get all the images file paths
                    pths = ["" for _ in range(2 * self._nb_image_mods + 1)]
                    for i, mod in enumerate(self._image_mods):
                        img_name = '{}_{}_{}_{}_{}_{}'.format(dataset_name, site, subj,
                                                              self._time_points[j],
                                                              mod, self._image_tag)

                        next_img_name = '{}_{}_{}_{}_{}_{}'.format(dataset_name, site, subj,
                                                                   self._time_points[j+1],
                                                                   mod, self._image_tag)
                        pths[2 * i] = join(subject_tp_folder, img_name)
                        pths[2 * i + 1] = join(next_subject_tp_folder, next_img_name)

                    # get the newt2 file paths
                    if dataset_name == 'MS-LAQ-302-STX':
                        newt2_img_name = '{}_{}_{}_{}_anewt2_75_TREF-{}.mnc.gz'.format(dataset_name, site, subj,
                                                                                       self._time_points[j + 1],
                                                                                       self._time_points[j])
                        newt2_path = _MASK_PATH[dataset_name]
                        pths[2 * len(list(self._image_mods.keys()))] = join(newt2_path, newt2_img_name)

                    elif dataset_name == '101MS326':
                        newt2_img_name = '{}_{}_{}_{}_newt2f_TREF-{}_ISPC-stx152lsq6.mnc.gz'.format(dataset_name,
                                                                                                    site, subj,
                                                                                                    self._time_points[j + 1],
                                                                                                    self._time_points[j])
                        newt2_path = join(_MASK_PATH[dataset_name], next_subject_tp_folder)
                        pths[2 * len(list(self._image_mods.keys()))] = join(newt2_path, newt2_img_name)

                    else:
                        raise Exception('unknown dataset')

                    # check if all the paths exist and if they do then add the subject to the dictionary of subjects
                    if np.all([exists(pth) for pth in pths]):
                        subj_dataset = subj
                        if subj_dataset not in ids.keys():
                            ids[subj_dataset] = {}
                            nb_subjects += 1
                        ids[subj_dataset][self._time_points[j]] = {'path': subject_folder,
                                                                   'dataset': dataset_name,
                                                                   'site': site,
                                                                   'image_tag': self._image_tag,
                                                                   'mask_tag': self._mask_tag}
        print("total number of subjects:", nb_subjects)
        return ids

    @staticmethod
    def _get_subject_ids(ids):
        return list(ids.keys())

    def _get_subjects_tps(self, mri_paths, subj_ids, placebo_ids):
        sb_tp = []
        ids_map = [_id.split('_')[-1] for _id in subj_ids]
        for i, subject in enumerate(subj_ids):
            subj = subject.split('__')[-1]
            tps = list(mri_paths[subject].keys())
            subj_clinical_id = ids_map[i]
            if subj_clinical_id in placebo_ids:
                for tp in tps:
                        info = [subj, tp, mri_paths[subject][tp]['path'], mri_paths[subject][tp]['dataset'],
                                mri_paths[subject][tp]['site'], mri_paths[subject][tp]['image_tag'],
                                mri_paths[subject][tp]['mask_tag']]
                        sb_tp.append(info)
        shuffle(sb_tp)
        return sb_tp

    def _get_partitions(self, subj_ids):
        mode = self._mode
        seed_val = self._seed
        seed(seed_val)
        subj_ids_s = sorted(subj_ids)
        shuffle(subj_ids_s)
        nb_subjects = len(subj_ids_s)
        partitions = []
        train_split = int(nb_subjects * 0.6)
        val_split = int(nb_subjects * 0.8)
        if mode == 'train':
            partitions = subj_ids_s[:train_split]
        elif mode == 'valid':
            partitions = subj_ids_s[train_split:val_split]
        elif mode == 'test':
            partitions = subj_ids_s[val_split:]
        return partitions

    def _get_treatment_arm_ids(self, ids, treatment='Placebo'):
        import csv
        id_attr = 'SUBJECT_Screening_Number'
        drug_attr = 'SUBJECT_Trial_Arm'
        clinical_file = self._clinical_file
        placebo_ids = []
        ids_map = [_id.split('_')[-1] for _id in ids]
        csvreader = csv.reader(open(clinical_file, 'r'))
        csvheader = next(csvreader)
        id_indx = csvheader.index(id_attr)
        drug_indx = csvheader.index(drug_attr)
        for i, row in enumerate(csvreader):
            try:
                subj = row[id_indx]
                drug = row[drug_indx]
                if (subj in ids_map) and (drug != treatment):
                    if subj not in list(placebo_ids):
                        placebo_ids.append(subj)
            except:
                continue
        return placebo_ids
