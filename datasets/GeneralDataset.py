from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset
import random
import csv
import nibabel as nib
import os
import torch
import pandas as pd
from monai.transforms import (Compose, ScaleIntensity, CenterSpatialCrop,
                              RandAdjustContrast, RandRotate, RandZoom, RandFlip,
                              RandAdjustContrastd, RandRotated, RandZoomd, RandFlipd)


def path_win2wsl(path):
    path = path.replace('\\', '/')
    path = path.replace('D:', '/mnt/c')
    path = path.replace('Datasets', 'Users/YWZhang/Projects/Datasets')
    return path


class TaskData(Dataset):
    """
    this class will load data for a specific task, thus if the label of the task is missing for a case,
    that case will be omitted by the dataloader
    """

    def __init__(self,
                 data_list,
                 task,  # name of the task (column name)
                 stage,  # stage could be 'train' or 'valid' or 'test'
                 seed=20230329,  # random seed
                 need_PET=False,
                 PET_tracer='AV45'):

        random.seed(seed)
        self.data_list = data_list
        self.task = task
        self.stage = stage
        self.trans = Compose([ScaleIntensity(), CenterSpatialCrop([160, 192, 160])])
        self.trans_aug = Compose([RandAdjustContrast(prob=0.5),
                                  RandRotate(prob=0.2, range_x=0.05),
                                  RandZoom(prob=0.2, min_zoom=0.95)])
        self.trans_aug_d = Compose([RandAdjustContrastd(prob=0.5, keys=['MRI', 'PET']),
                                    RandRotated(prob=0.2, range_x=0.05, keys=['MRI', 'PET']),
                                    RandZoomd(prob=0.2, min_zoom=0.95, max_zoom=1.05, keys=['MRI', 'PET'])])
        self.need_PET = need_PET
        self.PET_tracer = PET_tracer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Read Info From Dataframe
        row = self.data_list.iloc[idx]
        label_title = 'COG' if self.task == 'MCINC' else self.task
        # Read Subject Information
        t1_filepath = path_win2wsl(os.path.join(row['path'], row['filename'].split('_')[0], row['filename'].split('_')[1],
                                   't1', row['filename'] + '.nii.gz'))
        pet_filepath = path_win2wsl(os.path.join(row['path'], row['filename'].split('_')[0], row['filename'].split('_')[1],
                                    self.PET_tracer.lower(), row['filename'] + '.nii.gz'))
        label = int(row[label_title])
        dataset_name = self.map_path_to_dataset(row['path'])

        # Load Tabular Data
        # 37 features
        tabular = torch.from_numpy(row[5:42].values.astype(np.float32))
        tabular_missing = torch.from_numpy(row[42:].values.astype(np.float32))
        # 6 features
        # tabular = torch.from_numpy(row[5:11].values.astype(np.float32))
        # tabular_missing = torch.from_numpy(row[11:].values.astype(np.float32))
        # - demographic information
        # tabular = torch.from_numpy(row[5:37].values.astype(np.float32))
        # tabular_missing = torch.from_numpy(row[37:].values.astype(np.float32))

        # Load Image Data
        t1_data = nib.load(t1_filepath).get_fdata()
        t1_data = torch.from_numpy(np.expand_dims(t1_data, axis=0)).type(torch.float32)
        t1_data = self.trans(t1_data)

        if row[self.PET_tracer] == 1:
            pet_data = nib.load(pet_filepath).get_fdata()
            pet_data = torch.from_numpy(np.expand_dims(pet_data, axis=0)).type(torch.float32)
            pet_data = self.trans(pet_data)
            if self.stage == 'train':
                data = {'MRI': t1_data, 'PET': pet_data}
                if self.task == 'ADD':
                    if label == 0:
                        data = self.trans_aug_d(data)
                else:
                    data = self.trans_aug_d(data)
                t1_data = data['MRI']
                pet_data = data['PET']
            return {'MRI': t1_data, 'PET': pet_data, 'tabular': tabular, 'tabular_missing': tabular_missing,
                    'label': label, 'dataset_name': dataset_name, 'filename': row['filename']}
        else:
            if self.stage == 'train':
                if self.task == 'ADD':
                    if label == 0:
                        t1_data = self.trans_aug(t1_data)
                else:
                    t1_data = self.trans_aug(t1_data)
            return {'MRI': t1_data, 'PET': torch.zeros_like(t1_data), 'tabular': tabular,
                    'tabular_missing': tabular_missing,
                    'label': label, 'dataset_name': dataset_name, 'filename': row['filename']}

    def get_sample_weights(self, ratio={}):
        # ratio = {'PD':0.2} means averagely sample 2 PD cases and 8 no PD cases for each batch
        weights = []
        label_title = 'COG' if self.task == 'MCINC' else self.task
        label_list = self.data_list[label_title].to_list()
        dataset_name_list = [self.map_path_to_dataset(path) for path in self.data_list['path'].to_list()]
        if self.task in ratio:
            for i in label_list:
                if i == 0:
                    weights.append(1 - ratio[self.task])
                elif i == 1:
                    weights.append(ratio[self.task])
            return weights
        # if task is not in ratio, no specific value for the ratio, thus auto-balance the data according to data distribution
        unique = list(set(label_list))
        count = [float(label_list.count(a)) for a in unique]
        total = float(len(label_list))
        for i, name in zip(label_list, dataset_name_list):
            if 'ADNI' in name: name = 'ADNI'
            # factor = self.task_config['sampleWeights'][name]
            factor = 1
            unique_idx = unique.index(i)
            weights.append(total / count[unique_idx] * factor)
        return weights

    def map_path_to_dataset(self, path):
        for candi in ['ADNI', 'NACC', 'FHS', 'AIBL', 'OASIS', 'Stanford', 'PPMI', 'NIFD']:
            if candi in path:
                return candi
        return 'unknown'

# class TabularData(Dataset):
#
#     """
#     this class will load data for a specific task, thus if the label of the task is missing for a case,
#     that case will be omitted by the dataloader
#     """
#
#     def __init__(self,
#                  data_list,
#                  task,            # name of the task (column name)
#                  stage,           # stage could be 'train' or 'valid' or 'test'
#                  seed=20230329,   # random seed
#                  ):
#
#         random.seed(seed)
#         self.data_list = data_list
#         self.task = task
#         self.stage = stage
#
#     def __len__(self):
#         return len(self.data_list)
#
#     def __getitem__(self, idx):
#         # Read Info From Dataframe
#         row = self.data_list.iloc[idx]
#
#         # Read Subject Information
#         label = int(row[self.task])
#         dataset_name = self.map_path_to_dataset(row['path'])
#
#         # Load Tabular Data
#         tabular = torch.from_numpy(row[5:42].values.astype(np.float32))
#         tabular_missing = torch.from_numpy(row[42:].values.astype(np.float32))
#
#         return {'tabular': tabular, 'tabular_missing': tabular_missing, 'label': label, 'dataset_name': dataset_name}
#
#     def get_sample_weights(self, ratio={}):
#         # ratio = {'PD':0.2} means averagely sample 2 PD cases and 8 no PD cases for each batch
#         weights = []
#         label_list = self.data_list[self.task].to_list()
#         dataset_name_list = [self.map_path_to_dataset(path) for path in self.data_list['path'].to_list()]
#         if self.task in ratio:
#             for i in label_list:
#                 if i == 0:
#                     weights.append(1-ratio[self.task])
#                 elif i == 1:
#                     weights.append(ratio[self.task])
#             return weights
#         # if task is not in ratio, no specific value for the ratio, thus auto-balance the data according to data distribution
#         unique = list(set(label_list))
#         count = [float(label_list.count(a)) for a in unique]
#         total = float(len(label_list))
#         for i, name in zip(label_list, dataset_name_list):
#             if 'ADNI' in name: name = 'ADNI'
#             # factor = self.task_config['sampleWeights'][name]
#             factor = 1
#             unique_idx = unique.index(i)
#             weights.append(total / count[unique_idx] * factor)
#         return weights
#
#     def map_path_to_dataset(self, path):
#         for candi in ['ADNI', 'NACC', 'FHS', 'AIBL', 'OASIS', 'Stanford', 'PPMI', 'NIFD']:
#             if candi in path:
#                 return candi
#         return 'unknown'
