import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from datasets.GeneralDataset import TaskData


class SimpleDataModule():
    def __init__(self,
                 csv_dir,
                 task,
                 feature_list,
                 need_PET: bool = False,
                 PET_tracer: str = 'AV45',
                 batch_size: int = 1,
                 num_workers: int = 0,
                 seed: int = 20230329,
                 pin_memory: bool = False,
                 ):
        self.csv_dir = csv_dir

        # tabular features
        self.feature_list = feature_list
        self.imputer = self.init_imputer()

        # dataset info
        self.task = task
        self.need_PET = need_PET
        self.PET_tracer = PET_tracer

        # training configures
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory

        # create dataset
        self.train_mean = {}
        self.train_std = {}
        self.ds_train = TaskData(self.read_task_csv('train', task, need_PET=need_PET, PET_tracer=PET_tracer),
                                 task=task, stage='train', seed=seed, need_PET=need_PET, PET_tracer=PET_tracer)
        self.ds_val = TaskData(self.read_task_csv('valid', task, need_PET=need_PET, PET_tracer=PET_tracer),
                               task=task, stage='valid', seed=seed, need_PET=need_PET, PET_tracer=PET_tracer)
        self.ds_test = TaskData(self.read_task_csv('test', task, need_PET=need_PET, PET_tracer=PET_tracer),
                                task=task, stage='test', seed=seed, need_PET=need_PET, PET_tracer=PET_tracer)
        self.ds_externaltest = TaskData(self.read_task_csv('exter_test', task, need_PET=need_PET, PET_tracer=PET_tracer),
                                task=task, stage='exter_test', seed=seed, need_PET=need_PET, PET_tracer=PET_tracer)
        self.ds_background = TaskData(self.read_task_csv(f'{task}_shap_background', task, need_PET=False, PET_tracer=PET_tracer),
                                task=task, stage='exter_test', seed=seed, need_PET=False, PET_tracer=PET_tracer)

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        sampler = WeightedRandomSampler(self.ds_train.get_sample_weights(), len(self.ds_train.get_sample_weights()),
                                        generator=generator)
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=sampler, generator=generator, drop_last=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_val is not None:
            return DataLoader(self.ds_val, batch_size=1, num_workers=self.num_workers, shuffle=False,
                              generator=generator, drop_last=False, pin_memory=self.pin_memory)
        else:
            raise AssertionError("A validation set was not initialized.")

    def test_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_test is not None:
            return DataLoader(self.ds_test, batch_size=1, num_workers=self.num_workers, shuffle=False,
                              generator=generator, drop_last=False, pin_memory=self.pin_memory)
        else:
            raise AssertionError("A test test set was not initialized.")

    def extertest_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_test is not None:
            return DataLoader(self.ds_externaltest, batch_size=1, num_workers=self.num_workers, shuffle=False,
                              generator=generator, drop_last=False, pin_memory=self.pin_memory)
        else:
            raise AssertionError("A external test test set was not initialized.")

    def background_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_test is not None:
            return DataLoader(self.ds_background, batch_size=1, num_workers=self.num_workers, shuffle=False,
                              generator=generator, drop_last=False, pin_memory=self.pin_memory)
        else:
            raise AssertionError("A external test test set was not initialized.")

    def init_imputer(self):
        """
        since cases with ADD labels is only a subset of the cases with COG label
        in this function, we will initialize a single imputer
        and fit the imputer based on the COG cases from the training part
        """
        data = pd.read_csv(os.path.join(self.csv_dir, 'train.csv'))
        data = data[self.feature_list]
        data = data.replace({'male': 0, 'female': 1})

        imputation_method = 'KNN'
        if imputation_method == 'mean':
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif imputation_method == 'median':
            imp = SimpleImputer(missing_values=np.nan, strategy='median')
        elif imputation_method == 'most_frequent':
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        elif imputation_method == 'constant':
            imp = SimpleImputer(missing_values=np.nan, strategy='constant')
        elif imputation_method == 'KNN':
            imp = KNNImputer(n_neighbors=20)
        else:
            raise NameError('method for imputation not supported')

        imp.fit(data)
        return imp

    def read_task_csv(self, stage, task, need_PET=False, PET_tracer='AV45'):
        # read csv file and features
        data = pd.read_csv(os.path.join(self.csv_dir, f'{stage}.csv'))
        label_title = 'COG' if task == 'MCINC' else task
        data = data[['path', 'filename', 'FDG', 'AV45'] + [label_title] + self.feature_list]

        # drop subjects does not belong to this task
        data = data.dropna(axis=0, how='any', subset=[label_title], inplace=False)
        # drop some subjects with error
        data = data.drop(data[data['filename'] == 'sub-ADNI013S4616_ses-M00'].index)
        data = data.drop(data[data['filename'] == 'sub-ADNI067S5205_ses-M00'].index)
        if task == 'MCINC':
            data = data.drop(data[data['COG'] == 2].index)
        if need_PET:
            # drop subjects with no PET scans if need_PET
            data = data.drop(index=data[data[PET_tracer].isin([0])].index)
        if stage == 'exter_test':
            # drop NIFD and AIBL subjects
            data = data.drop(index=data[data['filename'].str.contains('NIFD')].index)
            data = data.drop(index=data[data['filename'].str.contains('AIBL')].index)
        data = data.reset_index(drop=True)

        # Pre-process the clinical information
        # 1) Transform categorical information
        data = data.replace({'male': 0.0, 'female': 1.0})
        # 2) Generate missing mask
        features_missing = data.copy()[self.feature_list]
        features_missing = features_missing.where(features_missing.isnull(), 1.0)
        features_missing = features_missing.fillna(0.0)
        # 3) Normalize imputed features
        features = data.copy()[self.feature_list]
        # if stage == 'train':
        #     for column in features.columns:
        #         features[column] = (features[column] - features[column].mean()) / features[column].std()
        #         self.train_mean[column] = features[column].mean()
        #         self.train_std[column] = features[column].std()
        # else:
        #     for column in features.columns:
        #         features[column] = (features[column] - self.train_mean[column]) / self.train_std[column]
        for column in features.columns:
            features[column] = (features[column] - features[column].mean()) / features[column].std()
        features = features.fillna(0.0)

        # zip all information and create dataset
        final_data = data[['path', 'filename', 'FDG', 'AV45'] + [label_title]]
        final_data = final_data.join(features, how='inner')
        final_data = final_data.join(features_missing, rsuffix='_missing', how='inner')

        return final_data
    

    def read_task_tabular(self, stage, task):
        # read csv file and features
        data = pd.read_csv(os.path.join(self.csv_dir, f'{stage}.csv'))
        data = data[['filename', task] + self.feature_list]

        # drop subjects does not belong to this task
        data = data.dropna(axis=0, how='any', subset=[task], inplace=False)
        if stage == 'exter_test':
            # drop NIFD and AIBL subjects
            data = data.drop(index=data[data['filename'].str.contains('NIFD')].index)
            data = data.drop(index=data[data['filename'].str.contains('AIBL')].index)
        data = data.reset_index(drop=True)

        # Pre-process the clinical information
        # 1) Transform categorical information
        data = data.replace({'male': 0.0, 'female': 1.0})
        # 2) Generate missing mask
        features_missing = data.copy()[self.feature_list]
        features_missing = features_missing.where(features_missing.isnull(), 1.0)
        features_missing = features_missing.fillna(0.0)
        # 3) Impute missing features
        features = data.copy()[self.feature_list]
        features = self.imputer.transform(features)
        features = pd.DataFrame(features, columns=self.feature_list)
        # 4) Normalize imputed features
        for column in features.columns:
            if features[column].std():  # normalize only when std != 0
                features[column] = (features[column] - features[column].mean()) / features[column].std()

        # zip all information and create dataset
        final_data = data[['filename'] + [task]]
        final_data = final_data.join(features, how='inner')
        # final_data = final_data.join(features_missing, rsuffix='_missing', how='inner')
        return final_data


if __name__ == '__main__':
    dm = SimpleDataModule('C:\\Users\\Lalal\\Projects\\ncomms2022\\lookupcsv\\CrossValid\\cross0', 'COG')
    tabular_data = dm.read_task_tabular('train', 'COG')
    print(tabular_data['COG'])