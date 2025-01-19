import os
from collections import OrderedDict

import torch
from torch import nn
import pandas as pd

from models.networks import TabularEmbedding
from models.backbones import MobileNetV2, sNet, SFCN
import ast


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class Logger():
    def __init__(self, log_dir, log_name='log.txt'):
        # create a logging file to store training losses
        self.log_name = os.path.join(log_dir, log_name)
        with open(self.log_name, "a") as log_file:
            log_file.write(f'================ {self.log_name} ================\n')

    def print_message(self, msg, noenter=False):
        if noenter:
            print(msg, end='', flush=True)
        else:
            print(msg, flush=True)
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % msg)

    def print_message_nocli(self, msg):
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % msg)


class CheckpointIO(object):
    def __init__(self, fname_template, **kwargs):
        self.fname_template = fname_template
        self.module_dict = kwargs

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self):
        fname = self.fname_template
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
                outdict[name] = module.state_dict()

        torch.save(outdict, fname)

    def load(self):
        fname = self.fname_template
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device('cpu'))

        for name, module in self.module_dict.items():
            module.load_state_dict(module_dict[name])


def get_pd_gt(csv_file, task):
    csv_file = pd.read_csv(csv_file)
    label = csv_file[task].to_list()
    pred = csv_file[task + '_pred'].to_list()
    if task == 'COG':
        pred_prob = [[row['COG_score_0'], row['COG_score_1'], row['COG_score_2']] for idx, row in csv_file.iterrows()]
    elif task == 'ADD':
        pred_prob = [[row['ADD_score_0'], row['ADD_score_1']] for idx, row in csv_file.iterrows()]
    elif task == 'MCIC':
        pred_prob = [[row['MCIC_score_0'], row['MCIC_score_1']] for idx, row in csv_file.iterrows()]
    else:
        pred_prob = None
    return label, pred, pred_prob


def get_pd_gt_bycohorts(csv_file, task):
    csv_file = pd.read_csv(csv_file)
    label_out, pred_out, pred_prob_out = {}, {}, {}

    if task == 'COG':
        cohorts_list = ['ADNI', 'NACC', 'OAS']
        # cohorts_list = ['Huashan']
        for cohort in cohorts_list:
            csv_cohort = csv_file[csv_file['filename'].str.contains(cohort)]
            label_out[cohort] = csv_cohort[task].to_list()
            pred_out[cohort] = csv_cohort[task + '_pred'].to_list()
            pred_prob_out[cohort] = [[row['COG_score_0'], row['COG_score_1'], row['COG_score_2']]
                                     for idx, row in csv_cohort.iterrows()]
    elif task == 'ADD':
        cohorts_list = ['ADNI', 'NACC', 'OAS']
        for cohort in cohorts_list:
            csv_cohort = csv_file[csv_file['filename'].str.contains(cohort)]
            label_out[cohort] = csv_cohort[task].to_list()
            pred_out[cohort] = csv_cohort[task + '_pred'].to_list()
            pred_prob_out[cohort] = [[row['ADD_score_0'], row['ADD_score_1']]
                                      for idx, row in csv_cohort.iterrows()]
    elif task == 'MCIC':
        cohorts_list = ['ADNI', 'NACC']
        for cohort in cohorts_list:
            csv_cohort = csv_file[csv_file['filename'].str.contains(cohort)]
            label_out[cohort] = csv_cohort[task].to_list()
            pred_out[cohort] = csv_cohort[task + '_pred'].to_list()
            pred_prob_out[cohort]= [[row['MCIC_score_0'], row['MCIC_score_1']]
                                    for idx, row in csv_cohort.iterrows()]
    return label_out, pred_out, pred_prob_out


if __name__ == '__main__':
    test_model = SFCN()
    ckpt_path = './checkpoints/pretrained.p'
    test_model.load_state_dict(torch.load(ckpt_path), strict=False)