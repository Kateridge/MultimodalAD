import os
import warnings

from models.model_trainers import model_CLS_Tabular
from utils import Logger
from options import Option
from datasets import SimpleDataModule
import torch
import random
import numpy as np

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    # initialize options and create output directory
    opt = Option().parse()
    save_dir = opt.expr_dir
    logger_main = Logger(save_dir)

    # prepare kfold splits
    num_fold = 5
    seed = 20230329

    # seed everything
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False

    print(f'The random seed is {seed}')
    results = []

    # cross validation loop
    for fold_idx in range(5):
        logger_main.print_message(f'************Fold {fold_idx}************')
        dm = SimpleDataModule(f'./lookupcsvs/cross{fold_idx}', opt.task, opt.tabular,
                              need_PET=False, PET_tracer='AV45',
                              batch_size=opt.batch_size, num_workers=opt.num_workers)
        if opt.model == 'mlp' or opt.model == 'catboost':
            solver = model_CLS_Tabular(opt, fold_idx)
            solver.start_train(dm.read_task_tabular('train', opt.task))
            solver.start_test(dm.read_task_tabular('test', opt.task), 'test')
            # solver.start_test(dm.read_task_tabular('exter_test', opt.task), 'exter_test')
            res_fold = solver.post_cls_eval('test')
        else:
            solver = None
            res_fold = None
        # logging
        logger_main.print_message(f'Test - ACC:{res_fold[0]:.4f} MCC:{res_fold[1]:.4f} '
                                  f'SEN:{res_fold[2]:.4f} SPE:{res_fold[3]:.4f} '
                                  f'F1:{res_fold[4]:.4f} AUC:{res_fold[5]:.4f}')
        results.append(res_fold)

    results = np.array(results)
    np.save(os.path.join(save_dir, 'results.npy'), results)
    res_mean = np.mean(results, axis=0)
    res_std = np.std(results, axis=0)
    logger_main.print_message(f'************Final Results************')
    logger_main.print_message(f'acc: {res_mean[0]:.4f} +- {res_std[0]:.4f}\n'
                              f'mcc: {res_mean[1]:.4f} +- {res_std[1]:.4f}\n'
                              f'sen: {res_mean[2]:.4f} +- {res_std[2]:.4f}\n'
                              f'spe: {res_mean[3]:.4f} +- {res_std[3]:.4f}\n'
                              f'f1: {res_mean[4]:.4f} +- {res_std[4]:.4f}\n'
                              f'auc: {res_mean[5]:.4f} +- {res_std[5]:.4f}\n')
