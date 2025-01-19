import os
import warnings

from ignite.distributed import model_name
from monai.utils import set_determinism

from models.model_trainers import model_CLS_CrossTransformer_IT, model_CLS_CNN_Single, \
    model_CLS_Transformer_T, model_CLS_Transformer, model_CLS_Transformer_IT, model_CLS_Tabular_MLP, \
    model_CLS_CrossTransformer_IT_Stage, model_CLS_CrossTransformer_IT_ALLPET, model_CLS_3MT, model_CLS_DAFT, \
    model_CLS_ResNet
from utils import Logger
from options import Option
from datasets import SimpleDataModule
import torch
import random
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    set_determinism(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # initialize options and create output directory
    opt = Option().parse()
    save_dir = opt.expr_dir
    logger_main = Logger(save_dir)

    print('Successfully load datasets.....')

    # prepare kfold splits
    num_fold = 5
    seed = 20230329

    # seed everything
    seed_torch(seed)

    print(f'The random seed is {seed}')
    results = []

    # cross validation loop
    for fold_idx in range(5):
        logger_main.print_message(f'************Fold {fold_idx}************')

        # create data loader according to specific models
        if opt.model == 'CrossTransformer_IT' or opt.model == 'CrossTransformer_IT_Stage' or opt.model == '3MT':
            dm = SimpleDataModule(f'./lookupcsvs/cross{fold_idx}', opt.task, opt.tabular,
                                  need_PET=False, PET_tracer='AV45',
                                  batch_size=opt.batch_size, num_workers=opt.num_workers)
            dm_PET = SimpleDataModule(f'./lookupcsvs/cross{fold_idx}', opt.task, opt.tabular,
                                      need_PET=True, PET_tracer='AV45',
                                      batch_size=opt.batch_size, num_workers=opt.num_workers)
            train_dataloader = dm.train_dataloader()
            val_dataloader = dm.val_dataloader()
            test_dataloader = dm.test_dataloader()
            train_dataloader2 = dm_PET.train_dataloader()
            val_dataloader2 = dm_PET.val_dataloader()
            test_dataloader2 = dm_PET.test_dataloader()
        elif opt.model == 'PET' or opt.model == 'CrossTransformer_IT_ALLPET':
            dm_PET = SimpleDataModule(f'./lookupcsvs/cross{fold_idx}', opt.task, opt.tabular,
                                      need_PET=True, PET_tracer='AV45',
                                      batch_size=opt.batch_size, num_workers=opt.num_workers)
            train_dataloader = dm_PET.train_dataloader()
            val_dataloader = dm_PET.val_dataloader()
            test_dataloader = dm_PET.test_dataloader()
        else:
            dm = SimpleDataModule(f'./lookupcsvs/cross{fold_idx}', opt.task, opt.tabular,
                                  need_PET=False, PET_tracer='AV45',
                                  batch_size=opt.batch_size, num_workers=opt.num_workers)
            train_dataloader = dm.train_dataloader()
            val_dataloader = dm.val_dataloader()
            test_dataloader = dm.test_dataloader()

        # create solvers and start training
        if opt.model == 'tabular':
            solver = model_CLS_Tabular_MLP(opt, fold_idx)
            solver.start_train(train_dataloader, val_dataloader)
            res_fold = solver.start_test(test_dataloader, 'test')
        elif opt.model == 'MRI' or opt.model == 'PET':
            solver = model_CLS_CNN_Single(opt, fold_idx, opt.model)
            solver.start_train(train_dataloader, val_dataloader)
            res_fold = solver.start_test(test_dataloader, 'test')
        elif opt.model == 'Transformer':
            solver = model_CLS_Transformer(opt, fold_idx, opt.model)
            solver.start_train(train_dataloader, val_dataloader)
            res_fold = solver.start_test(test_dataloader, 'test')
        elif opt.model == 'Transformer_T':
            solver = model_CLS_Transformer_T(opt, fold_idx)
            solver.start_train(train_dataloader, val_dataloader)
            res_fold = solver.start_test(test_dataloader, 'test')
        elif opt.model == 'Transformer_IT':
            solver = model_CLS_Transformer_IT(opt, fold_idx)
            solver.start_train(train_dataloader, val_dataloader)
            res_fold = solver.start_test(test_dataloader, 'test')
        elif opt.model == 'CrossTransformer_IT':
            solver = model_CLS_CrossTransformer_IT(opt, fold_idx)
            solver.start_train(train_dataloader, train_dataloader2, val_dataloader)
            res_fold = solver.start_test(test_dataloader, 'test')
        elif opt.model == 'CrossTransformer_IT_Stage':
            solver = model_CLS_CrossTransformer_IT_Stage(opt, fold_idx)
            solver.start_train(train_dataloader, train_dataloader2, val_dataloader)
            res_fold = solver.start_test(test_dataloader, 'test')
        elif opt.model == 'CrossTransformer_IT_ALLPET':
            solver = model_CLS_CrossTransformer_IT_ALLPET(opt, fold_idx)
            solver.start_train(train_dataloader, val_dataloader)
            res_fold = solver.start_test(test_dataloader, 'test')
        
        # comparison methods
        elif opt.model == '3MT':
            solver = model_CLS_3MT(opt, fold_idx)
            solver.start_train(train_dataloader, val_dataloader, 2)
            solver.start_train(train_dataloader2, val_dataloader2, 3)
            res_fold_test_S1, res_fold_test_S2, res_fold_test_S3 = solver.start_test(test_dataloader, "test")
            res_fold = [res_fold_test_S3[0], 0.0, res_fold_test_S3[1], res_fold_test_S3[2], res_fold_test_S3[3], res_fold_test_S3[4]]
        elif opt.model == 'DAFT':
            solver = model_CLS_DAFT(opt, fold_idx)
            solver.start_train(train_dataloader, val_dataloader)
            res_fold = solver.start_test(test_dataloader, 'test')
        elif opt.model == 'HNN':
            solver = model_CLS_DAFT(opt, fold_idx, 'HNN')
            solver.start_train(train_dataloader, val_dataloader)
            res_fold = solver.start_test(test_dataloader, 'test')
        elif opt.model == 'ResNet':
            solver = model_CLS_ResNet(opt, fold_idx, 'MRI')
            solver.start_train(train_dataloader, val_dataloader)
            res_fold = solver.start_test(test_dataloader, 'test')
        else:
            solver = None
            res_fold = None
            
        # logging
        logger_main.print_message(f'Test - ACC:{res_fold[0]:.4f} MCC:{res_fold[1]:.4f} '
                                  f'SEN:{res_fold[2]:.4f} SPE:{res_fold[3]:.4f} '
                                  f'F1:{res_fold[4]:.4f} AUC:{res_fold[-1]:.4f}')
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
                              f'auc: {res_mean[-1]:.4f} +- {res_std[-1]:.4f}\n')