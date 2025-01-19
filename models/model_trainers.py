import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.nn import AdaptiveAvgPool1d, AdaptiveMaxPool1d

import utils
import datetime
from monai.data import decollate_batch
from monai.metrics import CumulativeAverage, ROCAUCMetric
from monai.transforms import Compose, Activations, AsDiscrete
from munch import Munch
from models import build_models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pycm import ConfusionMatrix
import numpy as np
import csv
import shap


class model_CLS_Tabular(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, args.model)
        if args.task == 'ADCN' or args.task == 'pMCIsMCI':
            class_num = 2
        else:
            class_num = 3
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.expr_dir, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)

    def start_train(self, data):
        args = self.args
        self.logger.print_message('Start training...')
        X, Y = data.drop(['filename', args.task], axis=1), data[args.task]
        self.nets.fit(X, Y)

    def start_test(self, data, stage):
        task = self.args.task
        features = data.drop(['filename', task], axis=1)
        labels = data[task]
        filenames = data['filename']

        # make sure the features and labels has the same number of rows
        if len(features.index) != len(labels.index):
            raise ValueError('number of rows between features and labels have to be the same')

        predicts = self.nets.predict_proba(features)
        content = []
        for i in range(len(features.index)):
            # the feature and label are for the i th subject
            label = labels.iloc[i]
            filename = filenames.iloc[i]
            case = {'filename': filename}
            case[task] = int(label)
            if task == 'COG':
                case['COG_score_0'] = f'{predicts[i][0]:.8f}'
                case['COG_score_1'] = f'{predicts[i][1]:.8f}'
                case['COG_score_2'] = f'{predicts[i][2]:.8f}'
                case['COG_pred'] = predicts[i].argmax()
            elif task == 'ADD':
                case['ADD_score_0'] = f'{predicts[i][0]:.8f}'
                case['ADD_score_1'] = f'{predicts[i][1]:.8f}'
                case['ADD_pred'] = predicts[i].argmax()
            elif task == 'MCIC':
                case['MCIC_score_0'] = f'{predicts[i][0]:.8f}'
                case['MCIC_score_1'] = f'{predicts[i][1]:.8f}'
                case['MCIC_pred'] = predicts[i].argmax()
            content.append(case)

        with open(os.path.join(self.checkpoint_dir, stage + '_eval.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
            writer.writeheader()
            for case in content:
                writer.writerow(case)

    def post_cls_eval(self, stage):
        csv_file = os.path.join(self.checkpoint_dir, stage + '_eval.csv')
        label, pred, pred_prob = utils.get_pd_gt(csv_file, self.args.task)
        cm = ConfusionMatrix(actual_vector=label, predict_vector=pred)
        if self.args.task == 'COG':
            acc = cm.Overall_ACC
            mcc = cm.Overall_MCC
            sen = cm.TPR_Macro
            spe = cm.TNR_Macro
            f1 = cm.F1_Macro
            auc = cm.AUC
            return [acc, mcc, sen, spe, f1, np.mean(list(auc.values()))]
        else:
            acc = cm.Overall_ACC
            mcc = list(cm.MCC.values())[1]
            sen = list(cm.TPR.values())[1]
            spe = list(cm.TNR.values())[1]
            f1 = list(cm.F1.values())[1]
            auc = list(cm.AUC.values())[1]
            return [acc, mcc, sen, spe, f1, auc]


class model_CLS_Tabular_MLP(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'tabular')
        if args.task == 'MCIC' or args.task == 'ADD':
            class_num = 2
        else:
            class_num = 3
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.expr_dir, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        self.optims = Munch()
        for net in self.nets.keys():
            self.optims[net] = torch.optim.AdamW(
                params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=args.weight_decay)
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_label = Compose([AsDiscrete(to_onehot=class_num)])
        # define evaluation metrics
        self.loss_epoch = CumulativeAverage()
        self.AUC = ROCAUCMetric(average='macro')

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def _reset_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = os.path.join(checkpoint_dir, str(self.fold))
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]

    def compute_CLS_loss(self, tabular, tabular_missing, label):
        # tabular embedding
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        tabular_feats = torch.flatten(tabular_feats, start_dim=1, end_dim=-1)
        # forward MLP
        logits = self.nets.CLS(tabular_feats)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def start_train(self, train_loaders, val_loader):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_epoch = 1
        self.logger.print_message('Start training...')
        start_time = time.time()

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            # initialize or reset training predictions
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                DX = data['label'].to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(tabular, tabular_missing, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.T_Embedding.step()
                self.optims.CLS.step()
                # compute training metrics
                self.loss_epoch.append(loss_CLS)
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                label_real += [i for i in decollate_batch(data['label'])]
                label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log training status
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')

            # log training metrics
            loss_results = self.loss_epoch.aggregate()
            cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_train.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_train.TNR_Macro):.4f} '
                                          f'F1:{float(cm_train.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_train.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_train.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_train.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            # validation iterations
            for name, model in self.nets.items():
                model.eval()

            # reset metrics
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    tabular = data['tabular'].to(self.device)
                    tabular_missing = data['tabular_missing'].to(self.device)
                    DX = data['label'].to(self.device)
                    logits, loss_CLS = self.compute_CLS_loss(tabular, tabular_missing, DX)
                    # compute training metrics
                    self.loss_epoch.append(loss_CLS)
                    self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                             y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    label_real += [i for i in decollate_batch(data['label'])]
                    label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log validation metrics
            loss_results = self.loss_epoch.aggregate()
            cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_val.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_val.TNR_Macro):.4f} '
                                          f'F1:{float(cm_val.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_val.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_val.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_val.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # save best model according to the validation results
            acc = float(cm_val.Overall_ACC)
            f1 = float(cm_val.F1_Macro)
            if acc >= best_acc and f1 >= best_f1:
                self._save_checkpoint()
                best_acc = acc
                best_f1 = f1
                best_epoch = epoch + 1

    def start_test(self, test_loader, phase):
        args = self.args
        # load best model
        self._load_checkpoint()
        content = []
        # reset training predictions
        self.AUC.reset()
        # test loop
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                DX = data['label'].to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(tabular, tabular_missing, DX)
                pred_prob = [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits)]
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                # write pred information
                label = data['label'][0].numpy()
                filename = data['filename'][0]
                case = {'filename': filename}
                case[args.task] = int(label)
                if args.task == 'COG':
                    case['COG_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['COG_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['COG_score_2'] = f'{pred_prob[0][2]:.8f}'
                    case['COG_pred'] = pred_prob[0].argmax()
                elif args.task == 'ADD':
                    case['ADD_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['ADD_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['ADD_pred'] = pred_prob[0].argmax()
                elif args.task == 'MCIC':
                    case['MCIC_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['MCIC_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['MCIC_pred'] = pred_prob[0].argmax()
                content.append(case)

        # write to csv
        with open(os.path.join(self.checkpoint_dir, phase + '_eval.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
            writer.writeheader()
            for case in content:
                writer.writerow(case)

        # log validation metrics
        test_res_all = self.post_cls_eval(phase)
        self.logger.print_message(f'ACC:{test_res_all[0]:.4f} '
                                  f'MCC:{test_res_all[1]:.4f} '
                                  f'SEN:{test_res_all[2]:.4f} '
                                  f'SPE:{test_res_all[3]:.4f} '
                                  f'F1:{test_res_all[4]:.4f} '
                                  f'AUC:{self.AUC.aggregate():.4f}')

        return test_res_all + [self.AUC.aggregate()]

    def post_cls_eval(self, stage):
        csv_file = os.path.join(self.checkpoint_dir, stage + '_eval.csv')
        label, pred, pred_prob = utils.get_pd_gt(csv_file, self.args.task)
        cm = ConfusionMatrix(actual_vector=label, predict_vector=pred)
        if self.args == 'COG':
            acc = cm.Overall_ACC
            mcc = cm.Overall_MCC
            sen = cm.TPR_Macro
            spe = cm.TNR_Macro
            f1 = cm.F1_Macro
            auc = cm.AUC
            return [acc, mcc, sen, spe, f1, np.mean(list(auc.values()))]
        else:
            acc = cm.Overall_ACC
            mcc = list(cm.MCC.values())[1]
            sen = list(cm.TPR.values())[1]
            spe = list(cm.TNR.values())[1]
            f1 = list(cm.F1.values())[1]
            auc = list(cm.AUC.values())[1]
            return [acc, mcc, sen, spe, f1, auc]


class model_CLS_CNN_Single(nn.Module):
    def __init__(self, args, fold, modality):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'CNN_Single')
        if args.task == 'MCIC' or args.task == 'ADD' or args.task == 'MCINC':
            class_num = 2
        else:
            class_num = 3
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.expr_dir, str(fold))
        self.modality = modality
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        self.optims = Munch()
        for net in self.nets.keys():
            self.optims[net] = torch.optim.AdamW(
                params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=args.weight_decay)
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_label = Compose([AsDiscrete(to_onehot=class_num)])
        # define evaluation metrics
        self.loss_epoch = CumulativeAverage()
        self.AUC = ROCAUCMetric(average='macro')

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    # def _save_checkpoint(self, trial_number=None):
    #     param_dict = {}
    #     for name, network in self.named_children():
    #         param_dict[name] = network.state_dict()
    #     if trial_number:
    #         torch.save(param_dict, os.path.join(self.checkpoint_dir, f'best_nets_{trial_number}.pt'))
    #     else:
    #         torch.save(param_dict, os.path.join(self.checkpoint_dir, 'best_nets.pt'))
    #
    # def _load_checkpoint(self, trial_number=None):
    #     if trial_number:
    #         param_dict = torch.load(os.path.join(self.checkpoint_dir, f'best_nets_{trial_number}.pt'))
    #     else:
    #         param_dict = torch.load(os.path.join(self.checkpoint_dir, 'best_nets.pt'))
    #     for name, params in param_dict.items():
    #         self.nets[name].load_state_dict(params)

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def load_checkpoint(self):
        self._load_checkpoint()

    def _reset_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = os.path.join(checkpoint_dir, str(self.fold))
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]

    def compute_CLS_loss(self, IMG, label):
        # forward CNN
        feats = self.nets.CNN(IMG)
        # Average Pool
        feats = F.adaptive_avg_pool3d(feats, 1).view(IMG.shape[0], -1)
        # forward MLP
        logits = self.nets.CLS(feats)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def start_train(self, train_loaders, val_loader, optuna=None):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_auc = 0.0
        best_epoch = 1
        self.logger.print_message('Start training...')
        start_time = time.time()

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            # initialize or reset training predictions
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                if self.modality == 'MRI':
                    IMG = data['MRI'].to(self.device)
                    DX = data['label'].to(self.device)
                elif self.modality == 'PET':
                    IMG = data['PET'].to(self.device)
                    DX = data['label'].to(self.device)
                else:
                    IMG, DX = None, None
                logits, loss_CLS = self.compute_CLS_loss(IMG, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.CNN.step()
                self.optims.CLS.step()
                # compute training metrics
                self.loss_epoch.append(loss_CLS)
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                label_real += [i for i in decollate_batch(data['label'])]
                label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log training status
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')

            # log training metrics
            loss_results = self.loss_epoch.aggregate()
            cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_train.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_train.TNR_Macro):.4f} '
                                          f'F1:{float(cm_train.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f"Trainng    - Loss:{float(loss_results):.4f} "
                                          f"ACC:{float(cm_train.Overall_ACC):.4f} "
                                          f"SEN:{float(list(cm_train.TPR.values())[1]):.4f} "
                                          f"SPE:{float(list(cm_train.TNR.values())[1]):.4f} "
                                          f"F1:{float(list(cm_train.F1.values())[1]):.4f} "
                                          f"AUC:{self.AUC.aggregate():.4f}")

            # validation iterations
            for name, model in self.nets.items():
                model.eval()

            # reset metrics
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    if self.modality == 'MRI':
                        IMG = data['MRI'].to(self.device)
                        DX = data['label'].to(self.device)
                    elif self.modality == 'PET':
                        IMG = data['PET'].to(self.device)
                        DX = data['label'].to(self.device)
                    else:
                        IMG, DX = None, None
                    logits, loss_CLS = self.compute_CLS_loss(IMG, DX)
                    # compute training metrics
                    self.loss_epoch.append(loss_CLS)
                    self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                             y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    label_real += [i for i in decollate_batch(data['label'])]
                    label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log validation metrics
            loss_results = self.loss_epoch.aggregate()
            cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_val.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_val.TNR_Macro):.4f} '
                                          f'F1:{float(cm_val.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_val.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_val.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_val.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # save best model according to the validation results
            if self.args.task == 'ADD':
                auc = self.AUC.aggregate()
                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch + 1
                    self._save_checkpoint()
            else:
                acc = float(cm_val.Overall_ACC)
                f1 = float(cm_val.F1_Macro)
                if acc > best_acc and f1 >= best_f1:
                    best_acc = acc
                    best_f1 = f1
                    best_epoch = epoch + 1
                    if optuna:
                        optuna.report(f1, epoch)
                        self._save_checkpoint(optuna.number)
                    else:
                        self._save_checkpoint()

        return best_f1

    def start_test(self, test_loader, phase, trial=None):
        args = self.args
        # start test iterations
        if trial:
            self._load_checkpoint(trial_number=trial)
        else:
            self._load_checkpoint()
        content = []
        # reset training predictions
        self.AUC.reset()
        # test loop
        for name, model in self.nets.items():
            model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                if self.modality == 'MRI':
                    IMG = data['MRI'].to(self.device)
                    DX = data['label'].to(self.device)
                elif self.modality == 'PET':
                    IMG = data['PET'].to(self.device)
                    DX = data['label'].to(self.device)
                else:
                    IMG, DX = None, None
                logits, loss_CLS = self.compute_CLS_loss(IMG, DX)
                pred_prob = [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits)]
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                # write pred information
                label = data['label'][0].numpy()
                filename = data['filename'][0]
                case = {'filename': filename}
                case[args.task] = int(label)
                if args.task == 'COG':
                    case['COG_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['COG_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['COG_score_2'] = f'{pred_prob[0][2]:.8f}'
                    case['COG_pred'] = pred_prob[0].argmax()
                elif args.task == 'ADD':
                    case['ADD_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['ADD_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['ADD_pred'] = pred_prob[0].argmax()
                elif args.task == 'MCIC':
                    case['MCIC_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['MCIC_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['MCIC_pred'] = pred_prob[0].argmax()
                content.append(case)

        with open(os.path.join(self.checkpoint_dir, phase + '_eval.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
            writer.writeheader()
            for case in content:
                writer.writerow(case)

        # log validation metrics
        test_res_all = self.post_cls_eval(phase)
        self.logger.print_message(f'ACC:{test_res_all[0]:.4f} '
                                  f'MCC:{test_res_all[1]:.4f} '
                                  f'SEN:{test_res_all[2]:.4f} '
                                  f'SPE:{test_res_all[3]:.4f} '
                                  f'F1:{test_res_all[4]:.4f} '
                                  f'AUC:{self.AUC.aggregate():.4f}')
        return test_res_all + [self.AUC.aggregate()]

    def post_cls_eval(self, stage):
        csv_file = os.path.join(self.checkpoint_dir, stage + '_eval.csv')
        label, pred, pred_prob = utils.get_pd_gt(csv_file, self.args.task)
        cm = ConfusionMatrix(actual_vector=label, predict_vector=pred)
        if self.args.task == 'COG':
            acc = cm.Overall_ACC
            mcc = cm.Overall_MCC
            sen = cm.TPR_Macro
            spe = cm.TNR_Macro
            f1 = cm.F1_Macro
            auc = cm.AUC
            return [acc, mcc, sen, spe, f1, np.mean(list(auc.values()))]
        else:
            acc = cm.Overall_ACC
            mcc = list(cm.MCC.values())[1]
            sen = list(cm.TPR.values())[1]
            spe = list(cm.TNR.values())[1]
            f1 = list(cm.F1.values())[1]
            auc = list(cm.AUC.values())[1]
            return [acc, mcc, sen, spe, f1, auc]

    def shap_mid(self, background_dataloader, val_dataloader):
        shap_dir = os.path.join(self.checkpoint_dir,'shap')
        if not os.path.exists(shap_dir): # create the folder for storing shap heatmaps
            os.mkdir(shap_dir)
        self._load_checkpoint()
        for name, model in self.nets.items():
            model.eval()

        # initialize shap explainer
        background = []
        for i, data in enumerate(background_dataloader):
            background.append(data['MRI'])
        background = torch.cat(background, 0).to(self.device)
        e = shap.DeepExplainer(self.CNN, background)
        del background
        # loop over val_dataloder
        for i, data in enumerate(val_dataloader):
            MRI = data['MRI'].to(self.device)
            name = data['filename'][0]
            shap_value = e.shap_values(MRI)
            print('shap value shape is ', shap_value.shape)
            np.save(os.path.join(shap_dir, f'shap_{self.args.task}_{name}'), shap_value[1].squeeze().sum(0))

    def forward(self, IMG):
        feats = self.nets.CNN(IMG)
        # Average Pool
        feats = F.adaptive_max_pool3d(feats, 1).view(IMG.shape[0], -1)
        # forward MLP
        logits = self.nets.CLS(feats)
        return torch.nn.functional.softmax(logits)


class model_CLS_Transformer(nn.Module):
    def __init__(self, args, fold, modality):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'Transformer')
        if args.task == 'MCIC' or args.task == 'ADD':
            class_num = 2
        else:
            class_num = 3
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.expr_dir, str(fold))
        self.modality = 'MRI'
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = torch.optim.AdamW(
                    params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                    eps=1e-08, weight_decay=args.weight_decay)
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        else:
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_label = Compose([AsDiscrete(to_onehot=class_num)])
        # define evaluation metrics
        self.loss_epoch = CumulativeAverage()
        self.AUC = ROCAUCMetric(average='macro')

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def _reset_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = os.path.join(checkpoint_dir, str(self.fold))
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]

    def compute_CLS_loss(self, IMG, label):
        # forward CNN
        feats = self.nets.CNN(IMG)
        # Reshape to vectors
        feats = rearrange(feats, 'b c h w d -> b (h w d) c')
        # forward Trans
        feats = self.nets.Trans(feats)
        # Pooling
        gap = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveAvgPool1d(1), Rearrange('b d n -> b (d n)'))
        gmp = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveMaxPool1d(1), Rearrange('b d n -> b (d n)'))
        cls_avg = gap(feats)
        cls_max = gmp(feats)
        cls_token = torch.cat([cls_avg, cls_max], dim=1)
        # forward MLP
        logits = self.nets.CLS(cls_token)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def start_train(self, train_loaders, val_loader):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_epoch = 1
        self.logger.print_message('Start training...')
        start_time = time.time()

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            # initialize or reset training predictions
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                IMG = data['MRI'].to(self.device)
                DX = data['label'].to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(IMG, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.CNN.step()
                self.optims.Trans.step()
                self.optims.CLS.step()
                # compute training metrics
                self.loss_epoch.append(loss_CLS)
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                label_real += [i for i in decollate_batch(data['label'])]
                label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log training status
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')

            # log training metrics
            loss_results = self.loss_epoch.aggregate()
            cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_train.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_train.TNR_Macro):.4f} '
                                          f'F1:{float(cm_train.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_train.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_train.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_train.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # validation iterations
            for name, model in self.nets.items():
                model.eval()

            # reset metrics
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    IMG = data['MRI'].to(self.device)
                    DX = data['label'].to(self.device)
                    logits, loss_CLS = self.compute_CLS_loss(IMG, DX)
                    # compute training metrics
                    self.loss_epoch.append(loss_CLS)
                    self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                             y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    label_real += [i for i in decollate_batch(data['label'])]
                    label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log validation metrics
            loss_results = self.loss_epoch.aggregate()
            cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_val.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_val.TNR_Macro):.4f} '
                                          f'F1:{float(cm_val.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_val.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_val.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_val.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # save best model according to the validation results
            acc = float(cm_val.Overall_ACC)
            f1 = float(cm_val.F1_Macro)
            if acc >= best_acc and f1 >= best_f1:
                self._save_checkpoint()
                best_acc = acc
                best_f1 = f1
                best_epoch = epoch + 1

    def start_test(self, test_loader, phase):
        args = self.args
        # start test iterations
        self._load_checkpoint()
        content = []
        # reset training predictions
        self.AUC.reset()
        # test loop
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                IMG = data['MRI'].to(self.device)
                DX = data['label'].to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(IMG, DX)
                pred_prob = [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits)]
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                # write pred information
                label = data['label'][0].numpy()
                filename = data['filename'][0]
                case = {'filename': filename}
                case[args.task] = int(label)
                if args.task == 'COG':
                    case['COG_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['COG_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['COG_score_2'] = f'{pred_prob[0][2]:.8f}'
                    case['COG_pred'] = pred_prob[0].argmax()
                elif args.task == 'ADD':
                    case['ADD_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['ADD_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['ADD_pred'] = pred_prob[0].argmax()
                elif args.task == 'MCIC':
                    case['MCIC_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['MCIC_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['MCIC_pred'] = pred_prob[0].argmax()
                content.append(case)

        with open(os.path.join(self.checkpoint_dir, phase + '_eval.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
            writer.writeheader()
            for case in content:
                writer.writerow(case)

        # log validation metrics
        test_res_all = self.post_cls_eval(phase)
        self.logger.print_message(f'ACC:{test_res_all[0]:.4f} '
                                  f'MCC:{test_res_all[1]:.4f} '
                                  f'SEN:{test_res_all[2]:.4f} '
                                  f'SPE:{test_res_all[3]:.4f} '
                                  f'F1:{test_res_all[4]:.4f} '
                                  f'AUC:{self.AUC.aggregate():.4f}')
        return test_res_all + [self.AUC.aggregate()]

    def post_cls_eval(self, stage):
        csv_file = os.path.join(self.checkpoint_dir, stage + '_eval.csv')
        label, pred, pred_prob = utils.get_pd_gt(csv_file, self.args.task)
        cm = ConfusionMatrix(actual_vector=label, predict_vector=pred)
        if self.args == 'COG':
            acc = cm.Overall_ACC
            mcc = cm.Overall_MCC
            sen = cm.TPR_Macro
            spe = cm.TNR_Macro
            f1 = cm.F1_Macro
            auc = cm.AUC
            return [acc, mcc, sen, spe, f1, np.mean(list(auc.values()))]
        else:
            acc = cm.Overall_ACC
            mcc = list(cm.MCC.values())[1]
            sen = list(cm.TPR.values())[1]
            spe = list(cm.TNR.values())[1]
            f1 = list(cm.F1.values())[1]
            auc = list(cm.AUC.values())[1]
            return [acc, mcc, sen, spe, f1, auc]


class model_CLS_Transformer_T(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'Transformer_T')
        if args.task == 'MCIC' or args.task == 'ADD':
            class_num = 2
        else:
            class_num = 3
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.expr_dir, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        self.optims = Munch()
        for net in self.nets.keys():
            self.optims[net] = torch.optim.AdamW(
                params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=args.weight_decay)
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_label = Compose([AsDiscrete(to_onehot=class_num)])
        # define evaluation metrics
        self.loss_epoch = CumulativeAverage()
        self.AUC = ROCAUCMetric(average='macro')

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def _reset_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = os.path.join(checkpoint_dir, str(self.fold))
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]

    def compute_CLS_loss(self, IMG, tabular, tabular_missing, label):
        # forward CNN
        feats = self.nets.CNN(IMG)
        # Reshape to vectors
        feats = rearrange(feats, 'b c h w d -> b (h w d) c')
        # tabular embedding
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        tabular_feats = torch.flatten(tabular_feats, start_dim=1, end_dim=-1)
        feats = self.nets.Trans(feats, tabular_feats)
        # Pooling
        gap = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveAvgPool1d(1), Rearrange('b d n -> b (d n)'))
        gmp = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveMaxPool1d(1), Rearrange('b d n -> b (d n)'))
        cls_avg = gap(feats)
        cls_max = gmp(feats)
        cls_token = torch.cat([cls_avg, cls_max], dim=1)
        # forward MLP
        logits = self.nets.CLS(cls_token)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def start_train(self, train_loaders, val_loader):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_epoch = 1
        self.logger.print_message('Start training...')
        start_time = time.time()

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            # initialize or reset training predictions
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                MRI = data['MRI'].to(self.device)
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                DX = data['label'].to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, tabular, tabular_missing, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.CNN.step()
                self.optims.Trans.step()
                self.optims.T_Embedding.step()
                self.optims.CLS.step()
                # compute training metrics
                self.loss_epoch.append(loss_CLS)
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                label_real += [i for i in decollate_batch(data['label'])]
                label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log training status
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')

            # log training metrics
            loss_results = self.loss_epoch.aggregate()
            cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_train.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_train.TNR_Macro):.4f} '
                                          f'F1:{float(cm_train.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_train.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_train.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_train.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            # validation iterations
            for name, model in self.nets.items():
                model.eval()

            # reset metrics
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    MRI = data['MRI'].to(self.device)
                    tabular = data['tabular'].to(self.device)
                    tabular_missing = data['tabular_missing'].to(self.device)
                    DX = data['label'].to(self.device)
                    logits, loss_CLS = self.compute_CLS_loss(MRI, tabular, tabular_missing, DX)
                    # compute training metrics
                    self.loss_epoch.append(loss_CLS)
                    self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                             y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    label_real += [i for i in decollate_batch(data['label'])]
                    label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log validation metrics
            loss_results = self.loss_epoch.aggregate()
            cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_val.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_val.TNR_Macro):.4f} '
                                          f'F1:{float(cm_val.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_val.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_val.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_val.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # save best model according to the validation results
            acc = float(cm_val.Overall_ACC)
            f1 = float(cm_val.F1_Macro)
            if acc >= best_acc and f1 >= best_f1:
                self._save_checkpoint()
                best_acc = acc
                best_f1 = f1
                best_epoch = epoch + 1

    def start_test(self, test_loader, phase):
        args = self.args
        # load best model
        self._load_checkpoint()
        content = []
        # reset training predictions
        self.AUC.reset()
        # test loop
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                MRI = data['MRI'].to(self.device)
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                DX = data['label'].to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, tabular, tabular_missing, DX)
                pred_prob = [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits)]
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                # write pred information
                label = data['label'][0].numpy()
                filename = data['filename'][0]
                case = {'filename': filename}
                case[args.task] = int(label)
                if args.task == 'COG':
                    case['COG_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['COG_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['COG_score_2'] = f'{pred_prob[0][2]:.8f}'
                    case['COG_pred'] = pred_prob[0].argmax()
                elif args.task == 'ADD':
                    case['ADD_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['ADD_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['ADD_pred'] = pred_prob[0].argmax()
                elif args.task == 'MCIC':
                    case['MCIC_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['MCIC_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['MCIC_pred'] = pred_prob[0].argmax()
                content.append(case)

        # write to csv
        with open(os.path.join(self.checkpoint_dir, phase + '_eval.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
            writer.writeheader()
            for case in content:
                writer.writerow(case)

        # log validation metrics
        test_res_all = self.post_cls_eval(phase)
        self.logger.print_message(f'ACC:{test_res_all[0]:.4f} '
                                  f'MCC:{test_res_all[1]:.4f} '
                                  f'SEN:{test_res_all[2]:.4f} '
                                  f'SPE:{test_res_all[3]:.4f} '
                                  f'F1:{test_res_all[4]:.4f} '
                                  f'AUC:{self.AUC.aggregate():.4f}')

        return test_res_all + [self.AUC.aggregate()]

    def post_cls_eval(self, stage):
        csv_file = os.path.join(self.checkpoint_dir, stage + '_eval.csv')
        label, pred, pred_prob = utils.get_pd_gt(csv_file, self.args.task)
        cm = ConfusionMatrix(actual_vector=label, predict_vector=pred)
        if self.args == 'COG':
            acc = cm.Overall_ACC
            mcc = cm.Overall_MCC
            sen = cm.TPR_Macro
            spe = cm.TNR_Macro
            f1 = cm.F1_Macro
            auc = cm.AUC
            return [acc, mcc, sen, spe, f1, np.mean(list(auc.values()))]
        else:
            acc = cm.Overall_ACC
            mcc = list(cm.MCC.values())[1]
            sen = list(cm.TPR.values())[1]
            spe = list(cm.TNR.values())[1]
            f1 = list(cm.F1.values())[1]
            auc = list(cm.AUC.values())[1]
            return [acc, mcc, sen, spe, f1, auc]


class model_CLS_Transformer_IT(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'Transformer_IT')
        if args.task == 'MCIC' or args.task == 'ADD':
            class_num = 2
        else:
            class_num = 3
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.expr_dir, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        self.optims = Munch()
        for net in self.nets.keys():
            self.optims[net] = torch.optim.AdamW(
                params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=args.weight_decay)
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_label = Compose([AsDiscrete(to_onehot=class_num)])
        # define evaluation metrics
        self.loss_epoch = CumulativeAverage()
        self.AUC = ROCAUCMetric(average='macro')
        ## load pretrained checkpoints
        # ckpt_path = './checkpoints/pretrained.p'
        # self.CNN.load_state_dict(torch.load(ckpt_path), strict=False)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def _reset_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = os.path.join(checkpoint_dir, str(self.fold))
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]

    def compute_CLS_loss(self, IMG, tabular, tabular_missing, label):
        # forward CNN
        feats = self.nets.CNN(IMG)
        # Reshape to vectors
        feats = rearrange(feats, 'b c h w d -> b (h w d) c')
        # tabular embedding
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        tabular_feats = torch.flatten(tabular_feats, start_dim=1, end_dim=-1)
        feats = self.nets.Trans(feats, tabular_feats)
        # Pooling
        gap = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveAvgPool1d(1), Rearrange('b d n -> b (d n)'))
        gmp = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveMaxPool1d(1), Rearrange('b d n -> b (d n)'))
        cls_avg = gap(feats)
        cls_max = gmp(feats)
        cls_token = torch.cat([cls_avg, cls_max], dim=1)
        # forward MLP
        logits = self.nets.CLS(cls_token)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def start_train(self, train_loaders, val_loader):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_auc = 0.0
        best_epoch = 1
        self.logger.print_message('Start training...')
        start_time = time.time()

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            # initialize or reset training predictions
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                MRI = data['MRI'].to(self.device)
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                DX = data['label'].to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, tabular, tabular_missing, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.CNN.step()
                self.optims.Trans.step()
                self.optims.T_Embedding.step()
                self.optims.CLS.step()
                # compute training metrics
                self.loss_epoch.append(loss_CLS)
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                label_real += [i for i in decollate_batch(data['label'])]
                label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log training status
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')

            # log training metrics
            loss_results = self.loss_epoch.aggregate()
            cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_train.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_train.TNR_Macro):.4f} '
                                          f'F1:{float(cm_train.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_train.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_train.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_train.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # validation iterations
            for name, model in self.nets.items():
                model.eval()

            # reset metrics
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    MRI = data['MRI'].to(self.device)
                    tabular = data['tabular'].to(self.device)
                    tabular_missing = data['tabular_missing'].to(self.device)
                    DX = data['label'].to(self.device)
                    logits, loss_CLS = self.compute_CLS_loss(MRI, tabular, tabular_missing, DX)
                    # compute training metrics
                    self.loss_epoch.append(loss_CLS)
                    self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                             y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    label_real += [i for i in decollate_batch(data['label'])]
                    label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log validation metrics
            loss_results = self.loss_epoch.aggregate()
            cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_val.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_val.TNR_Macro):.4f} '
                                          f'F1:{float(cm_val.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_val.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_val.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_val.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # save best model according to the validation results
            if self.args.task == 'ADD':
                auc = self.AUC.aggregate()
                if auc >= best_auc:
                    self._save_checkpoint()
                    best_auc = auc
                    best_epoch = epoch + 1
            else:
                acc = float(cm_val.Overall_ACC)
                f1 = float(cm_val.F1_Macro)
                if acc >= best_acc and f1 >= best_f1:
                    self._save_checkpoint()
                    best_acc = acc
                    best_f1 = f1
                    best_epoch = epoch + 1

    def start_test(self, test_loader, phase):
        args = self.args
        # load best model
        self._load_checkpoint()
        content = []
        # reset training predictions
        self.AUC.reset()
        # test loop
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                MRI = data['MRI'].to(self.device)
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                DX = data['label'].to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, tabular, tabular_missing, DX)
                pred_prob = [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits)]
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                # write pred information
                label = data['label'][0].numpy()
                filename = data['filename'][0]
                case = {'filename': filename}
                case[args.task] = int(label)
                if args.task == 'COG':
                    case['COG_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['COG_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['COG_score_2'] = f'{pred_prob[0][2]:.8f}'
                    case['COG_pred'] = pred_prob[0].argmax()
                elif args.task == 'ADD':
                    case['ADD_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['ADD_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['ADD_pred'] = pred_prob[0].argmax()
                elif args.task == 'MCIC':
                    case['MCIC_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['MCIC_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['MCIC_pred'] = pred_prob[0].argmax()
                content.append(case)

        # write to csv
        with open(os.path.join(self.checkpoint_dir, phase + '_eval.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
            writer.writeheader()
            for case in content:
                writer.writerow(case)

        # log validation metrics
        test_res_all = self.post_cls_eval(phase)
        try:
            self.logger.print_message(f'ACC:{test_res_all[0]:.4f} '
                                      f'MCC:{test_res_all[1]:.4f} '
                                      f'SEN:{test_res_all[2]:.4f} '
                                      f'SPE:{test_res_all[3]:.4f} '
                                      f'F1:{test_res_all[4]:.4f} '
                                      f'AUC:{self.AUC.aggregate():.4f}')
        except:
            print('error on metrics')

        return test_res_all + [self.AUC.aggregate()]

    def post_cls_eval(self, stage):
        csv_file = os.path.join(self.checkpoint_dir, stage + '_eval.csv')
        label, pred, pred_prob = utils.get_pd_gt(csv_file, self.args.task)
        cm = ConfusionMatrix(actual_vector=label, predict_vector=pred)
        if self.args.task == 'COG':
            acc = cm.Overall_ACC
            mcc = cm.Overall_MCC
            sen = cm.TPR_Macro
            spe = cm.TNR_Macro
            f1 = cm.F1_Macro
            auc = cm.AUC
            return [acc, mcc, sen, spe, f1, np.mean(list(auc.values()))]
        else:
            acc = cm.Overall_ACC
            mcc = list(cm.MCC.values())[1]
            sen = list(cm.TPR.values())[1]
            spe = list(cm.TNR.values())[1]
            f1 = list(cm.F1.values())[1]
            auc = list(cm.AUC.values())[1]
            return [acc, mcc, sen, spe, f1, auc]


class model_CLS_CrossTransformer_IT(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'CrossTransformer_IT')
        if args.task == 'MCIC' or args.task == 'ADD':
            class_num = 2
        else:
            class_num = 3
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.expr_dir, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        self.optims = Munch()
        for net in self.nets.keys():
            self.optims[net] = torch.optim.AdamW(
                params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=args.weight_decay)
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_label = Compose([AsDiscrete(to_onehot=class_num)])
        # define evaluation metrics
        self.loss_epoch = CumulativeAverage()
        self.AUC = ROCAUCMetric(average='macro')

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def _reset_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = os.path.join(checkpoint_dir, str(self.fold))
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]

    def compute_CLS_loss_Tabualr(self, tabular, tabular_missing, label):
        # tabular embedding
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        tabular_feats = torch.flatten(tabular_feats, start_dim=1, end_dim=-1)
        # forward MLP
        logits = self.nets.Tabular_CLS(tabular_feats)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def compute_CLS_loss_MRI(self, MRI, tabular, tabular_missing, label):
        # forward CNN
        MRI_feats = self.nets.MRI(MRI)
        # Reshape to vectors
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        # tabular embedding
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        tabular_feats = torch.flatten(tabular_feats, start_dim=1, end_dim=-1)
        # forward cross transformer
        MRI_feats = self.nets.MRI_Trans(MRI_feats, tabular_feats)
        # Pooling
        gap = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveAvgPool1d(1), Rearrange('b d n -> b (d n)'))
        gmp = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveMaxPool1d(1), Rearrange('b d n -> b (d n)'))
        cls_avg = gap(MRI_feats)
        cls_max = gmp(MRI_feats)
        cls_token = torch.cat([cls_avg, cls_max], dim=1)
        # forward MLP
        logits = self.nets.MRI_CLS(cls_token)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def compute_CLS_loss_PET(self, MRI, PET, tabular, tabular_missing, label):
        # forward CNN
        MRI_feats = self.nets.MRI(MRI)
        PET_feats = self.nets.PET(PET)
        # Reshape to vectors
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        PET_feats = rearrange(PET_feats, 'b c h w d -> b (h w d) c')
        # tabular embedding
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        tabular_feats = torch.flatten(tabular_feats, start_dim=1, end_dim=-1)
        # forward cross transformer
        PET_feats = self.nets.PET_Trans(MRI_feats, PET_feats, tabular_feats)
        # Pooling
        gap = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveAvgPool1d(1), Rearrange('b d n -> b (d n)'))
        gmp = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveMaxPool1d(1), Rearrange('b d n -> b (d n)'))
        cls_avg = gap(PET_feats)
        cls_max = gmp(PET_feats)
        cls_token = torch.cat([cls_avg, cls_max], dim=1)
        # forward MLP
        logits = self.nets.PET_CLS(cls_token)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def start_train(self, train_loaders, train_loaders2, val_loader):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_auc = 0.0
        best_epoch = 1
        self.logger.print_message('Start training...')
        start_time = time.time()

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            # initialize or reset training predictions
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []
            # init PET dataloader
            PETloader_iterator = iter(train_loaders2)
            for i, data in enumerate(train_loaders):
                MRI = data['MRI'].to(self.device)
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                DX = data['label'].to(self.device)
                # forward Tabular only
                _, loss_CLS = self.compute_CLS_loss_Tabualr(tabular, tabular_missing, DX)
                self._reset_grad()
                loss_CLS.backward()
                self.optims.T_Embedding.step()
                self.optims.Tabular_CLS.step()

                # forward MRI and Tabular
                _, loss_CLS1 = self.compute_CLS_loss_MRI(MRI, tabular, tabular_missing, DX)
                # backward pass 1
                self._reset_grad()
                loss_CLS1.backward()
                self.optims.MRI.step()
                self.optims.MRI_Trans.step()
                self.optims.MRI_CLS.step()

                # forward MRI and PET
                try:
                    data2 = next(PETloader_iterator)
                except StopIteration:
                    PETloader_iterator = iter(train_loaders2)
                    data2 = next(PETloader_iterator)
                MRI = data2['MRI'].to(self.device)
                PET = data2['PET'].to(self.device)
                tabular = data2['tabular'].to(self.device)
                tabular_missing = data2['tabular_missing'].to(self.device)
                DX = data2['label'].to(self.device)
                logits2, loss_CLS2 = self.compute_CLS_loss_PET(MRI, PET, tabular, tabular_missing, DX)
                # backward pass 2
                self._reset_grad()
                loss_CLS2.backward()
                self.optims.PET.step()
                self.optims.PET_Trans.step()
                self.optims.PET_CLS.step()
                # compute training metrics
                self.loss_epoch.append(loss_CLS2)
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits2)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                label_real += [i for i in decollate_batch(data2['label'])]
                label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits2)]

            # log training metrics
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')

            # log training metrics
            loss_results = self.loss_epoch.aggregate()
            cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_train.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_train.TNR_Macro):.4f} '
                                          f'F1:{float(cm_train.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_train.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_train.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_train.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # validation iterations
            for name, model in self.nets.items():
                model.eval()

            # reset metrics
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []
            val_start_time = time.time()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    MRI = data['MRI'].to(self.device)
                    tabular = data['tabular'].to(self.device)
                    tabular_missing = data['tabular_missing'].to(self.device)
                    DX = data['label'].to(self.device)
                    PET = data['PET'].to(self.device)

                    if torch.equal(PET, torch.zeros_like(PET)):
                        logits, loss_CLS = self.compute_CLS_loss_MRI(MRI, tabular, tabular_missing, DX)
                    else:
                        logits, loss_CLS = self.compute_CLS_loss_PET(MRI, PET, tabular, tabular_missing, DX)
                    # compute validation metrics
                    self.loss_epoch.append(loss_CLS)
                    self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                             y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    label_real += [i for i in decollate_batch(data['label'])]
                    label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]
            val_elapsed = time.time() - val_start_time
            val_elapsed = str(datetime.timedelta(seconds=val_elapsed))[:-7]
            print(val_elapsed)
            # log validation metrics
            loss_results = self.loss_epoch.aggregate()
            cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_val.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_val.TNR_Macro):.4f} '
                                          f'F1:{float(cm_val.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_val.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_val.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_val.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # save best model according to the validation results
            if self.args.task == 'ADD':
                auc = self.AUC.aggregate()
                if auc >= best_auc:
                    self._save_checkpoint()
                    best_auc = auc
                    best_epoch = epoch + 1
            else:
                acc = float(cm_val.Overall_ACC)
                f1 = float(cm_val.F1_Macro)
                if acc >= best_acc and f1 >= best_f1:
                    self._save_checkpoint()
                    best_acc = acc
                    best_f1 = f1
                    best_epoch = epoch + 1

    def start_test(self, test_loader, phase):
        args = self.args
        # start test iterations
        self._load_checkpoint()
        content = []
        for name, model in self.nets.items():
            model.eval()
        # reset training predictions
        self.AUC.reset()
        # test loop
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                MRI = data['MRI'].to(self.device)
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                DX = data['label'].to(self.device)
                PET = data['PET'].to(self.device)
                if torch.equal(PET, torch.zeros_like(PET)):
                    logits, loss_CLS = self.compute_CLS_loss_MRI(MRI, tabular, tabular_missing, DX)
                else:
                    logits, loss_CLS = self.compute_CLS_loss_PET(MRI, PET, tabular, tabular_missing, DX)
                pred_prob = [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits)]
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                # write pred information
                label = data['label'][0].numpy()
                filename = data['filename'][0]
                case = {'filename': filename}
                case[args.task] = int(label)
                if args.task == 'COG':
                    case['COG_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['COG_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['COG_score_2'] = f'{pred_prob[0][2]:.8f}'
                    case['COG_pred'] = pred_prob[0].argmax()
                elif args.task == 'ADD':
                    case['ADD_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['ADD_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['ADD_pred'] = pred_prob[0].argmax()
                elif args.task == 'MCIC':
                    case['MCIC_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['MCIC_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['MCIC_pred'] = pred_prob[0].argmax()
                content.append(case)

        with open(os.path.join(self.checkpoint_dir, phase + '_eval.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
            writer.writeheader()
            for case in content:
                writer.writerow(case)

        # log validation metrics
        test_res_all = self.post_cls_eval(phase)
        try:
            self.logger.print_message(f'ACC:{test_res_all[0]:.4f} '
                                      f'MCC:{test_res_all[1]:.4f} '
                                      f'SEN:{test_res_all[2]:.4f} '
                                      f'SPE:{test_res_all[3]:.4f} '
                                      f'F1:{test_res_all[4]:.4f} '
                                      f'AUC:{self.AUC.aggregate():.4f}')
        except:
            self.logger.print_message('some wrong with metrics')
        return test_res_all + [self.AUC.aggregate()]

    def post_cls_eval(self, stage):
        csv_file = os.path.join(self.checkpoint_dir, stage + '_eval.csv')
        label, pred, pred_prob = utils.get_pd_gt(csv_file, self.args.task)
        cm = ConfusionMatrix(actual_vector=label, predict_vector=pred)
        if self.args == 'COG':
            acc = cm.Overall_ACC
            mcc = cm.Overall_MCC
            sen = cm.TPR_Macro
            spe = cm.TNR_Macro
            f1 = cm.F1_Macro
            auc = cm.AUC
            return [acc, mcc, sen, spe, f1, np.mean(list(auc.values()))]
        else:
            acc = cm.Overall_ACC
            mcc = list(cm.MCC.values())[1]
            sen = list(cm.TPR.values())[1]
            spe = list(cm.TNR.values())[1]
            f1 = list(cm.F1.values())[1]
            auc = list(cm.AUC.values())[1]
            return [acc, mcc, sen, spe, f1, auc]

    def shap_set_input(self, tabular_missing, MRI, PET, DX):
        self.tabular_missing = torch.from_numpy(tabular_missing).to(self.device)
        self.data_MRI = torch.from_numpy(MRI).to(self.device)
        self.data_PET = torch.from_numpy(PET).to(self.device)
        self.DX = torch.from_numpy(DX).to(self.device)

    def shap_forward(self, tabular):
        tabular = tabular.to(self.device)
        n = tabular.shape[0]
        n_other = self.data_MRI.shape[0]
        if n > 1 and n_other == 1:
            pred_prob_out = []
            for i in range(n):
                if torch.equal(self.data_PET, torch.zeros_like(self.data_PET)):
                    logits, loss_CLS = self.compute_CLS_loss_MRI(self.data_MRI, tabular[i:i+1], self.tabular_missing, self.DX)
                else:
                    logits, loss_CLS = self.compute_CLS_loss_PET(self.data_MRI, self.data_PET, tabular[i:i+1], self.tabular_missing, self.DX)
                pred_prob = [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits)]
                pred_prob_out.append(pred_prob)
            pred_prob_out = np.concatenate(pred_prob_out, 0)
        else:
            if torch.equal(self.data_PET, torch.zeros_like(self.data_PET)):
                logits, loss_CLS = self.compute_CLS_loss_MRI(self.data_MRI, tabular, self.tabular_missing, self.DX)
            else:
                logits, loss_CLS = self.compute_CLS_loss_PET(self.data_MRI, self.data_PET, tabular, self.tabular_missing, self.DX)
            pred_prob = [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits)]
            pred_prob_out = np.stack(pred_prob, 0)

        return pred_prob_out


class model_CLS_CrossTransformer_IT_Stage(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'CrossTransformer_IT')
        if args.task == 'MCIC' or args.task == 'ADD':
            class_num = 2
        else:
            class_num = 3
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.expr_dir, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        self.optims = Munch()
        for net in self.nets.keys():
            self.optims[net] = torch.optim.AdamW(
                params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=args.weight_decay)
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_label = Compose([AsDiscrete(to_onehot=class_num)])
        # define evaluation metrics
        self.loss_epoch = CumulativeAverage()
        self.AUC = ROCAUCMetric(average='macro')

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def _reset_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = os.path.join(checkpoint_dir, str(self.fold))
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]

    def compute_CLS_loss_Tabualr(self, tabular, tabular_missing, label):
        # tabular embedding
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        tabular_feats = torch.flatten(tabular_feats, start_dim=1, end_dim=-1)
        # forward MLP
        logits = self.nets.Tabular_CLS(tabular_feats)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def compute_CLS_loss_MRI(self, MRI, tabular, tabular_missing, label):
        # forward CNN
        MRI_feats = self.nets.MRI(MRI)
        # Reshape to vectors
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        # tabular embedding
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        tabular_feats = torch.flatten(tabular_feats, start_dim=1, end_dim=-1)
        # forward cross transformer
        MRI_feats = self.nets.MRI_Trans(MRI_feats, tabular_feats)
        # Pooling
        gap = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveAvgPool1d(1), Rearrange('b d n -> b (d n)'))
        gmp = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveMaxPool1d(1), Rearrange('b d n -> b (d n)'))
        cls_avg = gap(MRI_feats)
        cls_max = gmp(MRI_feats)
        cls_token = torch.cat([cls_avg, cls_max], dim=1)
        # forward MLP
        logits = self.nets.MRI_CLS(cls_token)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def compute_CLS_loss_PET(self, MRI, PET, tabular, tabular_missing, label):
        # forward CNN
        MRI_feats = self.nets.MRI(MRI)
        PET_feats = self.nets.PET(PET)
        # Reshape to vectors
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        PET_feats = rearrange(PET_feats, 'b c h w d -> b (h w d) c')
        # tabular embedding
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        tabular_feats = torch.flatten(tabular_feats, start_dim=1, end_dim=-1)
        # forward cross transformer
        PET_feats = self.nets.PET_Trans(MRI_feats, PET_feats, tabular_feats)
        # Pooling
        gap = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveAvgPool1d(1), Rearrange('b d n -> b (d n)'))
        gmp = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveMaxPool1d(1), Rearrange('b d n -> b (d n)'))
        cls_avg = gap(PET_feats)
        cls_max = gmp(PET_feats)
        cls_token = torch.cat([cls_avg, cls_max], dim=1)
        # forward MLP
        logits = self.nets.PET_CLS(cls_token)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def train_stage1(self, train_loader, val_loader):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_auc = 0.0
        best_epoch = 1
        self.logger.print_message('Start training for stage 1 ...')
        start_time = time.time()

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            # initialize or reset training predictions
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []
            for i, data in enumerate(train_loader):
                MRI = data['MRI'].to(self.device)
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                DX = data['label'].to(self.device)
                # forward Tabular only
                logits1, loss_CLS1 = self.compute_CLS_loss_Tabualr(tabular, tabular_missing, DX)
                # forward MRI and Tabular
                logits2, loss_CLS2 = self.compute_CLS_loss_MRI(MRI, tabular, tabular_missing, DX)

                loss = (loss_CLS1 + loss_CLS2) / 2

                # backward pass 1
                self._reset_grad()
                loss.backward()
                self.optims.T_Embedding.step()
                self.optims.Tabular_CLS.step()
                self.optims.MRI.step()
                self.optims.MRI_Trans.step()
                self.optims.MRI_CLS.step()

                # compute training metrics
                self.loss_epoch.append(loss)
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits2)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                label_real += [i for i in decollate_batch(data['label'])]
                label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits2)]

            # log training metrics
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')

            # log training metrics
            loss_results = self.loss_epoch.aggregate()
            cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_train.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_train.TNR_Macro):.4f} '
                                          f'F1:{float(cm_train.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_train.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_train.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_train.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # validation iterations
            for name, model in self.nets.items():
                model.eval()

            # reset metrics
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    MRI = data['MRI'].to(self.device)
                    tabular = data['tabular'].to(self.device)
                    tabular_missing = data['tabular_missing'].to(self.device)
                    DX = data['label'].to(self.device)
                    logits, loss_CLS = self.compute_CLS_loss_MRI(MRI, tabular, tabular_missing, DX)
                    # compute validation metrics
                    self.loss_epoch.append(loss_CLS)
                    self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                             y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    label_real += [i for i in decollate_batch(data['label'])]
                    label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log validation metrics
            loss_results = self.loss_epoch.aggregate()
            cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_val.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_val.TNR_Macro):.4f} '
                                          f'F1:{float(cm_val.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_val.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_val.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_val.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # save best model according to the validation results
            if self.args.task == 'ADD':
                auc = self.AUC.aggregate()
                if auc >= best_auc:
                    self._save_checkpoint()
                    best_auc = auc
                    best_epoch = epoch + 1
            else:
                acc = float(cm_val.Overall_ACC)
                f1 = float(cm_val.F1_Macro)
                if acc >= best_acc and f1 >= best_f1:
                    self._save_checkpoint()
                    best_acc = acc
                    best_f1 = f1
                    best_epoch = epoch + 1

    def train_stage2(self, train_loader, val_loader):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_auc = 0.0
        best_epoch = 1
        self.logger.print_message('Start training for stage 2 ...')
        start_time = time.time()
        # load checkpoint from stage 1
        self.logger.print_message('Load best network from stage 1 ...')
        self._load_checkpoint()

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            # initialize or reset training predictions
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []
            for i, data in enumerate(train_loader):
                MRI = data['MRI'].to(self.device)
                PET = data['PET'].to(self.device)
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                DX = data['label'].to(self.device)
                # forward
                logits, loss_CLS = self.compute_CLS_loss_PET(MRI, PET, tabular, tabular_missing, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.PET.step()
                self.optims.PET_Trans.step()
                self.optims.PET_CLS.step()
                # compute training metrics
                self.loss_epoch.append(loss_CLS)
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                label_real += [i for i in decollate_batch(data['label'])]
                label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log training metrics
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')

            # log training metrics
            loss_results = self.loss_epoch.aggregate()
            cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_train.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_train.TNR_Macro):.4f} '
                                          f'F1:{float(cm_train.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_train.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_train.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_train.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # validation iterations
            for name, model in self.nets.items():
                model.eval()

            # reset metrics
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    MRI = data['MRI'].to(self.device)
                    tabular = data['tabular'].to(self.device)
                    tabular_missing = data['tabular_missing'].to(self.device)
                    DX = data['label'].to(self.device)
                    PET = data['PET'].to(self.device)
                    if torch.equal(PET, torch.zeros_like(PET)):
                        logits, loss_CLS = self.compute_CLS_loss_MRI(MRI, tabular, tabular_missing, DX)
                    else:
                        logits, loss_CLS = self.compute_CLS_loss_PET(MRI, PET, tabular, tabular_missing, DX)
                    # compute validation metrics
                    self.loss_epoch.append(loss_CLS)
                    self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                             y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    label_real += [i for i in decollate_batch(data['label'])]
                    label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log validation metrics
            loss_results = self.loss_epoch.aggregate()
            cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_val.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_val.TNR_Macro):.4f} '
                                          f'F1:{float(cm_val.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_val.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_val.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_val.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # save best model according to the validation results
            if self.args.task == 'ADD':
                auc = self.AUC.aggregate()
                if auc >= best_auc:
                    self._save_checkpoint()
                    best_auc = auc
                    best_epoch = epoch + 1
            else:
                acc = float(cm_val.Overall_ACC)
                f1 = float(cm_val.F1_Macro)
                if acc >= best_acc and f1 >= best_f1:
                    self._save_checkpoint()
                    best_acc = acc
                    best_f1 = f1
                    best_epoch = epoch + 1

    def start_train(self, train_loader, train_loader2, val_loader):
        self.train_stage1(train_loader, val_loader)
        self.train_stage2(train_loader2, val_loader)

    def start_test(self, test_loader, phase):
        args = self.args
        # start test iterations
        self._load_checkpoint()
        content = []
        # reset training predictions
        self.AUC.reset()
        # test loop
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                MRI = data['MRI'].to(self.device)
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                DX = data['label'].to(self.device)
                PET = data['PET'].to(self.device)
                if torch.equal(PET, torch.zeros_like(PET)):
                    logits, loss_CLS = self.compute_CLS_loss_MRI(MRI, tabular, tabular_missing, DX)
                else:
                    logits, loss_CLS = self.compute_CLS_loss_PET(MRI, PET, tabular, tabular_missing, DX)
                pred_prob = [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits)]
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                # write pred information
                label = data['label'][0].numpy()
                filename = data['filename'][0]
                case = {'filename': filename}
                case[args.task] = int(label)
                if args.task == 'COG':
                    case['COG_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['COG_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['COG_score_2'] = f'{pred_prob[0][2]:.8f}'
                    case['COG_pred'] = pred_prob[0].argmax()
                elif args.task == 'ADD':
                    case['ADD_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['ADD_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['ADD_pred'] = pred_prob[0].argmax()
                elif args.task == 'MCIC':
                    case['MCIC_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['MCIC_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['MCIC_pred'] = pred_prob[0].argmax()
                content.append(case)

        with open(os.path.join(self.checkpoint_dir, phase + '_eval.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
            writer.writeheader()
            for case in content:
                writer.writerow(case)

        # log validation metrics
        test_res_all = self.post_cls_eval(phase)
        self.logger.print_message(f'ACC:{test_res_all[0]:.4f} '
                                  f'MCC:{test_res_all[1]:.4f} '
                                  f'SEN:{test_res_all[2]:.4f} '
                                  f'SPE:{test_res_all[3]:.4f} '
                                  f'F1:{test_res_all[4]:.4f} '
                                  f'AUC:{self.AUC.aggregate():.4f}')
        return test_res_all + [self.AUC.aggregate()]

    def post_cls_eval(self, stage):
        csv_file = os.path.join(self.checkpoint_dir, stage + '_eval.csv')
        label, pred, pred_prob = utils.get_pd_gt(csv_file, self.args.task)
        cm = ConfusionMatrix(actual_vector=label, predict_vector=pred)
        if self.args == 'COG':
            acc = cm.Overall_ACC
            mcc = cm.Overall_MCC
            sen = cm.TPR_Macro
            spe = cm.TNR_Macro
            f1 = cm.F1_Macro
            auc = cm.AUC
            return [acc, mcc, sen, spe, f1, np.mean(list(auc.values()))]
        else:
            acc = cm.Overall_ACC
            mcc = list(cm.MCC.values())[1]
            sen = list(cm.TPR.values())[1]
            spe = list(cm.TNR.values())[1]
            f1 = list(cm.F1.values())[1]
            auc = list(cm.AUC.values())[1]
            return [acc, mcc, sen, spe, f1, auc]


class model_CLS_CrossTransformer_IT_ALLPET(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'CrossTransformer_IT')
        if args.task == 'MCIC' or args.task == 'ADD':
            class_num = 2
        else:
            class_num = 3
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.expr_dir, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        self.optims = Munch()
        for net in self.nets.keys():
            self.optims[net] = torch.optim.AdamW(
                params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=args.weight_decay)
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_label = Compose([AsDiscrete(to_onehot=class_num)])
        # define evaluation metrics
        self.loss_epoch = CumulativeAverage()
        self.AUC = ROCAUCMetric(average='macro')

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def _reset_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = os.path.join(checkpoint_dir, str(self.fold))
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]

    def compute_CLS_loss_Tabualr(self, tabular, tabular_missing, label):
        # tabular embedding
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        tabular_feats = torch.flatten(tabular_feats, start_dim=1, end_dim=-1)
        # forward MLP
        logits = self.nets.Tabular_CLS(tabular_feats)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def compute_CLS_loss_MRI(self, MRI, tabular, tabular_missing, label):
        # forward CNN
        MRI_feats = self.nets.MRI(MRI)
        # Reshape to vectors
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        # tabular embedding
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        tabular_feats = torch.flatten(tabular_feats, start_dim=1, end_dim=-1)
        # forward cross transformer
        MRI_feats = self.nets.MRI_Trans(MRI_feats, tabular_feats)
        # Pooling
        gap = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveAvgPool1d(1), Rearrange('b d n -> b (d n)'))
        gmp = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveMaxPool1d(1), Rearrange('b d n -> b (d n)'))
        cls_avg = gap(MRI_feats)
        cls_max = gmp(MRI_feats)
        cls_token = torch.cat([cls_avg, cls_max], dim=1)
        # forward MLP
        logits = self.nets.MRI_CLS(cls_token)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def compute_CLS_loss_PET(self, MRI, PET, tabular, tabular_missing, label):
        # forward CNN
        MRI_feats = self.nets.MRI(MRI)
        PET_feats = self.nets.PET(PET)
        # Reshape to vectors
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        PET_feats = rearrange(PET_feats, 'b c h w d -> b (h w d) c')
        # tabular embedding
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        tabular_feats = torch.flatten(tabular_feats, start_dim=1, end_dim=-1)
        # forward cross transformer
        PET_feats = self.nets.PET_Trans(MRI_feats, PET_feats, tabular_feats)
        # Pooling
        gap = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveAvgPool1d(1), Rearrange('b d n -> b (d n)'))
        gmp = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveMaxPool1d(1), Rearrange('b d n -> b (d n)'))
        cls_avg = gap(PET_feats)
        cls_max = gmp(PET_feats)
        cls_token = torch.cat([cls_avg, cls_max], dim=1)
        # forward MLP
        logits = self.nets.PET_CLS(cls_token)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def start_train(self, train_loaders, val_loader):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_auc = 0.0
        best_epoch = 1
        self.logger.print_message('Start training...')
        start_time = time.time()

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            # initialize or reset training predictions
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []
            for i, data in enumerate(train_loaders):
                MRI = data['MRI'].to(self.device)
                PET = data['PET'].to(self.device)
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                DX = data['label'].to(self.device)
                logits, loss_CLS = self.compute_CLS_loss_PET(MRI, PET, tabular, tabular_missing, DX)
                # backward pass 2
                self._reset_grad()
                loss_CLS.backward()
                self.optims.T_Embedding.step()
                self.optims.Tabular_CLS.step()
                self.optims.MRI.step()
                self.optims.MRI_Trans.step()
                self.optims.MRI_CLS.step()
                self.optims.PET.step()
                self.optims.PET_Trans.step()
                self.optims.PET_CLS.step()
                # compute training metrics
                self.loss_epoch.append(loss_CLS)
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                label_real += [i for i in decollate_batch(data['label'])]
                label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log training metrics
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')

            # log training metrics
            loss_results = self.loss_epoch.aggregate()
            cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_train.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_train.TNR_Macro):.4f} '
                                          f'F1:{float(cm_train.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_train.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_train.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_train.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # validation iterations
            for name, model in self.nets.items():
                model.eval()

            # reset metrics
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    MRI = data['MRI'].to(self.device)
                    tabular = data['tabular'].to(self.device)
                    tabular_missing = data['tabular_missing'].to(self.device)
                    DX = data['label'].to(self.device)
                    PET = data['PET'].to(self.device)
                    logits, loss_CLS = self.compute_CLS_loss_PET(MRI, PET, tabular, tabular_missing, DX)
                    # compute validation metrics
                    self.loss_epoch.append(loss_CLS)
                    self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                             y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    label_real += [i for i in decollate_batch(data['label'])]
                    label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log validation metrics
            loss_results = self.loss_epoch.aggregate()
            cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_val.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_val.TNR_Macro):.4f} '
                                          f'F1:{float(cm_val.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_val.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_val.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_val.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # save best model according to the validation results
            if self.args.task == 'ADD':
                auc = self.AUC.aggregate()
                if auc >= best_auc:
                    self._save_checkpoint()
                    best_auc = auc
                    best_epoch = epoch + 1
            else:
                acc = float(cm_val.Overall_ACC)
                f1 = float(cm_val.F1_Macro)
                if acc >= best_acc and f1 >= best_f1:
                    self._save_checkpoint()
                    best_acc = acc
                    best_f1 = f1
                    best_epoch = epoch + 1

    def start_test(self, test_loader, phase):
        args = self.args
        # start test iterations
        self._load_checkpoint()
        content = []
        # reset training predictions
        self.AUC.reset()
        # test loop
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                MRI = data['MRI'].to(self.device)
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                DX = data['label'].to(self.device)
                PET = data['PET'].to(self.device)
                logits, loss_CLS = self.compute_CLS_loss_PET(MRI, PET, tabular, tabular_missing, DX)
                pred_prob = [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits)]
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                # write pred information
                label = data['label'][0].numpy()
                filename = data['filename'][0]
                case = {'filename': filename}
                case[args.task] = int(label)
                if args.task == 'COG':
                    case['COG_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['COG_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['COG_score_2'] = f'{pred_prob[0][2]:.8f}'
                    case['COG_pred'] = pred_prob[0].argmax()
                elif args.task == 'ADD':
                    case['ADD_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['ADD_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['ADD_pred'] = pred_prob[0].argmax()
                elif args.task == 'MCIC':
                    case['MCIC_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['MCIC_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['MCIC_pred'] = pred_prob[0].argmax()
                content.append(case)

        with open(os.path.join(self.checkpoint_dir, phase + '_eval.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
            writer.writeheader()
            for case in content:
                writer.writerow(case)

        # log validation metrics
        test_res_all = self.post_cls_eval(phase)
        self.logger.print_message(f'ACC:{test_res_all[0]:.4f} '
                                  f'MCC:{test_res_all[1]:.4f} '
                                  f'SEN:{test_res_all[2]:.4f} '
                                  f'SPE:{test_res_all[3]:.4f} '
                                  f'F1:{test_res_all[4]:.4f} '
                                  f'AUC:{self.AUC.aggregate():.4f}')
        return test_res_all + [self.AUC.aggregate()]

    def post_cls_eval(self, stage):
        csv_file = os.path.join(self.checkpoint_dir, stage + '_eval.csv')
        label, pred, pred_prob = utils.get_pd_gt(csv_file, self.args.task)
        cm = ConfusionMatrix(actual_vector=label, predict_vector=pred)
        if self.args == 'COG':
            acc = cm.Overall_ACC
            mcc = cm.Overall_MCC
            sen = cm.TPR_Macro
            spe = cm.TNR_Macro
            f1 = cm.F1_Macro
            auc = cm.AUC
            return [acc, mcc, sen, spe, f1, np.mean(list(auc.values()))]
        else:
            acc = cm.Overall_ACC
            mcc = list(cm.MCC.values())[1]
            sen = list(cm.TPR.values())[1]
            spe = list(cm.TNR.values())[1]
            f1 = list(cm.F1.values())[1]
            auc = list(cm.AUC.values())[1]
            return [acc, mcc, sen, spe, f1, auc]


class model_CLS_3MT(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, '3MT')
        if args.task == 'MCIC' or args.task == 'ADD':
            class_num = 2
        else:
            class_num = 3
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.expr_dir, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        self.optims = Munch()
        for net in self.nets.keys():
            self.optims[net] = torch.optim.AdamW(params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                                                 eps=1e-08, weight_decay=args.weight_decay)
        self.ckptios = [utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_label = Compose([AsDiscrete(to_onehot=class_num)])
        # define evaluation metrics
        self.loss_epoch = CumulativeAverage()
        self.loss_epoch_m1 = CumulativeAverage()
        self.loss_epoch_m2 = CumulativeAverage()
        self.loss_epoch_m3 = CumulativeAverage()
        self.AUC_m1 = ROCAUCMetric(average='macro')
        self.AUC_m2 = ROCAUCMetric(average='macro')
        self.AUC_m3 = ROCAUCMetric(average='macro')
        self.label_real = []
        self.label_pred_m1 = []
        self.label_pred_m2 = []
        self.label_pred_m3 = []
        # Pooling
        self.gap = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveAvgPool1d(1), Rearrange('b d n -> b (d n)'))
        self.gmp = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveMaxPool1d(1), Rearrange('b d n -> b (d n)'))

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def load_checkpoint(self):
        self._load_checkpoint()

    def _reset_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = os.path.join(checkpoint_dir, str(self.fold))
        self.ckptios = [utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]

    def _reset_metrics(self):
        self.loss_epoch.reset()
        self.AUC_m1.reset()
        self.AUC_m2.reset()
        self.AUC_m3.reset()
        self.label_real = []
        self.label_pred_m1 = []
        self.label_pred_m2 = []
        self.label_pred_m3 = []

    def _log_cm(self, cm):
        if self.args.task == 'COG':
            self.logger.print_message(f'ACC:{float(cm.Overall_ACC):.4f} '
                                      f'SEN:{float(cm.TPR_Macro):.4f} '
                                      f'SPE:{float(cm.TNR_Macro):.4f} '
                                      f'F1:{float(cm.F1_Macro):.4f} ')
            return [float(cm.Overall_ACC), float(cm.TPR_Macro), float(cm.TNR_Macro), float(cm.F1_Macro)]
        else:
            self.logger.print_message(f'ACC:{float(cm.Overall_ACC):.4f} '
                                      f'SEN:{float(list(cm.TPR.values())[1]):.4f} '
                                      f'SPE:{float(list(cm.TNR.values())[1]):.4f} '
                                      f'F1:{float(list(cm.F1.values())[1]):.4f} ')
            return [float(cm.Overall_ACC), float(list(cm.TPR.values())[1]), float(list(cm.TNR.values())[1]), float(list(cm.F1.values())[1])]

    def compute_loss_Tabualr(self, tabular, tabular_missing, label):
        query = repeat(self.nets.CLS_token(), '() h w -> c h w', c=tabular.shape[0])
        # tabular embedding
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        query = self.nets.T_FUSION(query, context=tabular_feats)
        # forward MLP
        logits = self.nets.T_CLS(rearrange(query, 'b c d -> b (c d)'))
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def compute_loss_MRI(self, MRI, tabular, tabular_missing, label):
        query = repeat(self.nets.CLS_token(), '() h w -> c h w', c=tabular.shape[0])
        # forward tabular path
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        query = self.nets.T_FUSION(query, context=tabular_feats)
        logits_t = self.nets.T_CLS(rearrange(query, 'b c d -> b (c d)'))
        loss_t = F.cross_entropy(logits_t, label)
        # forward MRI path
        MRI_feats = self.nets.MRI(MRI)
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        query = self.nets.MRI_FUSION(query, context=MRI_feats)
        logits_mri = self.nets.MRI_CLS(rearrange(query, 'b c d -> b (c d)'))
        loss_mri = F.cross_entropy(logits_mri, label)

        return logits_t, logits_mri, loss_t, loss_mri

    def compute_loss_PET(self, MRI, PET, tabular, tabular_missing, label):
        query = repeat(self.nets.CLS_token(), '() h w -> c h w', c=tabular.shape[0])
        # forward tabular path
        tabular_feats = self.nets.T_Embedding(tabular, tabular_missing)
        query = self.nets.T_FUSION(query, context=tabular_feats)
        logits_t = self.nets.T_CLS(rearrange(query, 'b c d -> b (c d)'))
        loss_t = F.cross_entropy(logits_t, label)
        # forward MRI path
        MRI_feats = self.nets.MRI(MRI)
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        query = self.nets.MRI_FUSION(query, context=MRI_feats)
        logits_mri = self.nets.MRI_CLS(rearrange(query, 'b c d -> b (c d)'))
        loss_mri = F.cross_entropy(logits_mri, label)
        # forward PET path
        PET_feats = self.nets.PET(PET)
        PET_feats = rearrange(PET_feats, 'b c h w d -> b (h w d) c')
        query = self.nets.PET_FUSION(query, context=PET_feats)
        logits_pet = self.nets.PET_CLS(rearrange(query, 'b c d -> b (c d)'))
        loss_pet = F.cross_entropy(logits_pet, label)

        return logits_t, logits_mri, logits_pet, loss_t, loss_mri, loss_pet

    def start_train(self, train_loaders, val_loader, stage):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_auc = 0.0
        best_epoch = 1
        self.logger.print_message(f'...Start training For Stage {stage}...')
        start_time = time.time()

        for epoch in range(0, args.epochs):
            for name, model in self.nets.items():
                model.train()
            self._reset_metrics()
            # Training
            for i, data in enumerate(train_loaders):
                # Load data
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                MRI = data['MRI'].to(self.device) if stage == 2 or stage == 3 else None
                PET = data['PET'].to(self.device) if stage == 3 else None
                DX = data['label'].to(self.device)
                # Forward
                if stage == 1:
                    logits_t, loss_t = self.compute_loss_Tabualr(tabular, tabular_missing, DX)
                    loss_CLS = loss_t
                elif stage == 2:
                    logits_t, logits_mri, loss_t, loss_mri = self.compute_loss_MRI(MRI, tabular, tabular_missing, DX)
                    loss_CLS = (loss_t + loss_mri) / 2
                elif stage == 3:
                    logits_t, logits_mri, logits_pet, loss_t, loss_mri, loss_pet = self.compute_loss_PET(MRI, PET, tabular, tabular_missing, DX)
                    loss_CLS = (loss_t + loss_mri + loss_pet) / 3
                else:
                    print('error on stage number')
                    break
                # Backward
                self._reset_grad()
                loss_CLS.backward()
                if stage == 1:
                    self.optims.T_Embedding.step()
                    self.optims.T_FUSION.step()
                    self.optims.T_CLS.step()
                if stage == 2:
                    # self.optims.T_Embedding.step()
                    # self.optims.T_FUSION.step()
                    # self.optims.T_CLS.step()
                    self.optims.MRI.step()
                    self.optims.MRI_FUSION.step()
                    self.optims.MRI_CLS.step()
                if stage == 3:
                    self.optims.PET.step()
                    self.optims.PET_FUSION.step()
                    self.optims.PET_CLS.step()
                # Metrics
                self.loss_epoch.append(loss_CLS)
                self.label_real += [i for i in decollate_batch(data['label'])]
                if stage == 1 or stage == 2 or stage == 3:
                    self.loss_epoch_m1.append(loss_t)
                    self.AUC_m1(y_pred=[self.post_pred(i) for i in decollate_batch(logits_t)],
                                y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    self.label_pred_m1 += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits_t)]
                if stage == 2 or stage == 3:
                    self.loss_epoch_m2.append(loss_mri)
                    self.AUC_m2(y_pred=[self.post_pred(i) for i in decollate_batch(logits_mri)],
                                y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    self.label_pred_m2 += [self.post_pred(i).detach().cpu().numpy().argmax() for i in
                                           decollate_batch(logits_mri)]
                if stage == 3:
                    self.loss_epoch_m3.append(loss_pet)
                    self.AUC_m3(y_pred=[self.post_pred(i) for i in decollate_batch(logits_pet)],
                                y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    self.label_pred_m3 += [self.post_pred(i).detach().cpu().numpy().argmax() for i in
                                           decollate_batch(logits_pet)]
            # Log metrics
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'*****Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}*****')
            loss_results = self.loss_epoch.aggregate()
            self.logger.print_message(f'Train             Overall Loss: {float(loss_results):.4f}')

            # Save checkpoints according to validation
            acc, f1 = self.start_validation(val_loader, stage)
            if acc >= best_acc and f1 >= best_f1:
                self._save_checkpoint()
                best_acc = acc
                best_f1 = f1
                best_epoch = epoch + 1

    def start_validation(self, val_loader, stage):
        for name, model in self.nets.items():
            model.eval()
        self._reset_metrics()

        # Validation
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                # fetch images and labels
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                MRI = data['MRI'].to(self.device) if stage == 2 or stage == 3 else None
                PET = data['PET'].to(self.device) if stage == 3 else None
                DX = data['label'].to(self.device)
                # Forward
                if stage == 1:
                    logits_t, loss_t = self.compute_loss_Tabualr(tabular, tabular_missing, DX)
                    loss_CLS = loss_t
                elif stage == 2:
                    logits_t, logits_mri, loss_t, loss_mri = self.compute_loss_MRI(MRI, tabular, tabular_missing, DX)
                    loss_CLS = (loss_t + loss_mri) / 2
                elif stage == 3:
                    logits_t, logits_mri, logits_pet, loss_t, loss_mri, loss_pet = self.compute_loss_PET(MRI, PET, tabular, tabular_missing, DX)
                    loss_CLS = (loss_t + loss_mri + loss_pet) / 3
                else:
                    print('error on stage number')
                    break
                # Metrics
                self.loss_epoch.append(loss_CLS)
                self.label_real += [i for i in decollate_batch(data['label'])]
                if stage == 1 or stage == 2 or stage == 3:
                    self.loss_epoch_m1.append(loss_t)
                    self.AUC_m1(y_pred=[self.post_pred(i) for i in decollate_batch(logits_t)],
                                y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    self.label_pred_m1 += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits_t)]
                if stage == 2 or stage == 3:
                    self.loss_epoch_m2.append(loss_mri)
                    self.AUC_m2(y_pred=[self.post_pred(i) for i in decollate_batch(logits_mri)],
                                y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    self.label_pred_m2 += [self.post_pred(i).detach().cpu().numpy().argmax() for i in
                                           decollate_batch(logits_mri)]
                if stage == 3:
                    self.loss_epoch_m3.append(loss_pet)
                    self.AUC_m3(y_pred=[self.post_pred(i) for i in decollate_batch(logits_pet)],
                                y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    self.label_pred_m3 += [self.post_pred(i).detach().cpu().numpy().argmax() for i in
                                           decollate_batch(logits_pet)]

        # log validation metrics
        loss_results = self.loss_epoch.aggregate()
        self.logger.print_message(f'Validation        Overall Loss: {float(loss_results):.4f}')
        if stage == 1 or stage == 2 or stage == 3:
            loss_results_m1 = self.loss_epoch_m1.aggregate()
            AUC_results_m1 = self.AUC_m1.aggregate()
            cm_m1 = ConfusionMatrix(actual_vector=self.label_real, predict_vector=self.label_pred_m1)
            self.logger.print_message(f'M1 Loss:{float(loss_results_m1):.4f} AUC: {float(AUC_results_m1):.4f} ', noenter=True)
            _ = self._log_cm(cm_m1)
        if stage == 2 or stage == 3:
            loss_results_m2 = self.loss_epoch_m2.aggregate()
            AUC_results_m2 = self.AUC_m2.aggregate()
            cm_m2 = ConfusionMatrix(actual_vector=self.label_real, predict_vector=self.label_pred_m2)
            self.logger.print_message(f'M2 Loss:{float(loss_results_m2):.4f} AUC: {float(AUC_results_m2):.4f} ', noenter=True)
            _ = self._log_cm(cm_m2)
        if stage == 3:
            loss_results_m3 = self.loss_epoch_m3.aggregate()
            AUC_results_m3 = self.AUC_m3.aggregate()
            cm_m3 = ConfusionMatrix(actual_vector=self.label_real, predict_vector=self.label_pred_m3)
            self.logger.print_message(f'M3 Loss:{float(loss_results_m3):.4f} AUC: {float(AUC_results_m3):.4f} ', noenter=True)
            _ = self._log_cm(cm_m3)

        # save best model according to the validation results
        if stage == 1:
            acc = float(cm_m1.Overall_ACC)
            f1 = float(cm_m1.F1_Macro)
        elif stage == 2:
            acc = (float(cm_m1.Overall_ACC) + float(cm_m2.Overall_ACC)) / 2
            f1 = (float(cm_m1.F1_Macro) + float(cm_m2.F1_Macro)) / 2
        else:
            acc = (float(cm_m1.Overall_ACC) + float(cm_m2.Overall_ACC) + float(cm_m3.Overall_ACC)) / 3
            f1 = (float(cm_m1.F1_Macro) + float(cm_m2.F1_Macro) + float(cm_m3.F1_Macro)) / 3

        return acc, f1

    def start_test(self, test_loader, phase):
        args = self.args
        # start test iterations
        self._load_checkpoint()
        for name, model in self.nets.items():
            model.eval()
        content = []
        label_pred_m1, label_pred_m2, label_pred_m3 = [], [], []
        prob_pred_m1, prob_pred_m2, prob_pred_m3 = [], [], []
        # reset training predictions
        self._reset_metrics()
        # test loop
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                tabular = data['tabular'].to(self.device)
                tabular_missing = data['tabular_missing'].to(self.device)
                MRI = data['MRI'].to(self.device)
                PET = data['PET'].to(self.device)
                DX = data['label'].to(self.device)
                # Forward
                if torch.equal(PET, torch.zeros_like(PET)):
                    logits_t, logits_mri, loss_t, loss_mri = self.compute_loss_MRI(MRI, tabular, tabular_missing, DX)
                    logits_pet = logits_mri
                    loss_pet = loss_mri
                else:
                    logits_t, logits_mri, logits_pet, loss_t, loss_mri, loss_pet = self.compute_loss_PET(MRI, PET, tabular, tabular_missing, DX)
                loss_CLS = (loss_t + loss_mri + loss_pet) / 3
                # Metrics
                self.loss_epoch.append(loss_CLS)
                self.label_real += [i for i in decollate_batch(data['label'])]
                # calculate metrics for stage 1
                self.loss_epoch_m1.append(loss_t)
                self.AUC_m1(y_pred=[self.post_pred(i) for i in decollate_batch(logits_t)],
                            y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                label_pred_m1 += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits_t)]
                prob_pred_m1 += [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits_t)]
                # calculate metrics for stage 2
                self.loss_epoch_m2.append(loss_mri)
                self.AUC_m2(y_pred=[self.post_pred(i) for i in decollate_batch(logits_mri)],
                            y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                label_pred_m2 += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits_mri)]
                prob_pred_m2 += [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits_mri)]
                # calculate metrics for stage 3
                self.loss_epoch_m3.append(loss_pet)
                self.AUC_m3(y_pred=[self.post_pred(i) for i in decollate_batch(logits_pet)],
                            y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                label_pred_m3 += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits_pet)]
                prob_pred_m3 += [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits_pet)]
                # write pred information
                label = data['label'][0].numpy()
                filename = data['filename'][0]
                case = {'filename': filename}
                case[args.task] = int(label)
                if args.task == 'COG':
                    case['COG_score_0_m1'] = f'{prob_pred_m1[i][0]:.8f}'
                    case['COG_score_1_m1'] = f'{prob_pred_m1[i][1]:.8f}'
                    case['COG_score_2_m1'] = f'{prob_pred_m1[i][2]:.8f}'
                    case['COG_pred_m1'] = prob_pred_m1[i].argmax()
                    case['COG_score_0_m2'] = f'{prob_pred_m2[i][0]:.8f}'
                    case['COG_score_1_m2'] = f'{prob_pred_m2[i][1]:.8f}'
                    case['COG_score_2_m2'] = f'{prob_pred_m2[i][2]:.8f}'
                    case['COG_pred_m2'] = prob_pred_m2[i].argmax()
                    case['COG_score_0_m3'] = f'{prob_pred_m3[i][0]:.8f}'
                    case['COG_score_1_m3'] = f'{prob_pred_m3[i][1]:.8f}'
                    case['COG_score_2_m3'] = f'{prob_pred_m3[i][2]:.8f}'
                    case['COG_pred_m3'] = prob_pred_m3[i].argmax()
                elif args.task == 'ADD':
                    case['ADD_score_0_m1'] = f'{prob_pred_m1[i][0]:.8f}'
                    case['ADD_score_1_m1'] = f'{prob_pred_m1[i][1]:.8f}'
                    case['ADD_pred_m1'] = prob_pred_m1[i].argmax()
                    case['ADD_score_0_m2'] = f'{prob_pred_m2[i][0]:.8f}'
                    case['ADD_score_1_m2'] = f'{prob_pred_m2[i][1]:.8f}'
                    case['ADD_pred_m2'] = prob_pred_m2[i].argmax()
                    case['ADD_score_0_m3'] = f'{prob_pred_m3[i][0]:.8f}'
                    case['ADD_score_1_m3'] = f'{prob_pred_m3[i][1]:.8f}'
                    case['ADD_pred_m3'] = prob_pred_m3[i].argmax()
                elif args.task == 'MCIC':
                    case['MCIC_score_0_m1'] = f'{prob_pred_m1[i][0]:.8f}'
                    case['MCIC_score_1_m1'] = f'{prob_pred_m1[i][1]:.8f}'
                    case['MCIC_pred_m1'] = prob_pred_m1[i].argmax()
                    case['MCIC_score_0_m2'] = f'{prob_pred_m2[i][0]:.8f}'
                    case['MCIC_score_1_m2'] = f'{prob_pred_m2[i][1]:.8f}'
                    case['MCIC_pred_m2'] = prob_pred_m2[i].argmax()
                    case['MCIC_score_0_m3'] = f'{prob_pred_m3[i][0]:.8f}'
                    case['MCIC_score_1_m3'] = f'{prob_pred_m3[i][1]:.8f}'
                    case['MCIC_pred_m3'] = prob_pred_m3[i].argmax()
                content.append(case)

        with open(os.path.join(self.checkpoint_dir, phase + '_eval.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
            writer.writeheader()
            for case in content:
                writer.writerow(case)

        # log test metrics
        # stage 1
        loss_results = self.loss_epoch.aggregate()
        self.logger.print_message(f'Test        Overall Loss: {float(loss_results):.4f}')
        loss_results_m1 = self.loss_epoch_m1.aggregate()
        AUC_results_m1 = self.AUC_m1.aggregate()
        cm_m1 = ConfusionMatrix(actual_vector=self.label_real, predict_vector=label_pred_m1)
        self.logger.print_message(f'M1 Loss:{float(loss_results_m1):.4f} AUC: {float(AUC_results_m1):.4f} ', noenter=True)
        metrics_S1 = self._log_cm(cm_m1)
        metrics_S1.append(float(AUC_results_m1))
        # stage 2
        loss_results_m2 = self.loss_epoch_m2.aggregate()
        AUC_results_m2 = self.AUC_m2.aggregate()
        cm_m2 = ConfusionMatrix(actual_vector=self.label_real, predict_vector=label_pred_m2)
        self.logger.print_message(f'M2 Loss:{float(loss_results_m2):.4f} AUC: {float(AUC_results_m2):.4f} ', noenter=True)
        metrics_S2 = self._log_cm(cm_m2)
        metrics_S2.append(float(AUC_results_m2))
        # stage 3
        loss_results_m3 = self.loss_epoch_m3.aggregate()
        AUC_results_m3 = self.AUC_m3.aggregate()
        cm_m3 = ConfusionMatrix(actual_vector=self.label_real, predict_vector=label_pred_m3)
        self.logger.print_message(f'M3 Loss:{float(loss_results_m3):.4f} AUC: {float(AUC_results_m3):.4f} ', noenter=True)
        metrics_S3 = self._log_cm(cm_m3)
        metrics_S3.append(float(AUC_results_m3))

        return metrics_S1, metrics_S2, metrics_S3


class model_CLS_DAFT(nn.Module):
    def __init__(self, args, fold, model_name='DAFT'):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, model_name)
        if args.task == 'MCIC' or args.task == 'ADD' or args.task == 'MCINC':
            class_num = 2
        else:
            class_num = 3
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.expr_dir, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        self.optims = Munch()
        for net in self.nets.keys():
            self.optims[net] = torch.optim.AdamW(
                params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=args.weight_decay)
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_label = Compose([AsDiscrete(to_onehot=class_num)])
        # define evaluation metrics
        self.loss_epoch = CumulativeAverage()
        self.AUC = ROCAUCMetric(average='macro')

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def load_checkpoint(self):
        self._load_checkpoint()

    def _reset_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = os.path.join(checkpoint_dir, str(self.fold))
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]

    def compute_CLS_loss(self, IMG, tabular, label):
        # forward CNN
        logits = self.nets.DAFT(IMG, tabular)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def start_train(self, train_loaders, val_loader, optuna=None):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_auc = 0.0
        best_epoch = 1
        self.logger.print_message('Start training...')
        start_time = time.time()

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            # initialize or reset training predictions
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                IMG = data['MRI'].to(self.device)
                tabular = data['tabular'].to(self.device)
                DX = data['label'].to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(IMG, tabular, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.DAFT.step()
                # compute training metrics
                self.loss_epoch.append(loss_CLS)
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                label_real += [i for i in decollate_batch(data['label'])]
                label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log training status
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')

            # log training metrics
            loss_results = self.loss_epoch.aggregate()
            cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_train.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_train.TNR_Macro):.4f} '
                                          f'F1:{float(cm_train.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f"Trainng    - Loss:{float(loss_results):.4f} "
                                          f"ACC:{float(cm_train.Overall_ACC):.4f} "
                                          f"SEN:{float(list(cm_train.TPR.values())[1]):.4f} "
                                          f"SPE:{float(list(cm_train.TNR.values())[1]):.4f} "
                                          f"F1:{float(list(cm_train.F1.values())[1]):.4f} "
                                          f"AUC:{self.AUC.aggregate():.4f}")

            # validation iterations
            for name, model in self.nets.items():
                model.eval()

            # reset metrics
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    IMG = data['MRI'].to(self.device)
                    tabular = data['tabular'].to(self.device)
                    DX = data['label'].to(self.device)
                    logits, loss_CLS = self.compute_CLS_loss(IMG, tabular, DX)

                    # compute training metrics
                    self.loss_epoch.append(loss_CLS)
                    self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                             y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    label_real += [i for i in decollate_batch(data['label'])]
                    label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log validation metrics
            loss_results = self.loss_epoch.aggregate()
            cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_val.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_val.TNR_Macro):.4f} '
                                          f'F1:{float(cm_val.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_val.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_val.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_val.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # save best model according to the validation results
            if self.args.task == 'ADD':
                auc = self.AUC.aggregate()
                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch + 1
                    self._save_checkpoint()
            else:
                acc = float(cm_val.Overall_ACC)
                f1 = float(cm_val.F1_Macro)
                if acc > best_acc and f1 >= best_f1:
                    best_acc = acc
                    best_f1 = f1
                    best_epoch = epoch + 1
                    if optuna:
                        optuna.report(f1, epoch)
                        self._save_checkpoint(optuna.number)
                    else:
                        self._save_checkpoint()

        return best_f1

    def start_test(self, test_loader, phase, trial=None):
        args = self.args
        # start test iterations
        if trial:
            self._load_checkpoint(trial_number=trial)
        else:
            self._load_checkpoint()
        content = []
        # reset training predictions
        self.AUC.reset()
        # test loop
        for name, model in self.nets.items():
            model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                IMG = data['MRI'].to(self.device)
                tabular = data['tabular'].to(self.device)
                DX = data['label'].to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(IMG, tabular, DX)
                pred_prob = [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits)]
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                # write pred information
                label = data['label'][0].numpy()
                filename = data['filename'][0]
                case = {'filename': filename}
                case[args.task] = int(label)
                if args.task == 'COG':
                    case['COG_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['COG_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['COG_score_2'] = f'{pred_prob[0][2]:.8f}'
                    case['COG_pred'] = pred_prob[0].argmax()
                elif args.task == 'ADD':
                    case['ADD_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['ADD_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['ADD_pred'] = pred_prob[0].argmax()
                elif args.task == 'MCIC':
                    case['MCIC_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['MCIC_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['MCIC_pred'] = pred_prob[0].argmax()
                content.append(case)

        with open(os.path.join(self.checkpoint_dir, phase + '_eval.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
            writer.writeheader()
            for case in content:
                writer.writerow(case)

        # log validation metrics
        test_res_all = self.post_cls_eval(phase)
        self.logger.print_message(f'ACC:{test_res_all[0]:.4f} '
                                  f'MCC:{test_res_all[1]:.4f} '
                                  f'SEN:{test_res_all[2]:.4f} '
                                  f'SPE:{test_res_all[3]:.4f} '
                                  f'F1:{test_res_all[4]:.4f} '
                                  f'AUC:{self.AUC.aggregate():.4f}')
        return test_res_all + [self.AUC.aggregate()]

    def post_cls_eval(self, stage):
        csv_file = os.path.join(self.checkpoint_dir, stage + '_eval.csv')
        label, pred, pred_prob = utils.get_pd_gt(csv_file, self.args.task)
        cm = ConfusionMatrix(actual_vector=label, predict_vector=pred)
        if self.args.task == 'COG':
            acc = cm.Overall_ACC
            mcc = cm.Overall_MCC
            sen = cm.TPR_Macro
            spe = cm.TNR_Macro
            f1 = cm.F1_Macro
            auc = cm.AUC
            return [acc, mcc, sen, spe, f1, np.mean(list(auc.values()))]
        else:
            acc = cm.Overall_ACC
            mcc = list(cm.MCC.values())[1]
            sen = list(cm.TPR.values())[1]
            spe = list(cm.TNR.values())[1]
            f1 = list(cm.F1.values())[1]
            auc = list(cm.AUC.values())[1]
            return [acc, mcc, sen, spe, f1, auc]

    def forward(self, IMG):
        feats = self.nets.CNN(IMG)
        # Average Pool
        feats = F.adaptive_max_pool3d(feats, 1).view(IMG.shape[0], -1)
        # forward MLP
        logits = self.nets.CLS(feats)
        return torch.nn.functional.softmax(logits)


class model_CLS_ResNet(nn.Module):
    def __init__(self, args, fold, modality):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'ResNet')
        if args.task == 'MCIC' or args.task == 'ADD' or args.task == 'MCINC':
            class_num = 2
        else:
            class_num = 3
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.expr_dir, str(fold))
        self.modality = modality
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        self.optims = Munch()
        for net in self.nets.keys():
            self.optims[net] = torch.optim.AdamW(
                params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=args.weight_decay)
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_label = Compose([AsDiscrete(to_onehot=class_num)])
        # define evaluation metrics
        self.loss_epoch = CumulativeAverage()
        self.AUC = ROCAUCMetric(average='macro')

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    # def _save_checkpoint(self, trial_number=None):
    #     param_dict = {}
    #     for name, network in self.named_children():
    #         param_dict[name] = network.state_dict()
    #     if trial_number:
    #         torch.save(param_dict, os.path.join(self.checkpoint_dir, f'best_nets_{trial_number}.pt'))
    #     else:
    #         torch.save(param_dict, os.path.join(self.checkpoint_dir, 'best_nets.pt'))
    #
    # def _load_checkpoint(self, trial_number=None):
    #     if trial_number:
    #         param_dict = torch.load(os.path.join(self.checkpoint_dir, f'best_nets_{trial_number}.pt'))
    #     else:
    #         param_dict = torch.load(os.path.join(self.checkpoint_dir, 'best_nets.pt'))
    #     for name, params in param_dict.items():
    #         self.nets[name].load_state_dict(params)

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def load_checkpoint(self):
        self._load_checkpoint()

    def _reset_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = os.path.join(checkpoint_dir, str(self.fold))
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]

    def compute_CLS_loss(self, IMG, label):
        # forward CNN
        feats = self.nets.CNN(IMG)
        # Average Pool
        feats = F.adaptive_avg_pool3d(feats, 1).view(IMG.shape[0], -1)
        # forward MLP
        logits = self.nets.CLS(feats)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def start_train(self, train_loaders, val_loader, optuna=None):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_auc = 0.0
        best_epoch = 1
        self.logger.print_message('Start training...')
        start_time = time.time()

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            # initialize or reset training predictions
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                if self.modality == 'MRI':
                    IMG = data['MRI'].to(self.device)
                    DX = data['label'].to(self.device)
                elif self.modality == 'PET':
                    IMG = data['PET'].to(self.device)
                    DX = data['label'].to(self.device)
                else:
                    IMG, DX = None, None
                logits, loss_CLS = self.compute_CLS_loss(IMG, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.CNN.step()
                self.optims.CLS.step()
                # compute training metrics
                self.loss_epoch.append(loss_CLS)
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                label_real += [i for i in decollate_batch(data['label'])]
                label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log training status
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')

            # log training metrics
            loss_results = self.loss_epoch.aggregate()
            cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_train.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_train.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_train.TNR_Macro):.4f} '
                                          f'F1:{float(cm_train.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f"Trainng    - Loss:{float(loss_results):.4f} "
                                          f"ACC:{float(cm_train.Overall_ACC):.4f} "
                                          f"SEN:{float(list(cm_train.TPR.values())[1]):.4f} "
                                          f"SPE:{float(list(cm_train.TNR.values())[1]):.4f} "
                                          f"F1:{float(list(cm_train.F1.values())[1]):.4f} "
                                          f"AUC:{self.AUC.aggregate():.4f}")

            # validation iterations
            for name, model in self.nets.items():
                model.eval()

            # reset metrics
            self.loss_epoch.reset()
            self.AUC.reset()
            label_pred = []
            label_real = []

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    if self.modality == 'MRI':
                        IMG = data['MRI'].to(self.device)
                        DX = data['label'].to(self.device)
                    elif self.modality == 'PET':
                        IMG = data['PET'].to(self.device)
                        DX = data['label'].to(self.device)
                    else:
                        IMG, DX = None, None
                    logits, loss_CLS = self.compute_CLS_loss(IMG, DX)
                    # compute training metrics
                    self.loss_epoch.append(loss_CLS)
                    self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                             y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                    label_real += [i for i in decollate_batch(data['label'])]
                    label_pred += [self.post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(logits)]

            # log validation metrics
            loss_results = self.loss_epoch.aggregate()
            cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
            if self.args.task == 'COG':
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(cm_val.TPR_Macro):.4f} '
                                          f'SPE:{float(cm_val.TNR_Macro):.4f} '
                                          f'F1:{float(cm_val.F1_Macro):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')
            else:
                self.logger.print_message(f'Validation    - Loss:{float(loss_results):.4f} '
                                          f'ACC:{float(cm_val.Overall_ACC):.4f} '
                                          f'SEN:{float(list(cm_val.TPR.values())[1]):.4f} '
                                          f'SPE:{float(list(cm_val.TNR.values())[1]):.4f} '
                                          f'F1:{float(list(cm_val.F1.values())[1]):.4f} '
                                          f'AUC:{self.AUC.aggregate():.4f}')

            # save best model according to the validation results
            if self.args.task == 'ADD':
                auc = self.AUC.aggregate()
                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch + 1
                    self._save_checkpoint()
            else:
                acc = float(cm_val.Overall_ACC)
                f1 = float(cm_val.F1_Macro)
                if acc > best_acc and f1 >= best_f1:
                    best_acc = acc
                    best_f1 = f1
                    best_epoch = epoch + 1
                    if optuna:
                        optuna.report(f1, epoch)
                        self._save_checkpoint(optuna.number)
                    else:
                        self._save_checkpoint()

        return best_f1

    def start_test(self, test_loader, phase, trial=None):
        args = self.args
        # start test iterations
        if trial:
            self._load_checkpoint(trial_number=trial)
        else:
            self._load_checkpoint()
        content = []
        # reset training predictions
        self.AUC.reset()
        # test loop
        for name, model in self.nets.items():
            model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                if self.modality == 'MRI':
                    IMG = data['MRI'].to(self.device)
                    DX = data['label'].to(self.device)
                elif self.modality == 'PET':
                    IMG = data['PET'].to(self.device)
                    DX = data['label'].to(self.device)
                else:
                    IMG, DX = None, None
                logits, loss_CLS = self.compute_CLS_loss(IMG, DX)
                pred_prob = [self.post_pred(i).detach().cpu().numpy() for i in decollate_batch(logits)]
                self.AUC(y_pred=[self.post_pred(i) for i in decollate_batch(logits)],
                         y=[self.post_label(i) for i in decollate_batch(DX, detach=False)])
                # write pred information
                label = data['label'][0].numpy()
                filename = data['filename'][0]
                case = {'filename': filename}
                case[args.task] = int(label)
                if args.task == 'COG':
                    case['COG_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['COG_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['COG_score_2'] = f'{pred_prob[0][2]:.8f}'
                    case['COG_pred'] = pred_prob[0].argmax()
                elif args.task == 'ADD':
                    case['ADD_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['ADD_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['ADD_pred'] = pred_prob[0].argmax()
                elif args.task == 'MCIC':
                    case['MCIC_score_0'] = f'{pred_prob[0][0]:.8f}'
                    case['MCIC_score_1'] = f'{pred_prob[0][1]:.8f}'
                    case['MCIC_pred'] = pred_prob[0].argmax()
                content.append(case)

        with open(os.path.join(self.checkpoint_dir, phase + '_eval.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
            writer.writeheader()
            for case in content:
                writer.writerow(case)

        # log validation metrics
        test_res_all = self.post_cls_eval(phase)
        self.logger.print_message(f'ACC:{test_res_all[0]:.4f} '
                                  f'MCC:{test_res_all[1]:.4f} '
                                  f'SEN:{test_res_all[2]:.4f} '
                                  f'SPE:{test_res_all[3]:.4f} '
                                  f'F1:{test_res_all[4]:.4f} '
                                  f'AUC:{self.AUC.aggregate():.4f}')
        return test_res_all + [self.AUC.aggregate()]

    def post_cls_eval(self, stage):
        csv_file = os.path.join(self.checkpoint_dir, stage + '_eval.csv')
        label, pred, pred_prob = utils.get_pd_gt(csv_file, self.args.task)
        cm = ConfusionMatrix(actual_vector=label, predict_vector=pred)
        if self.args.task == 'COG':
            acc = cm.Overall_ACC
            mcc = cm.Overall_MCC
            sen = cm.TPR_Macro
            spe = cm.TNR_Macro
            f1 = cm.F1_Macro
            auc = cm.AUC
            return [acc, mcc, sen, spe, f1, np.mean(list(auc.values()))]
        else:
            acc = cm.Overall_ACC
            mcc = list(cm.MCC.values())[1]
            sen = list(cm.TPR.values())[1]
            spe = list(cm.TNR.values())[1]
            f1 = list(cm.F1.values())[1]
            auc = list(cm.AUC.values())[1]
            return [acc, mcc, sen, spe, f1, auc]

    def shap_mid(self, background_dataloader, val_dataloader):
        shap_dir = os.path.join(self.checkpoint_dir,'shap')
        if not os.path.exists(shap_dir): # create the folder for storing shap heatmaps
            os.mkdir(shap_dir)
        self._load_checkpoint()
        for name, model in self.nets.items():
            model.eval()

        # initialize shap explainer
        background = []
        for i, data in enumerate(background_dataloader):
            background.append(data['MRI'])
        background = torch.cat(background, 0).to(self.device)
        e = shap.DeepExplainer(self.CNN, background)
        del background
        # loop over val_dataloder
        for i, data in enumerate(val_dataloader):
            MRI = data['MRI'].to(self.device)
            name = data['filename'][0]
            shap_value = e.shap_values(MRI)
            print('shap value shape is ', shap_value.shape)
            np.save(os.path.join(shap_dir, f'shap_{self.args.task}_{name}'), shap_value[1].squeeze().sum(0))

    def forward(self, IMG):
        feats = self.nets.CNN(IMG)
        # Average Pool
        feats = F.adaptive_max_pool3d(feats, 1).view(IMG.shape[0], -1)
        # forward MLP
        logits = self.nets.CLS(feats)
        return torch.nn.functional.softmax(logits)