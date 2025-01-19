import os
import utils
import matplotlib

matplotlib.use('Agg')
from matplotlib import rc
from pycm import ConfusionMatrix, ROCCurve
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import numpy as np
import glob

rc('axes', linewidth=1.5)
rc('font', weight='bold', size=15)

def checknone(val):
    if val == 'None':
        return 0.0
    else:
        return val


def get_metrics_by_cm(cm, task, auc, bacc):
    if task == 'COG':
        # acc = balanced_accuracy_score(bacc['y_true'], bacc['y_pred'])
        acc = cm.Overall_ACC
        sen = cm.TPR_Macro
        spe = cm.TNR_Macro
        f1 = cm.F1_Macro
        mcc = cm.Overall_MCC
        auc = roc_auc_score(np.eye(3)[np.array(auc['label'])], np.array(auc['prob']), multi_class='ovo')
    else:
        # acc = balanced_accuracy_score(bacc['y_true'], bacc['y_pred'])
        acc = cm.Overall_ACC
        try:
            sen = cm.TPR[1]
        except KeyError:
            sen = 0.0
        try:
            spe = cm.TNR[1]
        except KeyError:
            spe = 0.0
        try:
            f1 = cm.F1[1]
        except KeyError:
            f1 = 0.0
        try:
            mcc = cm.MCC[1]
        except:
            mcc = 0.0
        try:
            auc = roc_auc_score(np.eye(2)[np.array(auc['label'])], np.array(auc['prob']), multi_class='ovo')
        except:
            auc = 0.0

    return [checknone(acc), checknone(sen), checknone(spe), checknone(f1), checknone(mcc), checknone(auc)]


def summerize_on_res_folds(res_folds):
    for cohort, res in res_folds.items():
        print(f'{cohort}')
        results = np.array(res)
        res_mean = np.mean(results, axis=0) * 100
        res_std = np.std(results, axis=0) * 100
        print(f'acc: {res_mean[0]:.1f} +- {res_std[0]:.1f}\n'
              f'sen: {res_mean[1]:.1f} +- {res_std[1]:.1f}\n'
              f'spe: {res_mean[2]:.1f} +- {res_std[2]:.1f}\n'
              f'f1: {res_mean[3]:.1f} +- {res_std[3]:.1f}\n'
              f'mcc: {res_mean[4]:.1f} +- {res_std[4]:.1f}\n'
              f'auc: {res_mean[5]:.1f} +- {res_std[5]:.1f}\n')


def summerize_on_res_folds2(res_folds):
    print('Overall')
    results = np.array(res_folds)
    for res_fold in results:
        print(res_fold)
    res_mean = np.mean(results, axis=0) * 100
    res_std = np.std(results, axis=0) * 100
    print(f'acc: {res_mean[0]:.1f} +- {res_std[0]:.1f}\n'
          f'sen: {res_mean[1]:.1f} +- {res_std[1]:.1f}\n'
          f'spe: {res_mean[2]:.1f} +- {res_std[2]:.1f}\n'
          f'f1: {res_mean[3]:.1f} +- {res_std[3]:.1f}\n'
          f'mcc: {res_mean[4]:.1f} +- {res_std[4]:.1f}\n'
          f'auc: {res_mean[5]:.1f} +- {res_std[5]:.1f}\n')


if __name__ == '__main__':
    root_path = './checkpoints/37features'
    eval_task = ['ADD']

    # eval on different cohorts respectively
    task_cohort_mapping = {'COG': ['ADNI', 'NACC', 'OAS'], 'MCIC': ['ADNI', 'NACC'], 'ADD': ['ADNI', 'NACC', 'OAS']}
    # task_cohort_mapping = {'COG': ['Huashan']}
    for task in eval_task:
        ckpt_dir_lists = glob.glob(os.path.join(root_path, task + '*'))
        for ckpt_dir in ckpt_dir_lists:
            cohorts_list = task_cohort_mapping[task]
            res_folds = {cohort: [] for cohort in cohorts_list}
            res_folds_overall = []
            for fold_idx in range(5):
                csv_path = os.path.join(ckpt_dir, str(fold_idx), 'test_eval.csv')
                label, pred, pred_prob = utils.get_pd_gt_bycohorts(csv_path, task)
                label_overall, pred_overall, pred_prob_overall = [], [], []
                for cohort in cohorts_list:
                    label_overall += label[cohort]
                    pred_overall += pred[cohort]
                    pred_prob_overall += pred_prob[cohort]
                    auc = {'label': label[cohort], 'prob': pred_prob[cohort]}
                    bacc = {'y_true': label[cohort], 'y_pred': pred[cohort]}
                    cm = ConfusionMatrix(actual_vector=label[cohort], predict_vector=pred[cohort], digit=4)
                    cls_metrics = get_metrics_by_cm(cm, task, auc, bacc)
                    res_folds[cohort].append(cls_metrics)
                auc_overall = {'label': label_overall, 'prob': pred_prob_overall}
                bacc_overall = {'y_true': label_overall, 'y_pred': pred_overall}
                cm_overall = ConfusionMatrix(actual_vector=label_overall, predict_vector=pred_overall, digit=4)
                res_folds_overall.append(get_metrics_by_cm(cm_overall, task, auc_overall, bacc_overall))
            print(f'*************** Evaluate {ckpt_dir} ***************')
            summerize_on_res_folds(res_folds)
            summerize_on_res_folds2(res_folds_overall)

    # # eval on whloe dataset
    # for task in eval_task:
    #     ckpt_dir_lists = glob.glob(os.path.join(root_path, '*' + task))
    #     for ckpt_dir in ckpt_dir_lists:
    #         res_folds = []
    #         for fold_idx in range(5):
    #             csv_path = os.path.join(ckpt_dir, str(fold_idx), 'test_eval.csv')
    #             label, pred, pred_prob = utils.get_pd_gt(csv_path, task)
    #             cm = ConfusionMatrix(actual_vector=label, predict_vector=pred, digit=4)
    #             cls_metrics = get_metrics_by_cm(cm, task)
    #             res_folds.append(cls_metrics)
    #         print(f'*************** Evaluate {ckpt_dir} ***************')
    #         summerize_on_res_folds2(res_folds)