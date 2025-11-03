import logging
from typing import Callable, List

import torch
import torch.nn as nn
from .predict import predict
from chemprop.data import MoleculeDataset, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
def calc_rocauc_score(labels, preds, valid):
    """compute ROC-AUC and averaged across tasks"""
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
        preds = preds.reshape(-1, 1)

    rocauc_list = []
    acc_list = []
    for i in range(labels.shape[1]):
        c_valid = valid[:, i].astype("bool")
        c_label, c_pred = labels[c_valid, i], preds[c_valid, i]
        #AUC is only defined when there is at least one positive data.
        if len(np.unique(c_label)) == 2:
            rocauc_list.append(roc_auc_score(c_label, c_pred))
            acc_list.append(accuracy_score(c_label.astype('int32'), np.round(c_pred).astype('int32')))
        else:
            print(i)

    print('Valid ratio: %s' % (np.mean(valid)))
    print('Task evaluated: %s/%s' % (len(rocauc_list), labels.shape[1]))
    if len(rocauc_list) == 0:
        raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")

    return sum(rocauc_list) / len(rocauc_list), sum(acc_list) / len(acc_list)


def evaluate(model: nn.Module,
             data: MoleculeDataset,
             scaler: StandardScaler = None,
             analyze_dir = None) -> List[float]:
    preds, uq_sum, label, sum_atoms, iter_uq, a_scope, batch,all_atoms_uq_list = predict(
        model=model,
        datasets=data,
        scaler=scaler,
        analyze_dir=analyze_dir
    )

    valid = np.where(label == 0.5, 0., 1.)
    auc, acc = calc_rocauc_score(label, preds, valid)

    results = uq_sum.sum() / sum_atoms
    return results, auc, acc, iter_uq,a_scope,batch,all_atoms_uq_list

