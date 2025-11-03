import datetime
from argparse import Namespace
import csv
from logging import Logger
import os
from pprint import pformat
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from torch.utils.data import DataLoader
import torch


import pickle
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate
from .train import train
from chemprop.data import StandardScaler, MoleculeDataset
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import build_model
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, \
    makedirs, save_checkpoint, transfer_learning_check



def collate_funtion(x):
    x = MoleculeDataset(x)
    smiles_batch, features_batch, target_batch = x.smiles(), x.features(), x.targets()
    targets = torch.tensor(target_batch)
    targets = targets.cuda()
    return smiles_batch, features_batch, targets


def run_training(args: Namespace):
    # Get data
    date_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_dir = f'{args.fold_num}_{date_time}'
    os.makedirs(save_dir+'/测试集所有原子的迭代UQ')
    print('\n Loading data.....')
    args.task_names = get_task_names(args.data_path)
    if args.fold_num in ['bbbp','clinTox','tox21','hiv']:
        train_data = get_data(path=f'datasets/{args.fold_num}_train.csv', args=args)
        test_data = get_data(path=f'datasets/{args.fold_num}_test.csv', args=args)
        val_data = get_data(path=f'datasets/{args.fold_num}_val.csv', args=args)
    else:
        data = get_data(path=args.data_path, args=args)
        train_data, test_data,val_data = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.1, 0.1),
                                           seed=args.seed, args=args)

    args.num_tasks = train_data.num_tasks()
    args.features_size = train_data.features_size()
    # args.num_tasks = data.num_tasks()
    # args.features_size = data.features_size()
    print(f'Number of tasks = {args.num_tasks}')

    # Split data
    print(f'Splitting data with seed {args.seed}')


    args.train_data_size = len(train_data)

    print(f'Total size = {len(train_data)+len(test_data) +len(val_data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    scaler = None

    model = build_model(args)

    print(f'Number of parameters = {param_count(model):,}')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cuda:
        print('Moving model to cuda')
        model = model.to(device)

    # Optimizers
    optimizer = build_optimizer(model, args)

    # Learning rate schedulers
    print(f'train_data_size: {args.train_data_size}, args.batch_size: {args.batch_size}')
    scheduler = build_lr_scheduler(optimizer, args)

    loss = []
    list_out = []
    iter_uq_list = []
    atom_uq_list = []
    sum_uq_list = []


    loss_func = torch.nn.BCELoss(reduction='none')

    train_dataloader = DataLoader(train_data.data, args.batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_funtion)
    test_dataloader = DataLoader(test_data.data, args.batch_size, shuffle=False, drop_last=True,
                                 collate_fn=collate_funtion)
    val_dataloader = DataLoader(val_data.data, args.batch_size, shuffle=False, drop_last=True,
                                 collate_fn=collate_funtion)
    max_auc = 0
    max_acc = 0
    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')

        n_iter, loss_list, uq = train(
            model=model,
            data=train_dataloader,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args)
        loss.append(loss_list)

        if isinstance(scheduler, ExponentialLR):
            scheduler.step()
        val_uq, val_auc, val_acc, iter_uq, a_scope, batch, _ = evaluate(
            model=model,
            data=val_dataloader,
            scaler=scaler,
            analyze_dir = None
        )
        atom_uq_list.append(torch.stack(iter_uq).narrow(1, a_scope[0][0], a_scope[0][1]).sum(-1).detach().cpu().numpy())
        iter_uq = torch.stack(iter_uq).mean(0)
        sum_uq_list.append(iter_uq.detach().cpu().numpy().sum())
        list_out.append([val_auc, val_acc])
        mol_uq_list = []
        for i in [0, 1, 3, 8]:
            mol_uq_list.append(iter_uq.narrow(0, a_scope[i][0], a_scope[i][1]).detach().cpu().numpy().sum())
        mol_uq_list.append(np.array(mol_uq_list).mean())
        iter_uq_list.append(mol_uq_list)
        if max_auc < val_auc:
            torch.save(model, f'{save_dir}/model.pth')
            max_auc = val_auc / 1.0
            max_acc = val_acc / 1.0
            print('model saved!')

        # print(
        #     f'Validation uq = {val_uq:.6f} | Validation rmse = {val_rmses:.6f} | Validation mae = {val_maes:.6f}')
        print(
            f'Validation uq = {val_uq:.6f} | Validation auc = {val_auc:.6f} | Validation acc = {val_acc:.6f}')
    model = torch.load(f'{save_dir}/model.pth')
    test_uq, test_auc, test_acc, _, _, _,all_atoms_uq_list = evaluate(
        model=model,
        data=test_dataloader,
        scaler=scaler,
        analyze_dir=save_dir
    )
    print(
        f'Test uq = {test_uq:.6f} | Test auc = {test_auc:.6f} | Test acc = {test_acc:.6f}')

    plt.plot(range(iter_uq_list.__len__()), iter_uq_list,
             label=[batch[0], batch[1], batch[3], batch[8], 'Mean UQ of five molecules'])
    plt.legend()
    plt.xlabel(u'epoch')
    plt.title(args.fold_num + ' molecules iter_uq')
    plt.show()
    np.savetxt(f'{save_dir}/五个分子的表征UQ.csv', iter_uq_list, delimiter=',')

    plt.plot(range(sum_uq_list.__len__()), sum_uq_list)
    plt.legend()
    plt.xlabel(u'epoch')
    plt.title(args.fold_num + ' Test total_iter_uq')
    plt.show()
    np.savetxt(f'{save_dir}/单个batch的总表征UQ.csv', sum_uq_list, delimiter=',')
    atom_uq = np.array(atom_uq_list)[:, :, 0]
    single_atom_uq = []
    for i in range(10):
        single_atom_uq.append(np.convolve(atom_uq[:, i], np.ones(3) / 3, mode='valid'))
    single_atom_uq = np.array(single_atom_uq)
    plt.plot(single_atom_uq.T, label=['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5', 'Iteration 6', 'Iteration 7', 'Iteration 8', 'Iteration 9', 'Iteration 10'])
    plt.legend()
    plt.xlabel(u'Epochs')
    plt.ylabel(u'UQ')
    plt.title(args.fold_num + ' Iterative atom uq')
    plt.show()
    np.savetxt(f'{save_dir}/原子的迭代表征UQ.csv', single_atom_uq, delimiter=',')

    plt.plot(range(list_out.__len__()), list_out,
             label=[f'RMSE:{np.array(list_out).max(0)[0]}', f'MAE:{np.array(list_out).max(0)[1]}'])
    plt.legend()
    plt.xlabel(u'epoch')
    plt.title(args.fold_num + ' Result')
    plt.show()

    namespace_obj = args

    data = vars(namespace_obj)

    with open(f'{save_dir}/outputs.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter', 'Value'])
        for key, value in data.items():
            if not isinstance(value, (str, int, float, bool)):
                value = str(value)
            writer.writerow([key, value])
        writer.writerow(['Result Min Validation AUC=', max_auc])
        writer.writerow(['Result Min Validation ACC=', max_acc])
        writer.writerow(['Test Result By Eval AUC=', test_auc])
        writer.writerow(['Test Result By Eval ACC=', test_acc])

    print('最大验证AUC=', max_auc)
    print('最大验证ACC=', max_acc)
    return val_uq, val_auc, val_acc

