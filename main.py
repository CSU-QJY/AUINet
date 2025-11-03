import os
from chemprop.parsing import parse_train_args
from chemprop.train import run_training
from chemprop.utils import makedirs

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    args = parse_train_args()
    init_seed = args.seed

    all_scores = []

    for fold_num in ['bbbp','bace','sider','clinTox','tox21','hiv']:

        args.dropouts = 15
        args.fold_num = fold_num
        makedirs(args.save_dir)
        args.data_path = 'datasets/' + fold_num + '.csv'
        if fold_num == 'bbbp':
            args.batch_size = 128
            args.seed = 64
            args.init_lr = 1e-5
            args.max_lr = 1e-4
            args.final_lr = 1e-5
        elif fold_num == 'bace':
            args.batch_size = 128
            args.seed = 64
            args.init_lr = 1e-5
            args.max_lr = 1e-4
            args.final_lr = 1e-5
        elif fold_num == 'clinTox':
            args.batch_size = 128
            args.seed = 2048
            args.init_lr = 1e-4
            args.max_lr = 1e-3
            args.final_lr = 1e-4
        elif fold_num == 'sider':
            args.batch_size = 128
            args.seed = 64
            args.init_lr = 1e-4
            args.max_lr = 1e-3
            args.final_lr = 1e-4
        elif fold_num == 'tox21':
            args.batch_size = 256
            args.seed = 1024
        elif fold_num == 'hiv':
            args.batch_size = 256
            args.seed = 32

        model_scores, model_rmse, model_mae = run_training(args)
        all_scores.append(model_scores)
