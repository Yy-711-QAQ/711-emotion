import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.cli import get_args
from src.datasets import (
    get_dataset_iemocap,
    collate_fn,
    HCFDataLoader,
    get_dataset_mosei,
    collate_fn_hcf_mosei,
    get_dataset_emotiontalk,
)
from src.models.baselines.lf_rnn import LF_RNN
from src.models.baselines.lf_transformer import LF_Transformer
from src.trainers.emotiontrainer import IemocapTrainer


if __name__ == "__main__":
    start = time.time()

    args = get_args()
    if 'ckpt' not in args:
        args['ckpt'] = None
    if args['ckpt'] is None:
        args['ckpt'] = '/home/adc/my_torch_project/Multimodal-End2end-Sparse/savings/models/emotiontalk_mme2e_a_best_epoch29.pt'
    if 'early_stop_patience' not in args:
        args['early_stop_patience'] = 3
    if 'early_stop_min_delta' not in args:
        args['early_stop_min_delta'] = 1e-4
    if 'early_stop_patience' not in args:
        args['early_stop_patience'] = 3
    if 'early_stop_min_delta' not in args:
        args['early_stop_min_delta'] = 1e-4

    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Start loading the data....")

    if args['dataset'] == 'iemocap':
        train_dataset = get_dataset_iemocap(
            data_folder=args['datapath'],
            phase='train',
            img_interval=args['img_interval'],
            hand_crafted_features=args['hand_crafted']
        )
        valid_dataset = get_dataset_iemocap(
            data_folder=args['datapath'],
            phase='valid',
            img_interval=args['img_interval'],
            hand_crafted_features=args['hand_crafted']
        )
        test_dataset = get_dataset_iemocap(
            data_folder=args['datapath'],
            phase='test',
            img_interval=args['img_interval'],
            hand_crafted_features=args['hand_crafted']
        )

        if args['hand_crafted']:
            train_loader = HCFDataLoader(
                dataset=train_dataset,
                feature_type=args['audio_feature_type'],
                batch_size=args['batch_size'],
                shuffle=True,
                num_workers=2
            )
            valid_loader = HCFDataLoader(
                dataset=valid_dataset,
                feature_type=args['audio_feature_type'],
                batch_size=args['batch_size'],
                shuffle=False,
                num_workers=2
            )
            test_loader = HCFDataLoader(
                dataset=test_dataset,
                feature_type=args['audio_feature_type'],
                batch_size=args['batch_size'],
                shuffle=False,
                num_workers=2
            )
        else:
            train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2, collate_fn=collate_fn)

    elif args['dataset'] == 'mosei':
        train_dataset = get_dataset_mosei(
            data_folder=args['datapath'],
            phase='train',
            img_interval=args['img_interval'],
            hand_crafted_features=args['hand_crafted']
        )
        valid_dataset = get_dataset_mosei(
            data_folder=args['datapath'],
            phase='valid',
            img_interval=args['img_interval'],
            hand_crafted_features=args['hand_crafted']
        )
        test_dataset = get_dataset_mosei(
            data_folder=args['datapath'],
            phase='test',
            img_interval=args['img_interval'],
            hand_crafted_features=args['hand_crafted']
        )

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2, collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2, collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)

    elif args['dataset'] == 'emotiontalk':
        print('Start loading the EmotionTalk data....')
        train_dataset = get_dataset_emotiontalk(
            data_folder=args['datapath'],
            phase='train',
            img_interval=args['img_interval']
        )
        valid_dataset = get_dataset_emotiontalk(
            data_folder=args['datapath'],
            phase='valid',
            img_interval=args['img_interval']
        )
        test_dataset = get_dataset_emotiontalk(
            data_folder=args['datapath'],
            phase='test',
            img_interval=args['img_interval']
        )

        train_labels_idx = np.argmax(train_dataset.labels, axis=1)
        class_counts = np.bincount(train_labels_idx, minlength=args['num_emotions'])
        class_weights_np = 1.0 / np.power(np.clip(class_counts, 1, None), 0.25)
        sample_weights = class_weights_np[train_labels_idx]
        train_sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=False, sampler=train_sampler, num_workers=2, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2, collate_fn=collate_fn)

    else:
        raise ValueError(f"Unsupported dataset: {args['dataset']}")

    print(f'# Train samples = {len(train_loader.dataset)}')
    print(f'# Valid samples = {len(valid_loader.dataset)}')
    print(f'# Test samples = {len(test_loader.dataset)}')

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    lr = args['learning_rate']

    if args['model'] == 'mme2e':
        from src.models.e2e import MME2E
        model = MME2E(args=args, device=device)
        model = model.to(device=device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=args['learning_rate'],
            weight_decay=args['weight_decay']
        )

    elif args['model'] == 'mme2e_sparse':
        from src.models.sparse_e2e import MME2E_Sparse
        model = MME2E_Sparse(args=args, device=device)
        model = model.to(device=device)

        # 🔥 统一安全写法（避免A分支错误）
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=args['learning_rate'],
            weight_decay=args['weight_decay']
        )


    elif args['model'] == 'lf_rnn':
        model = LF_RNN(args)
        model = model.to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args['weight_decay'])

    elif args['model'] == 'lf_transformer':
        model = LF_Transformer(args)
        model = model.to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args['weight_decay'])

    else:
        raise ValueError(f"Incorrect model name: {args['model']}")

    if args['scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args['epochs'] * len(train_loader.dataset) // args['batch_size']
        )
    else:
        scheduler = None

    if args['loss'] == 'l1':
        criterion = torch.nn.L1Loss()
    elif args['loss'] == 'mse':
        criterion = torch.nn.MSELoss()
    elif args['loss'] == 'ce':
        if args['dataset'] == 'emotiontalk':
            train_labels_idx = np.argmax(train_dataset.labels, axis=1)
            class_counts = np.bincount(train_labels_idx, minlength=args['num_emotions'])
            class_weights = class_counts.sum() / np.clip(class_counts, 1, None)
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
        else:
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    elif args['loss'] == 'bce':
        pos_weight = train_dataset.getPosWeight()
        pos_weight = torch.tensor(pos_weight).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unsupported loss: {args['loss']}")

    if args['dataset'] in ['iemocap', 'mosei', 'emotiontalk']:
        trainer = IemocapTrainer(args, model, criterion, optimizer, scheduler, device, dataloaders)
    else:
        raise ValueError(f"Unsupported dataset for trainer: {args['dataset']}")

    if args.get('ckpt'):
        state = torch.load(args['ckpt'], map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint from: {args['ckpt']}")

    if args['test']:
        trainer.test()
    elif args['valid']:
        trainer.valid()
    else:
        trainer.train()

    end = time.time()
    print(f'Total time usage = {(end - start) / 3600:.2f} hours.')
