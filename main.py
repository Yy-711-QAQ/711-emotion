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

    # ================= 基础配置 =================
    args.setdefault('ckpt', None)
    args.setdefault('early_stop_patience', 3)
    args.setdefault('early_stop_min_delta', 1e-4)

    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Start loading the data....")

    # ================= DATA =================
    if args['dataset'] == 'emotiontalk':
        print('Start loading the EmotionTalk data....')

        train_dataset = get_dataset_emotiontalk(args['datapath'], 'train', args['img_interval'])
        valid_dataset = get_dataset_emotiontalk(args['datapath'], 'valid', args['img_interval'])
        test_dataset = get_dataset_emotiontalk(args['datapath'], 'test', args['img_interval'])

        # ===== Weighted Sampler =====
        labels_idx = np.argmax(train_dataset.labels, axis=1)
        class_counts = np.bincount(labels_idx)
        class_weights = 1.0 / np.clip(class_counts, 1e-6, None)
        sample_weights = [class_weights[l] for l in labels_idx]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args['batch_size'],
            sampler=sampler,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )

        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2, collate_fn=collate_fn)

    else:
        raise ValueError("This stable version focuses on emotiontalk only")

    print(f'# Train samples = {len(train_loader.dataset)}')
    print(f'# Valid samples = {len(valid_loader.dataset)}')
    print(f'# Test samples = {len(test_loader.dataset)}')

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    # ================= MODEL =================
    if args['model'] == 'mme2e_sparse':
        from src.models.sparse_e2e import MME2E_Sparse

        # ✅ 修复关键点
        model = MME2E_Sparse(num_classes=7, device=device)
        model = model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args['learning_rate'],
            weight_decay=1e-3
        )

    else:
        raise ValueError("Only mme2e_sparse supported in stable version")

    # ================= SCHEDULER =================
    scheduler = None
    if args.get('scheduler', False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args['epochs'] * len(train_loader))
        )

    # ================= LOSS =================
    if args['loss'] == 'ce':
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    else:
        raise ValueError("Use CE for stable training")

    # ================= TRAINER =================
    trainer = IemocapTrainer(args, model, criterion, optimizer, scheduler, device, dataloaders)

    # ================= LOAD CKPT =================
    if args['ckpt'] is not None and os.path.exists(args['ckpt']):
        state = torch.load(args['ckpt'], map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint from: {args['ckpt']}")

    # ================= RUN =================
    if args.get('test', False):
        trainer.test()
    else:
        trainer.train()

    end = time.time()
    print(f'Total time usage = {(end - start) / 3600:.2f} hours.')