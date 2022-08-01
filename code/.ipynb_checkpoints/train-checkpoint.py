import json
from pathlib import Path
from dataset import *
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import *
from tqdm import tqdm
import sys, os
from metrics import *
import torch
import argparse
import random


rand_seed = 1120
torch.manual_seed(rand_seed)
random.seed(rand_seed)
np.random.seed(rand_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

data_dir = Path('/workspace/Kaggle/AI4Code/')
# data_dir = Path('..//input/')


parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--device', default='cuda:1', help='device id (i.e., 0 or 0,1 or cpu)')

parser.add_argument('--exp_name', type=str, default='codebert-base-v1')
parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base-mlm')
parser.add_argument('--train_mark_path', type=str, default=data_dir / 'data/train_mark.csv')
parser.add_argument('--train_features_path', type=str, default=data_dir / 'data/train_fts.json')
parser.add_argument('--val_mark_path', type=str, default=data_dir / 'data/val_mark.csv')
parser.add_argument('--val_features_path', type=str, default=data_dir / 'data/val_fts.json')
parser.add_argument('--val_path', type=str, default=data_dir / 'data/val.csv')

parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--code_max_len', type=int, default=23)
parser.add_argument('--total_max_len', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--patience', type=int, default=2)

args = parser.parse_args()

os.makedirs(data_dir / "outputs", exist_ok=True)

exp_name = args.exp_name
os.makedirs(data_dir / f"outputs/{exp_name}", exist_ok=True)

patience = args.patience

train_df_mark = pd.read_csv(args.train_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
train_fts = json.load(open(args.train_features_path))
val_df_mark = pd.read_csv(args.val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
val_fts = json.load(open(args.val_features_path))
val_df = pd.read_csv(args.val_path)

df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

train_ds = MarkdownDataset(train_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                           total_max_len=args.total_max_len, fts=train_fts, code_max_len=args.code_max_len)
val_ds = MarkdownDataset(val_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                         total_max_len=args.total_max_len, fts=val_fts, code_max_len=args.code_max_len)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                          pin_memory=False, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                        pin_memory=False, drop_last=False)


def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


def train(model, train_loader, val_loader, epochs):
    np.random.seed(0)
    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(args.epochs * len(train_loader) / args.accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()


    best_valid = None
    best_valid_epoch = None
    patience_count = 0
    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e} Loss: {avg_loss:.4f} lr: {scheduler.get_last_lr()[0]:.6f}")

        y_val, y_pred = validate(model, val_loader)
        val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
        y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)

        valid_score = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
        print(f"Valid Score: {valid_score:.4f}")

        if best_valid is None:
            best_valid = valid_score
            best_valid_epoch = e
        elif valid_score > best_valid:
            prev_best_valid = best_valid
            best_valid = valid_score
            best_valid_epoch = e
            print(f"Improved (Valid Score: {prev_best_valid:.4f} --> {best_valid:.4f}, Epoch: {best_valid_epoch})")
            torch.save(model.state_dict(), data_dir / f"outputs/{exp_name}/model_best_{best_valid_epoch}epochs_{best_valid:.4f}.bin")
            # Reset
            patience_count = 0
            continue
        else:
            patience_count += 1
            if patience_count > patience:
                print(f"Early stopped (Best Valid Score: {best_valid:.4f}, Epoch: {best_valid_epoch})")
                break
            else:
                print("Not improved.")

        torch.save(model.state_dict(), data_dir / f"outputs/{exp_name}/model_last.bin")

    return model, y_pred


model = MarkdownModel(args.model_name_or_path)
model = model.cuda(args.device)
model, y_pred = train(model, train_loader, val_loader, epochs=args.epochs)
