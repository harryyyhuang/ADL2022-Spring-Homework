from ast import arg
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict


import torch
from tqdm import trange
from tqdm import tqdm

from dataset import SeqClsDataset
from utils import Vocab

from torch.utils.data import DataLoader

from model import SeqClassifier

import numpy as np

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    same_seeds(876)
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len, 'train')
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(datasets[TRAIN], batch_size=args.batch_size, shuffle=True,
                            collate_fn=datasets[TRAIN].collate_fn)
    eval_loader = DataLoader(datasets[DEV], batch_size=args.batch_size, shuffle=False,
                            collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings=embeddings, num_class=datasets[TRAIN].num_classes,
                             args=args).to(args.device)
    
    # TODO: init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")

    model.embed.eval()

    best_eval_acc = 0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        accs_train = []
        losses_train = []
        model.train()
        for batch_data in train_loader:
            data_ids, data_strs, data_labels = batch_data
            data_strs = torch.tensor(data_strs, dtype=torch.long ).to(args.device)
            data_labels = torch.tensor(data_labels, dtype=torch.long).to(args.device)
            output = model(data_strs) 

            loss = criterion(output, data_labels)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            out_label = output.argmax(dim=1)
            
            acc_train = (out_label == data_labels).float().mean()

            accs_train.append(acc_train)
            losses_train.append(loss)

        print(f"avg train accuracy : {sum(accs_train)/len(accs_train):.2f}, epoch : {epoch}")
        print(f"avg train loss : {sum(losses_train)/len(losses_train):.2f}, epoch : {epoch}")
        # # TODO: Evaluation loop - calculate accuracy and save model weights
        
        accs_eval = []
        losses_eval = []
        # model.rnn.eval()
        # model.fcn.eval()
        model.eval()
        for batch_data in eval_loader:
            data_ids, data_strs, data_labels = batch_data
            data_strs = torch.tensor(data_strs, dtype=torch.long ).to(args.device)
            data_labels = torch.tensor(data_labels, dtype=torch.long).to(args.device)
            output = model(data_strs) 
            
            loss = criterion(output, data_labels)

            out_label = output.argmax(dim=1)

            acc_eval = (out_label == data_labels).float().mean()

            accs_eval.append(acc_eval)
            losses_eval.append(loss)

        print(f"avg eval accuracy : {sum(accs_eval)/len(accs_eval):.2f}, epoch : {epoch}")
        print(f"avg eval loss : {sum(losses_eval)/len(losses_eval):.2f}, epoch : {epoch}")
        current_eval_acc = sum(accs_eval)/len(accs_eval)
        if(current_eval_acc > best_eval_acc):
            best_eval_acc = current_eval_acc
            print("saving model ...")
            torch.save(model.state_dict(), args.ckpt_dir / "baseline.pt")



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--backbone", type=str, help="RNN, LSTM, GRU", default="GRU") 
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--use_hidden", type=bool, default=False)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=20)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
