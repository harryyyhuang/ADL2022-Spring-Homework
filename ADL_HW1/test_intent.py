from ast import arg
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

from torch.utils.data import DataLoader
import csv

from tqdm import tqdm

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len, "test")
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(embeddings=embeddings, num_class=dataset.num_classes,
                             args=args).to(args.device)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    # load weights into model
    outputs = []
    for batch_data in tqdm(test_loader):
        data_ids, data_strs, data_labels = batch_data
        data_strs = torch.tensor(data_strs, dtype=torch.long ).to(args.device)
        output = model(data_strs) 
        out_label = output.argmax(dim=1)
        for i, id in enumerate(data_ids):
            label_string = dataset._idx2label[int(out_label[i])]
            outputs.append([id, label_string])
    # TODO: predict dataset

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id', 'intent'])
        for output in outputs:
            writer.writerow(output)



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/baseline.pt",
    )
    parser.add_argument("--pred_file", type=Path, help="Output prediction file", required=True)

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--backbone", type=str, help="RNN, LSTM, GRU", default="GRU") 
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--use_hidden", type=bool, default=False)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
