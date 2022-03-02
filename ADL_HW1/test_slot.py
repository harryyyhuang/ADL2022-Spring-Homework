import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from dataset import SeqSlotDataset
from model import SeqSlotClassifier
from utils import Vocab

from torch.utils.data import DataLoader
import csv

from tqdm import tqdm

import numpy as np

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

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqSlotDataset(data, vocab, tag2idx, args.max_len, "test")
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqSlotClassifier(embeddings=embeddings, num_class=dataset.num_classes, args=args).to(args.device)


    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    # ckpt = torch.load(args.ckpt_path)
    # load weights into model
    outputs = []

    for batch_data in tqdm(test_loader):
        data_ids, data_strs, data_labels, data_lengths = batch_data
        data_strs = torch.tensor(data_strs, dtype=torch.long ).to(args.device)
        output = model(data_strs) 
        out_tags = output.argmax(dim=2)
        for id, out_tag, data_length in zip(data_ids, out_tags, data_lengths):
            tag_string = dataset.idx2label_list(out_tag, data_length)
            outputs.append([id, tag_string])


    # TODO: predict dataset

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as out_file:
        out_file.write("id,tags\n")
        for output in outputs:
            out_file.write(f"{output[0]},")
            for tag in output[1][:-1]:
                out_file.write(f"{tag} ")
            out_file.write(f"{output[1][-1]}\n")



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
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/slot/baseline.pt",
    )

    parser.add_argument("--pred_file", type=Path, help="Output prediction file", required=True)


    # data
    parser.add_argument("--max_len", type=int, default=40)

    # model
    parser.add_argument("--backbone", type=str, help="RNN, LSTM, GRU", default="GRU") 
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)


    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
