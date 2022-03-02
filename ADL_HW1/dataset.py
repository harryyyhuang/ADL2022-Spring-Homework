from tkinter.messagebox import NO
from typing import List, Dict
import torch

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        data_split: str
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.data_split = data_split

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def getMaxLength(self):
        self.max_length = max(len(per_data["text"].split()) for per_data in self.data)
        print(self.max_length)
    
    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]):
        # TODO: implement collate_fn
        text_strs = []
        text_labels = []
        text_ids = []
        for data in samples:
            text_strs.append(data['text'])
            if(self.data_split != "test"):
                text_labels.append(self.label_mapping[data['intent']])
            text_ids.append(data['id'])
        text_strs = Vocab.encode_batch(self.vocab, text_strs, self.max_len)
        return text_ids, text_strs, text_labels

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class SeqSlotDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        tag_mapping: Dict[str, int],
        max_len: int,
        data_split: str
        ):
        
        self.data = data
        self.vocab = vocab
        self.tag_mapping = tag_mapping
        self._idx2tag = {idx: tag for tag, idx in self.tag_mapping.items()}
        self.data_split = data_split
        self.max_len = max_len
        self.pad_tag = len(tag_mapping)

        print(f'The data length of {self.data_split} is {self.getMaxLength():.2f}')

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        return self.data[index]

    # @property
    def getMaxLength(self):
        return max(len(per_data["tokens"]) for per_data in self.data)

    @property
    def num_classes(self) -> int:
        return len(self.tag_mapping) + 1

    def collate_fn(self, samples: List[Dict]):
        text_strs = []
        text_tags = []
        text_ids = []
        text_lengths = []

        for data in samples:
            text_strs.append(data['tokens'])
            if(self.data_split != 'test'):
                text_tags.append(data['tags'])
            text_ids.append(data['id'])
            text_lengths.append(len(data['tokens']))

        text_strs = Vocab.encode_batch(self.vocab, text_strs, self.max_len)
        text_tags = self.label2idx_batch(text_tags, self.max_len)
        return text_ids, text_strs, text_tags, text_lengths

    def encod_tags(self, tags: List[str]) -> List[int]:
        return [self.label2idx(tag) for tag in tags]

    def label2idx_batch(self, batch_tags: List[List[int]], to_len: int = None):
        batch_ids = [self.encod_tags(tags) for tags in batch_tags]
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len
        padded_ids = pad_to_len(batch_ids, to_len, self.pad_tag)

        return padded_ids

    def idx2label_list(self, list_tags: List[int], to_len: int):
        list_tagstrs = [self.idx2label(int(idx)) for idx in list_tags[:to_len]]
        return list_tagstrs
        
    def label2idx(self, label: str):
        return self.tag_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2tag[idx]

