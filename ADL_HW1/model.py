from ast import arg
from typing import Dict

import torch
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings,
        num_class,
        args
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.rnn = getattr(torch.nn, args.backbone)(embeddings.shape[-1], args.hidden_size, args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional)

        rnn_out_num = 2 if (args.bidirectional == True and args.use_hidden == False) else 1

        self.fcn = torch.nn.Linear(args.hidden_size*rnn_out_num, num_class)
        self.fcn_drop = torch.nn.Dropout(args.dropout)
        self.bn1 = torch.nn.BatchNorm1d(args.hidden_size*rnn_out_num)

        self.use_hidden = args.use_hidden


    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward

        text_strs = self.embed(batch)
        text_out, text_hn = self.rnn(text_strs.permute(1, 0, 2))

        out = text_hn if self.use_hidden else text_out
        out = torch.mean(out, dim=0)
        out = self.bn1(out)
        out = self.fcn(out)
        out = self.fcn_drop(out)
     
        return out



class SeqSlotClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings,
        num_class,
        args
    ) -> None:
        super(SeqSlotClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.max_len = args.max_len
        self.num_class = num_class 
        self.hidden_size = args.hidden_size
        
        # TODO: model architecture
        self.rnn = getattr(torch.nn, args.backbone)(embeddings.shape[-1], args.hidden_size, args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional)
        
        self.rnn_out_num = 2 if (args.bidirectional == True) else 1
        self.fcn = torch.nn.Linear(args.hidden_size*self.rnn_out_num, self.num_class)
        self.fcn_drop = torch.nn.Dropout(args.dropout)
        self.bn1 = torch.nn.BatchNorm1d(args.hidden_size*self.rnn_out_num)


    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward

        text_strs = self.embed(batch)
        text_out, text_hn = self.rnn(text_strs.permute(1, 0, 2))
        text_out = text_out.reshape(-1, self.hidden_size*self.rnn_out_num)
        out = self.fcn(text_out)
        out = self.fcn_drop(out)
        
        return out.reshape(self.max_len, -1, self.num_class).permute(1, 0, 2)

    def variableAccuracy(self, outputs, targets, lengths):
        outputs = outputs.reshape(-1, self.max_len)
        targets = targets.reshape(-1, self.max_len)

        accs = 0
        for output, target, length in zip(outputs, targets, lengths):
            accs += all(output[:length] == target[:length])
        return accs/len(lengths)
