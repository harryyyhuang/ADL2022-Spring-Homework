# Homework 1 ADL NTU 109 Spring

This homework contain two task. First one is intent classification and second one is slot tagging.

## Installation

### Requirements

* Python >= 3.8
* pytorch >= 1.10.2
* numpy
* matplotlib
* tqdm
* seqeval

### Dataset

To run the demo, please run the ./download.sh to download demo dataset and the pretrained model on demo dataset.

If you want to train on custom dataset, please prepare intent data and slot data as demo dataset and run ./preprocess.sh

## Training

### Intent Classification

```
python3 train_intent.py
```

--data_dir is the path to the path to training data
--cache_dir is the path to preprocessed data
--ckpt_dir is the path to save the checkpoint

There's also other few argument associate with model architecture, please check the code for more detail.


### Slot Tagging

```
python3 train_slot.py
```

--data_dir is the path to the path to training data
--cache_dir is the path to preprocessed data
--ckpt_dir is the path to save the checkpoint

There's also other few argument associate with model architecture, please check the code for more detail.

## Evaluation

### Intent Classification

```
./intent_cls.sh {path to testing file} {output_csv}
```

The default checkpoint path will set to ckpt/intent/baseline.pt

### Slot Tagging

```
./slot_tag.sh {path to testing file} {output_csv}
```

The default checkpoint path will set to ckpt/slot/baseline.pt

