# Homework 3 ADL NTU 109 Spring

This homework is to summarize context download from udn web.

### Requirements

* Python >= 3.8
* pytorch >= 1.10.2
* numpy
* matplotlib
* tqdm
* huggingface relevent package (clearify in the future)
* nlk

### Dataset
To load the pretrained model trained on demo dataset, please run ./download.sh
To load the demo dataset, please run ./download_data.sh

## Training

To run the training, please simply run 

```
python3 summary_no_trainer.py
```


We have provide some default config in train_summary.sh file.



## Inference 

To Inference test data, please prepare data set as given demo public.jsonl or sample_test.jsonl

start inference !

```
./run.sh
```
