# Homework 2 ADL NTU 109 Spring

This homework contain two tasks. First we select the correct context for question, than we answer the question using context.

### Requirements

* Python >= 3.8
* pytorch >= 1.10.2
* numpy
* matplotlib
* tqdm
* huggingface relevent package (clearify in the future)

### Dataset
To load the pretrained model trained on demo dataset, please run ./download.sh
To load the demo dataset, please run ./download_data.sh

## Training

To run the training, we need preprocess the data to specific form

```
python3 preprocess_choice.py
```

```
python3 preprocess_qa.py
```

start training selection !

```
./train_selection.sh
```

We have provide some default config in train_selection.sh file.


start training qa !

```
./train_qa.sh
```


## Inference 

To Inference test data, please prepare data set as given demo test.json and context.json

start inference !

```
./run.sh
```
