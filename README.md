# CoBiC
An unofficial implementation of "Joint Intent Detection and Slot Filling via CNN-LSTM-CRF(https://ieeexplore.ieee.org/abstract/document/9357183)"

# How to run?
## Dataset
To use any other dataset, you should put the files in the following format in `data` folder:
```
$CoBiC
.
├── data
|    ├── dataset_name
|            ├── train
|            |      ├── label
|            |      ├── seq.in
|            |      └── seq.out    
|            ├── dev
|            |      ├── label
|            |      ├── seq.in
|            |      └── seq.out
|            └── test
|                   ├── label
|                   ├── seq.in
|                   └── seq.out
```
You can take a look at `data/atis/` to understand the structure of dataset better.
## Requirements
This code is written in `python3.8`. All the other dependencies and libraries can be installed using following command:
```
~/CoBiC$pip install -r requirements.txt
```
## Configuration
You can change tha parameters in `config.json` to change any configuration. For example, change the name of the dataset.
## Embeddings
The embeddings used here, are the same as the one used in the paper and can be downloaded from here(https://wikipedia2vec.github.io/wikipedia2vec/pretrained/). Contact me in case of facing problems downloading the pre-trained embeddings.
If you are going to use any other kind of embedding file than `word2vec` text format, you should change the body of `get_embedding` function in `modules/data_utils.py`. The return value of the method should be 2d array.
## Run
Finally, to run the code, you can use the following command:
```
~/CoBiC$python main.py
```
# Future
1. Adding single training mode
2. complete documentation

