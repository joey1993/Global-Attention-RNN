# Global-Attention-RNN


## Installation

For training, a GPU is strongly recommended for speed. CPU is supported but training could be extremely slow.

### Tensorflow

The code is based on Tensorflow and **supports Tensorflow 0.12.0 now** . You can find installation instructions [here](https://www.tensorflow.org/).

### Dependencies

The code is written in Python 2.7. Its dependencies are summarized in the file ```requirements.txt```. You can install these dependencies like this:
```
pip install -r requirements.txt
```

## Data

We mainly focus on the RecSys2015 dataset, and the code takes vowpal wabbit format [here](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format) as input. 
However, due to the license issue, we are restricted to distribute this dataset.
You should be able to get it [here](https://recsys.acm.org/recsys15/challenge/).

The format of data samples is like:
```
-1 | 6132 8175 8175 10678
1  | 353 1604 1604 1604 1604 1604 1604 1604
-1 | 1007 315 315
```

{-1,1} are conversion tags and {6132,8175,...} are indexes of user behaviors.

## Usage

Here we provide implementations for two global-attention models, one is **GATT** and the other is **LATT**, which jointly trains an **LR** module.
```train.py```, ```attention.py``` and ```module.py``` are scripts for the new attenton models,
```
python train.py
```

## Benchmarks

Here we compare Global Attention models with recent state-of-the-art models (Local Attention model, Multi-head Self-Attention model, LR model) on the RecSys2015 dataset and three private advertising datasets. All experiments are conducted on a 2.8 GHz Intel Core i7 CPU.

