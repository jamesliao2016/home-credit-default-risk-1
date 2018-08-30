## About

- 167th solution of [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)

## Usage

- Put `*.csv.zip` into `./data/`

```
$ make data/*
$ python join.py
$ python normalize.py
$ python train_temporal_ensembling.py
$ python train_lightgbm.py --seed 0 --param-idx 0
```
