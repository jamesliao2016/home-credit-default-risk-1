import numpy as np
import pandas as pd
from chainer.training import Trainer
from chainer.iterators import SerialIterator
from chainer.optimizers import Adam
from chainer.training.updaters import StandardUpdater
from chainer.training.extensions import Evaluator
from nn import create_model_and_datasets


def _create_doc_feature(prefix, n):
    data = []
    for i in range(n):
        tmp = []
        for j in range(5):
            k = np.random.randint(20)
            tmp.append(prefix+str(k))
        data.append(' '.join(tmp))
    return np.array(data)


def _create_df(n):
    df = pd.DataFrame(index=np.arange(n))
    df['A'] = _create_doc_feature('a', n)
    df['B'] = _create_doc_feature('b', n)
    df.loc[:50, 'B'] = np.NaN
    df['C'] = np.random.randn(n)
    df['D'] = np.random.randn(n)
    df.loc[:50, 'D'] = np.NaN
    return df


def test_fit():
    np.random.seed(215)
    train_df = _create_df(100)
    train_df['T'] = (np.random.randn(100) > 0).astype('i')
    test_df = _create_df(20)
    model, train_dset, test_dset = create_model_and_datasets(
        train_df, test_df, 'T')

    train_iter = SerialIterator(train_dset, 10)
    test_iter = SerialIterator(train_dset, 10, False, False)
    optimizer = Adam()
    optimizer.setup(model)
    updater = StandardUpdater(train_iter, optimizer, device=0)
    trainer = Trainer(updater, (5, 'epoch'), out='nn_result')
    trainer.extend(Evaluator(test_iter, model, device=0))
    trainer.run()
