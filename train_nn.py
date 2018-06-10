import matplotlib
matplotlib.use('Agg')  # noqa
import random
import chainer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from chainer.training import Trainer
from chainer.iterators import SerialIterator
from chainer.optimizers import Adam
from chainer.training.updaters import StandardUpdater
from chainer.training import extensions
from nn import create_model_and_datasets
pd.set_option("display.max_columns", 1000)


def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


def main():
    reset_seed(0)
    train_df = pd.read_feather(
        './data/application_train.concat.feather', 12)
    test_df = pd.read_feather(
        './data/application_test.concat.feather', 12)[:1000]

    model, train_dset, test_dset = create_model_and_datasets(
        train_df, test_df, 'TARGET')

    train_dset, valid_dset = train_test_split(
        train_dset, test_size=0.1, random_state=215,
        stratify=train_df['TARGET'])

    train_iter = SerialIterator(train_dset, 256)
    valid_iter = SerialIterator(valid_dset, len(valid_dset), False, False)
    # test_iter = SerialIterator(test_dset, 10, False, False)
    optimizer = Adam()
    optimizer.setup(model)
    updater = StandardUpdater(train_iter, optimizer, device=0)
    trainer = Trainer(updater, (100, 'epoch'), out='nn_result')
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(
        valid_iter, model, device=0), name='val')
    trainer.extend(extensions.PrintReport([
        'epoch', 'main/loss', 'main/score',
        'val/main/loss', 'val/main/score', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'],
        x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/score', 'val/main/score'],
        x_key='epoch', file_name='score.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.run()


if __name__ == '__main__':
    main()
