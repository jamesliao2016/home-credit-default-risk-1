import matplotlib
matplotlib.use('Agg')  # noqa
import chainer
import numpy as np
import pandas as pd
import chainer.links as L
import chainer.functions as F
from chainer import Chain, cuda
from chainer.optimizers import Adam
from chainer.training import Trainer
from chainer.training import extensions, make_extension
from chainer.iterators import SerialIterator
from chainer.training.updaters import StandardUpdater
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from utility import reset_seed
from temporal_ensembling import TemporalEnsembling, create_dataset


class UpdateExtention():

    __name__ = 'update-extention'

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, trainer: Trainer):
        t = trainer.updater.epoch
        model = trainer.updater.get_optimizer('main').target
        model.update(t)
        model.validate(**self.dataset[range(len(self.dataset))])


def main():
    reset_seed(0)
    df = pd.read_feather('./data/features.normalized.feather')
    fold = pd.read_feather('./data/fold.0.feather')

    df['valid'] = df['SK_ID_CURR'].isin(fold['SK_ID_CURR'])
    model = TemporalEnsembling(df)

    train_dset = create_dataset(df, 'TARGET', 'valid')
    train_iter = SerialIterator(train_dset, 512)

    optimizer = Adam()
    optimizer.setup(model)
    updater = StandardUpdater(train_iter, optimizer, device=0)
    trainer = Trainer(updater, (10, 'epoch'), out='tempem_result')
    trainer.extend(make_extension((1, 'epoch'))(UpdateExtention(train_dset)))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.PrintReport([
        'epoch', 'main/loss', 'main/score',
        'val/main/loss', 'val/main/score',
        'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'],
        x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.run()


if __name__ == '__main__':
    main()
