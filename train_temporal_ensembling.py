import matplotlib
matplotlib.use('Agg')  # noqa
import pandas as pd
import chainer
import numpy as np
from chainer import cuda
from chainer.optimizers import Adam
from chainer.training import Trainer
from chainer.training import extensions, make_extension
from chainer.iterators import SerialIterator
from chainer.training.updaters import StandardUpdater
from utility import reset_seed
from sklearn.metrics import roc_auc_score
from temporal_ensembling import TemporalEnsembling, create_dataset


class UpdateExtention():

    __name__ = 'update-extention'

    def __init__(self, dataset, filename):
        d = dataset[range(len(dataset))]
        self.dataset = {'valid': d['valid'], 'target': d['target']}
        self.filename = filename

    def __call__(self, trainer: Trainer):
        print('update')
        model = trainer.updater.get_optimizer('main').target
        model.update(trainer.updater.epoch)
        valid = self.dataset['valid']
        y_true = self.dataset['target'].reshape(-1, 1)
        y_score = cuda.to_cpu(model.predict())
        auc = roc_auc_score(y_true[valid], y_score[valid])
        chainer.report({'val/auc': auc}, model)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            f = cuda.to_cpu(model.feature)
            f = np.concatenate([y_score, f], axis=1)
            df = pd.DataFrame(f, columns=['nn_{}'.format(i) for i in range(f.shape[1])])
            df.to_feather(self.filename)


def train(idx):
    print('train by fold: {}'.format(idx))
    df = pd.read_feather('./data/features.normalized.feather')
    fold = pd.read_feather('./data/fold.{}.feather'.format(idx))

    df['valid'] = df['SK_ID_CURR'].isin(fold['SK_ID_CURR'])
    model = TemporalEnsembling(df)

    train_dset = create_dataset(df, 'TARGET', 'valid')
    train_iter = SerialIterator(train_dset, 512)

    optimizer = Adam()
    optimizer.setup(model)
    updater = StandardUpdater(train_iter, optimizer, device=0)
    trainer = Trainer(updater, (20, 'epoch'), out='tempem_result')
    trainer.extend(make_extension((1, 'epoch'))(
        UpdateExtention(train_dset, './data/nn.fold.{}.feather'.format(idx))))
    trainer.extend(extensions.LogReport())
    filename = 'fold_%d_snapshot_epoch-{.updater.epoch}' % (idx)
    trainer.extend(extensions.snapshot(
        filename=filename))
    trainer.extend(extensions.PrintReport([
        'epoch', 'main/loss', 'main/val/auc',
        'elapsed_time']))
    trainer.run()


def main():
    reset_seed(0)
    for i in range(5):
        train(i)


if __name__ == '__main__':
    main()
