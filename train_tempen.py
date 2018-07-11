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


class UpdateExtention():

    __name__ = 'update-extention'

    def __call__(self, trainer: Trainer):
        t = trainer.updater.epoch
        trainer.updater.get_optimizer('main').target.update(t)


class TemporalEmsenbling(Chain):

    def __init__(self, n):
        super(TemporalEmsenbling, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 128)
            self.l2 = L.Linear(None, 64)
            self.l3 = L.Linear(None, 1)

        self.n = n
        self.initialized = False

    def reset(self):
        xp = self.xp
        self.z = xp.zeros((self.n, 1), dtype='f')  # temporal
        self.Z = xp.zeros((self.n, 1), dtype='f')  # ensemble prediction
        self.z_hat = xp.zeros((self.n, 1), dtype='f')  # target vector

    def update(self, t):
        alpha = 0.6
        xp = self.xp
        self.Z = alpha * self.Z + (1.-alpha) * self.z
        self.z_hat = self.Z / (1.-xp.power(alpha, t))
        self.z = xp.zeros((self.n, 1))

    def forward(self, x):
        # TODO: augmentation
        h = self.l1(x)
        h = F.relu(h)
        h = F.dropout(h)
        h = self.l2(h)
        h = F.relu(h)
        h = F.dropout(h)
        h = self.l3(h)
        return h

    def __call__(self, data):
        if not self.initialized:
            self.reset()

        x = data[:, :-2]  # features
        i = data[:, -2].astype('i')  # index
        y = data[:, -1:]  # label

        h = self.forward(x)

        self.z[i] = h.array

        xp = self.xp
        idx = ~xp.isnan(y)

        loss = 0
        loss += F.bernoulli_nll(y[idx], h[idx]) / len(y[idx])
        loss += F.mean_squared_error(h, self.z_hat[i])

        a = cuda.to_cpu(y[idx]).reshape(-1).astype('i')
        b = cuda.to_cpu(F.sigmoid(h)[idx].array).reshape(-1)
        score = roc_auc_score(a, b)

        chainer.report({'loss': loss}, self)
        chainer.report({'score': score}, self)

        return loss


def main():
    reset_seed(0)
    train = pd.read_feather('./data/application_train.preprocessed.feather')
    test = pd.read_feather('./data/application_test.preprocessed.feather')
    test['TARGET'] = np.nan

    df = pd.concat([train, test], sort=False).reset_index(drop=True)
    df['index'] = np.arange(len(df))

    features = ['EXT_SOURCE_1']
    for c in features:
        m = df[c].mean()
        df[pd.isnull(df[c])] = m

    train = df[pd.notnull(df['TARGET'])]
    test = df[pd.isnull(df['TARGET'])]

    features += ['index', 'TARGET']
    train_dset = train[features].values.astype('f')
    test_dset = test[features].values.astype('f')

    train_dset, valid_dset = train_test_split(
        train_dset, test_size=0.20, random_state=215,
        stratify=train['TARGET'])

    train_dset = np.vstack([train_dset, test_dset])
    print(train_dset.shape)

    train_iter = SerialIterator(train_dset, 512)
    valid_iter = SerialIterator(valid_dset, len(valid_dset), False, False)

    model = TemporalEmsenbling(len(df))

    optimizer = Adam()
    optimizer.setup(model)
    updater = StandardUpdater(train_iter, optimizer, device=0)
    trainer = Trainer(updater, (10, 'epoch'), out='tempem_result')
    trainer.extend(make_extension((1, 'epoch'))(UpdateExtention()))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(
        valid_iter, model, device=0), name='val')
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
