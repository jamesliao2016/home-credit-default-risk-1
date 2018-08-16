import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, cuda
from chainer.optimizers import Adam
from chainer.datasets import DictDataset
from chainer.training import Trainer
from chainer.training import extensions, make_extension
from chainer.iterators import SerialIterator
from chainer.training.updaters import StandardUpdater
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from utility import reset_seed


def create_dataset(df, target, valid):
    datasets = {}
    datasets['target'] = df.pop(target).values
    datasets['valid'] = df.pop(valid).values
    nums = []
    for c in df.columns:
        if str(df[c].dtype) != 'category':
            df[c] = df[c].astype('f')
            nums.append(c)
            continue
        datasets[c] = df[c].astype('i').values
    if len(nums) > 0:
        datasets['numerical'] = df[nums].values
    return DataSet(**datasets)


class DataSet(DictDataset):

    def __getitem__(self, index):
        item = super(DataSet, self).__getitem__(index)
        item['index'] = index
        return item


class UpdateExtention():

    __name__ = 'update-extention'

    def __call__(self, trainer: Trainer):
        t = trainer.updater.epoch
        trainer.updater.get_optimizer('main').target.update(t)


class Model(Chain):

    def __init__(self, n):
        super(Model, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 128)
            self.l2 = L.Linear(None, 64)
            self.l3 = L.Linear(None, 1)

        self.n = n


class TemporalEnsembling(Chain):

    def __init__(self, df):
        super(TemporalEnsembling, self).__init__()
        self.initialized = False
        with self.init_scope():
            self.n = df.shape[0]
            self.cats = []
            for c in df.columns:
                if str(df[c].dtype) != 'category':
                    continue
                self.cats.append(c)
                num_cat = len(df[c].cat.categories)
                print(df[c].cat.categories, num_cat)
                out_size = min(30, num_cat // 2)
                setattr(self, c, L.EmbedID(num_cat, out_size))

            self.l1 = L.Linear(None, 1200)
            self.bn1 = L.BatchNormalization(1200)
            self.l2 = L.Linear(None, 800)
            self.bn2 = L.BatchNormalization(800)

            self.l3 = L.Linear(None, 1)
        print(self.cats)

    def update(self, t):
        alpha = 0.6
        xp = self.xp
        self.Z = alpha * self.Z + (1.-alpha) * self.z
        self.z_hat = (self.Z / (1.-xp.power(alpha, t))).astype('f')
        self.z = xp.zeros((self.n, 1))

    def reset(self):
        xp = self.xp
        self.z = xp.zeros((self.n, 1), dtype='f')  # temporal
        self.Z = xp.zeros((self.n, 1), dtype='f')  # ensemble prediction
        self.z_hat = xp.zeros((self.n, 1), dtype='f')  # target vector
        self.initialized = True

    def validate(self, **X):
        valid = X['valid'].reshape(-1, 1)

        y = X['target'].reshape(-1, 1)
        a = cuda.to_cpu(y[valid]).reshape(-1).astype('i')
        b = cuda.to_cpu(F.sigmoid(self.Z)[valid].array).reshape(-1)
        score = roc_auc_score(a, b)

        chainer.report({'score': score}, self)

    def forward(self, X):
        h = []
        for c in self.cats:
            embed = getattr(self, c)
            h.append(embed(X[c]))

        if 'numerical' in X:
            h.append(X['numerical'])

        h = F.concat(h, axis=1)
        h = self.l1(h)
        h = self.bn1(h)
        h = F.dropout(h, 0.1)
        h = F.relu(h)
        h = self.l2(h)
        h = self.bn2(h)
        h = F.dropout(h, 0.1)

        h = self.l3(h)

        return h

    def __call__(self, **X):
        if not self.initialized:
            self.reset()

        xp = self.xp
        index = X['index']
        y = X['target'].reshape(-1, 1)
        valid = X['valid'].reshape(-1, 1)
        notnull = (~xp.isnan(y)) & (~valid)

        h = self.forward(X)
        self.z[index] = h.array

        loss = 0
        loss += F.bernoulli_nll(y[notnull], h[notnull])
        loss += F.mean_squared_error(h, self.z_hat[index])

        chainer.report({'loss': loss}, self)

        return loss
