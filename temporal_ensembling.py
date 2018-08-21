import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain
from chainer.datasets import DictDataset
from chainer.training import Trainer


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


class Block(Chain):

    def __init__(self, n_out):
        super(Block, self).__init__()
        with self.init_scope():
            self.li = L.Linear(None, n_out)
            self.bn = L.BatchNormalization(n_out)

    def __call__(self, h):
        h = self.li(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


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
                out_size = min(20, num_cat // 2)
                setattr(self, c, L.EmbedID(num_cat, out_size))

            self.b1 = Block(2048)
            self.b2 = Block(1024)
            self.b3 = Block(512)
            self.b4 = Block(256)
            self.b5 = Block(128)
            self.b6 = Block(64)
            self.l1 = L.Linear(None, 1)

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

    def predict(self, **X):
        p = F.sigmoid(self.Z).array
        return p

    def forward(self, X, return_feature=False):
        h = []
        for c in self.cats:
            embed = getattr(self, c)
            h.append(embed(X[c]))

        if 'numerical' in X:
            h.append(X['numerical'])

        h = F.concat(h, axis=1)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = self.b4(h)
        h = self.b5(h)
        h = self.b6(h)

        if return_feature:
            return h

        h = self.l1(h)

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

        bn_loss = F.bernoulli_nll(y[notnull], h[notnull], reduce='no')
        bn_loss = F.mean(bn_loss * (1 + y[notnull]*15))
        loss = 0
        loss += F.mean_squared_error(h, self.z_hat[index])
        loss += bn_loss

        chainer.report({'loss': loss}, self)

        return loss
