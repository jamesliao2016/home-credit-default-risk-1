import numpy as np
import pandas as pd
import chainer.links as L
import chainer.functions as F
from chainer import Chain
from chainer.datasets import DictDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler


def create_model_and_datasets(train_df, test_df, target):
    t = train_df.pop(target).values.reshape((-1, 1))
    test_df.pop(target)
    df = pd.concat([train_df, test_df], axis=0)

    # categorical
    categorical_columns = df.select_dtypes('object').columns.tolist()
    train_datasets = {'target': t}
    test_datasets = {}
    num_categories = {}
    for c in categorical_columns:
        df.loc[pd.isnull(df[c]), c] = ''
        train_df.loc[pd.isnull(train_df[c]), c] = ''
        test_df.loc[pd.isnull(test_df[c]), c] = ''
        vectorizer = CountVectorizer()
        vectorizer.fit(df[c])
        train_datasets[c] = vectorizer.transform(train_df[c]).todense()
        test_datasets[c] = vectorizer.transform(test_df[c]).todense()
        num_categories[c] = len(vectorizer.get_feature_names())

    # numeric
    numeric_columns = df.select_dtypes(np.number).columns.tolist()
    for c in numeric_columns:
        vmin = df[c].min()
        vmax = df[c].max()
        v = vmin - 2*(vmax-vmin)
        df.loc[pd.isnull(df[c]), c] = v
        train_df.loc[pd.isnull(train_df[c]), c] = v
        test_df.loc[pd.isnull(test_df[c]), c] = v
    if len(numeric_columns) >= 0:
        scaler = StandardScaler()
        scaler.fit(df[numeric_columns])
        train_datasets['numeric'] = scaler.transform(train_df[numeric_columns])
        test_datasets['numeric'] = scaler.transform(test_df[numeric_columns])

    for c in train_datasets:
        if c == 'target':
            continue
        train_datasets[c] = train_datasets[c].astype('f')
    for c in test_datasets:
        test_datasets[c] = test_datasets[c].astype('f')

    model = NN(num_categories, len(numeric_columns))

    return model, DictDataset(**train_datasets), DictDataset(**test_datasets)


class NN(Chain):
    def __init__(self, num_categories, num_numeric):
        super(NN, self).__init__()
        self.num_categories = num_categories
        self.num_numeric = num_numeric
        with self.init_scope():
            for c, num_category in num_categories.items():
                out_size = min(10, num_category // 2)
                name = 'embed_{}'.format(c)
                setattr(self, name, L.Linear(num_category, out_size))
            self.l1 = L.Linear(num_numeric, 100)
            self.l2 = L.Linear(None, 1)

    def __call__(self, **X):
        h1 = []
        for c in self.num_categories.keys():
            name = 'embed_{}'.format(c)
            embed = getattr(self, name)
            x = X[c]
            h1.append(embed(x))

        if self.num_numeric > 0:
            x = X['numeric']
            h1.append(self.l1(x))

        h1 = F.relu(F.concat(h1, axis=1))
        y = F.sigmoid(self.l2(h1))
        loss = F.sigmoid_cross_entropy(y, X['target'])

        return loss
