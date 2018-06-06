import chainer.links as L
import chainer.functions as F
from chainer import Chain
from sklearn.feature_extraction.text import CountVectorizer


class NN(Chain):
    # TODO: separate preprocessing and network?

    def __init__(self, df):
        super(NN, self).__init__()
        self.cat_columns = df.select_dtypes('object').columns.tolist()
        self.feature_names = {}
        self.vectorizer = {}
        for c in self.cat_columns:
            vectorizer = CountVectorizer()
            vectorizer.fit(df[c])
            self.vectorizer[c] = vectorizer
            self.feature_names[c] = vectorizer.get_feature_names()

        with self.init_scope():
            for c, values in self.feature_names.items():
                out_size = min(10, len(values) // 2)
                name = 'embed_{}'.format(c)
                setattr(self, name, L.Linear(len(values), out_size))

    def fit(self, X, y):
        embed_output = []
        for c, values in self.feature_names.items():
            name = 'embed_{}'.format(c)
            embed = getattr(self, name)
            x = self.vectorizer[c].transform(X[c])
            x = x.todense().astype('f')
            x = embed(x)
            embed_output.append(x)
        embed_output = F.concat(embed_output, axis=1)
        print(embed_output.shape)
