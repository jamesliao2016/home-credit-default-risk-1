import numpy as np
import pandas as pd
from nn import NN


def _create_doc_feature(prefix, n):
    data = []
    for i in range(n):
        tmp = []
        for j in range(5):
            k = np.random.randint(20)
            tmp.append(prefix+str(k))
        data.append(' '.join(tmp))
    return np.array(data)


def test_fit():
    np.random.seed(215)
    n = 100
    df = pd.DataFrame(index=np.arange(n))

    df['A'] = _create_doc_feature('a', n)
    df['B'] = _create_doc_feature('b', n)
    df['T'] = np.random.randn(n)
    y = df.pop('T')
    nn = NN(df)

    assert len(nn.cat_columns) == 2

    nn.fit(df, y)
