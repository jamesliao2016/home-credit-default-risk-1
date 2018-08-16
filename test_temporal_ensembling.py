import pandas as pd
from temporal_ensembling import TemporalEnsembling, create_dataset


def test_hoge():
    df = pd.DataFrame({
        'id': [0, 1, 2, 3],
        'c1': [0, 1, 2, 3],
        'n1': [0, 1, 2, 3],
        't1': [0, 1, 0, None],
    })
    df['c1'] = df['c1'].astype('category')
    df['v1'] = df['c1'].isin([2])
    ds = create_dataset(df, 't1', 'v1')
    model = TemporalEnsembling(df)
    model(**ds[[0, 1, 2, 3]])
    assert 0
