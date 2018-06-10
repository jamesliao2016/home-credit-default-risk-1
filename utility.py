import numpy as np
import pandas as pd


def one_hot_encoder(df):
    original_columns = list(df.columns)
    categorical_columns = [
        col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(
        df, columns=categorical_columns, dummy_na=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def calc_2nd_order_feature(df):
    columns = df.select_dtypes([np.number]).columns.tolist()
    for c in [
        'SK_ID_CURR',
        'SK_ID_PREV',
        'SK_ID_BUREAU',
    ]:
        try:
            columns.remove(c)
        except Exception as e:
            continue

    grp = df.groupby('SK_ID_CURR')[columns].sum().reset_index()
    tcolumns = []
    for i in range(len(columns)):
        a = columns[i]
        for j in range(i+1, len(columns)):
            b = columns[j]
            c = 'NUMERIC_DIV_{}__{}'.format(a, b)
            d = 'NUMERIC_MUL_{}__{}'.format(a, b)
            e = 'NUMERIC_ADD_{}__{}'.format(a, b)
            f = 'NUMERIC_MIN_{}__{}'.format(a, b)
            tcolumns += [c, d, e, f]
            grp[c] = grp[a] / grp[b]
            grp[d] = grp[a] * grp[b]
            grp[e] = grp[a] + grp[b]
            grp[f] = grp[a] - grp[b]

    grp = grp[['SK_ID_CURR'] + tcolumns]
    return grp
