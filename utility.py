import numpy as np


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
