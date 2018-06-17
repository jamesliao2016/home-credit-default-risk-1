from utility import one_hot_encoder


def aggregate_pos(df, key):
    # count
    grp = df.groupby(key)
    g = grp[['MONTHS_BALANCE']].count()
    g.columns = ['COUNT']
    pos_agg = g

    # aggregate
    df, cat_columns = one_hot_encoder(df)
    grp = df.groupby(key)
    fs = ['sum', 'median', 'mean', 'std', 'max', 'min']
    agg = {
        # original
        'CNT_INSTALMENT': fs,
        'CNT_INSTALMENT_FUTURE': fs,
        'SK_DPD': fs,
        'SK_DPD_DEF': fs,
        # preprocessed
        'RATIO_CNT_INST': ['max', 'min'],
        'DIFF_CNT_INSTALMENT': ['mean', 'std', 'max', 'min'],
        'DIFF_CNT_INSTALMENT_FUTURE': ['mean', 'std', 'max', 'min'],
        'DIFF_SK_DPD': ['min', 'max'],
        'DIFF_SK_DPD_DEF': ['min', 'max'],
    }
    for c in cat_columns:
        agg[c] = ['mean']
    g = grp.agg(agg)
    g.columns = ['{}_{}'.format(a, b.upper()) for a, b in g.columns]
    pos_agg = pos_agg.join(g, on=key, how='left')

    return pos_agg
