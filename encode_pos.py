import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 100)


def main():
    pos_df = pd.read_feather('./data/POS_CASH_balance.csv.feather')
    columns = pos_df.select_dtypes([np.number]).columns.tolist()
    columns.remove('SK_ID_CURR')
    columns.remove('SK_ID_PREV')

    grp = pos_df.groupby('SK_ID_CURR')[columns].sum().reset_index()
    tcolumns = []
    for i in range(len(columns)):
        a = columns[i]
        for j in range(i+1, len(columns)):
            b = columns[j]
            c = 'pos_NUMERIC_DIV_{}__{}'.format(a, b)
            d = 'pos_NUMERIC_MUL_{}__{}'.format(a, b)
            tcolumns += [c, d]
            grp[c] = grp[a] / grp[b]
            grp[d] = grp[a] * grp[b]

    grp = grp[['SK_ID_CURR'] + tcolumns]
    grp.to_feather('./data/POS_CASH_balance.numeric.2nd-order.feather')


if __name__ == '__main__':
    main()
