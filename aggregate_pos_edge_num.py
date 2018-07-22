import pandas as pd
from utility import reduce_memory
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


def main():
    pos = pd.read_feather('./data/pos.edge.feather')
    columns = [c for c in pos.columns if pos[c].dtype != 'object']
    pos = pos[columns]
    pos.drop(['SK_ID_PREV'], axis=1)
    pos = pos.groupby('SK_ID_CURR').agg(['mean', 'std', 'sum', 'min', 'max'])
    pos.columns = ['POSEDGE_{}_{}'.format(a, b.upper()) for a, b in pos.columns]
    reduce_memory(pos)
    pos = pos.reset_index()
    pos.to_feather('./data/pos.edge.agg.num.feather')


if __name__ == '__main__':
    main()
