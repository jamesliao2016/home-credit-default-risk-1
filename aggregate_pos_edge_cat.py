import pandas as pd
from utility import one_hot_encoder, reduce_memory
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


def main():
    pos = pd.read_feather('./data/pos.edge.feather')
    columns = [c for c in pos.columns if pos[c].dtype == 'object']
    pos = pos[['SK_ID_CURR']+columns]
    pos, _ = one_hot_encoder(pos)
    pos = pos.groupby('SK_ID_CURR').mean()
    pos.columns = ['POSEDGE_{}'.format(c) for c in pos.columns]
    reduce_memory(pos)
    pos = pos.reset_index()
    pos.to_feather('./data/pos.edge.agg.cat.feather')


if __name__ == '__main__':
    main()
