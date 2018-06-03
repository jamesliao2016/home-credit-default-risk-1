import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
pd.set_option("display.max_columns", 100)


def concat(x):
    doc = [str(t) for t in x.tolist()]
    return ' '.join(doc)


def preprocess_prev(prev_df):
    grp = prev_df.groupby('SK_ID_CURR')
    grp = grp[['SK_ID_PREV']].count()
    res = grp.reset_index()
    res = res[['SK_ID_CURR']]

    grp = prev_df.groupby('SK_ID_CURR')
    grp = grp['SELLERPLACE_AREA'].apply(concat)
    grp = grp.reset_index()
    grp = grp.rename(columns={'SELLERPLACE_AREA': 'AREA_DOC'})
    vectorizer = CountVectorizer()
    tf = vectorizer.fit_transform(grp['AREA_DOC'])
    lda = LatentDirichletAllocation(learning_method='online', n_jobs=12)
    comp = lda.fit_transform(tf)
    for i in range(comp.shape[1]):
        grp['LDA_COMP_{}_SELLERPLACE_AREA'.format(i)] = comp[:, i]
    del grp['AREA_DOC']
    res = res.merge(grp, on='SK_ID_CURR', how='left')
    print(res.head())

    return res


def main():
    np.random.seed(215)
    prev_df = pd.read_feather('./data/previous_application.csv.feather')
    prev_df = prev_df.sort_values('SK_ID_CURR')
    res = preprocess_prev(prev_df)
    res = res.set_index('SK_ID_CURR')
    res.columns = [
        'prev_decomposed_{}'.format(c) for c in res.columns]
    res = res.reset_index()
    res.to_feather('./data/decomposed_previous_application.csv.feather')


if __name__ == '__main__':
    main()
