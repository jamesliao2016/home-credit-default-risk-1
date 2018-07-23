import numpy as np
import pandas as pd
from utility import reduce_memory
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


def preprocess_bureau():
    bure = pd.read_feather('./data/bureau.feather')
    bure['FINISHED'] = (bure['DAYS_ENDDATE_FACT'] <= 0).astype('i')
    indexer = pd.isnull(bure['DAYS_CREDIT_ENDDATE'])
    bure.loc[indexer, 'DAYS_CREDIT_ENDDATE'] = bure.loc[indexer, 'DAYS_ENDDATE_FACT']
    bure['DIFF_ENDDATE'] = bure['DAYS_ENDDATE_FACT'] - bure['DAYS_CREDIT_ENDDATE']
    bure['TERM'] = bure['DAYS_CREDIT_ENDDATE'] - bure['DAYS_CREDIT']
    bure['AMT_CREDIT_SUM_OVERDUE'].fillna(0, inplace=True)
    bure['AMT_CREDIT_SUM'].fillna(0, inplace=True)
    bure['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)
    bure['CNT_CREDIT_PROLONG'].fillna(0, inplace=True)
    bure['AMT_ANNUITY'].fillna(0, inplace=True)
    # discussion/57175
    bure.loc[bure['DAYS_CREDIT_ENDDATE'] < -40000, 'DAYS_CREDIT_ENDDATE'] = np.nan
    bure.loc[bure['DAYS_CREDIT_UPDATE'] < -40000, 'DAYS_CREDIT_UPDATE'] = np.nan
    bure.loc[bure['DAYS_ENDDATE_FACT'] < -40000, 'DAYS_ENDDATE_FACT'] = np.nan
    bure['FLAG_ACTIVE'] = (bure['CREDIT_ACTIVE'] != 'Closed').astype('int8')
    bure['FLAG_ONGOING'] = (bure['DAYS_CREDIT_ENDDATE'] > 0).astype('int8')

    bb = pd.read_feather('./data/bb.agg.feather')
    bure = bure.merge(bb, on='SK_ID_BUREAU', how='left')

    bure.drop(['SK_ID_BUREAU'], axis=1)
    reduce_memory(bure)
    return bure


def main():
    bure = preprocess_bureau()
    bure.to_feather('data/bureau.preprocessed.feather')


if __name__ == '__main__':
    main()
