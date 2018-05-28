import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import roc_auc_score
pd.set_option("display.max_columns", 100)


def join_pos_df(df, test_df, orig_pos_df, features):
    prefix = 'pos'
    for recent in [
        0,
        1,
        12,
        10*12,
    ]:
        pos_df = orig_pos_df[orig_pos_df['MONTHS_BALANCE'] >= -recent]
        grp = pos_df.groupby('SK_ID_CURR')
        for agg, columns in [
            [
                'count', []
            ],
            [
                'mean', [
                    'CNT_INSTALMENT',  # Term of previous credit (can change over time) # noqa
                    'CNT_INSTALMENT_FUTURE',  # Installments left to pay on the previous credit # noqa
                    'SK_DPD',  # (days past due) during the month of previous credit # noqa
                    'SK_DPD_DEF',  # DPD during the month with tolerance (debts with low loan amounts are ignored) of the previous credit # noqa
                ],
            ],
            [
                'max', [
                    'CNT_INSTALMENT',  # Term of previous credit (can change over time) # noqa
                    'CNT_INSTALMENT_FUTURE',  # Installments left to pay on the previous credit # noqa
                    'SK_DPD',  # (days past due) during the month of previous credit # noqa
                    'SK_DPD_DEF',  # DPD during the month with tolerance (debts with low loan amounts are ignored) of the previous credit # noqa
                ],
            ],
            [
                'min', [
                    'CNT_INSTALMENT',  # Term of previous credit (can change over time) # noqa
                    'CNT_INSTALMENT_FUTURE',  # Installments left to pay on the previous credit # noqa
                    'SK_DPD',  # (days past due) during the month of previous credit # noqa
                    'SK_DPD_DEF',  # DPD during the month with tolerance (debts with low loan amounts are ignored) of the previous credit # noqa
                ],
            ],
        ]:
            if agg == 'count':
                g = grp[['SK_ID_PREV']].count()
            elif agg == 'mean':
                g = grp[columns].mean()
            elif agg == 'min':
                g = grp[columns].min()
            elif agg == 'max':
                g = grp[columns].max()
            else:
                raise RuntimeError('agg is invalid {}'.format(agg))

            if agg == 'count':
                columns = ['{}_recent_{}_COUNT'.format(prefix, recent)]
            else:
                columns = ['{}_recent_{}_{}_{}'.format(
                    prefix, recent, c, agg) for c in columns]
            g.columns = columns
            features += columns
            g = g.reset_index()
            df = df.merge(g, on='SK_ID_CURR', how='left')
            test_df = test_df.merge(g, on='SK_ID_CURR', how='left')

        # categorical
        for f in ['NAME_CONTRACT_STATUS']:
            g = pos_df.groupby(['SK_ID_CURR', f])['SK_ID_PREV'].count()
            g = g.unstack(1)
            columns = ['{}_recent_{}_{}_count'.format(
                prefix, recent, c) for c in g.columns]
            g.columns = columns
            features += columns
            g = g.reset_index()
            df = df.merge(g, on='SK_ID_CURR', how='left')
            test_df = test_df.merge(g, on='SK_ID_CURR', how='left')

    return df, test_df, features


def join_bure_df(df, test_df, bure_df, orig_bbal_df, features):
    # balance
    bbal_features = []
    for recent in [
        0,
        1,
        12,
        12*10,
    ]:
        bbal_df = orig_bbal_df[orig_bbal_df['MONTHS_BALANCE'] >= -recent]
        g = bbal_df.groupby(['SK_ID_BUREAU'])[['MONTHS_BALANCE']].count()
        columns = ['recent_{}_bbal_count'.format(recent)]
        g.columns = columns
        bbal_features += columns
        bure_df = bure_df.merge(g, on='SK_ID_BUREAU', how='left')

        g = bbal_df.groupby(
            ['SK_ID_BUREAU', 'STATUS'])['MONTHS_BALANCE'].count()
        g = g.unstack(1)
        columns = ['recent_{}_bbal_STATUS_{}_count'.format(
            recent, c) for c in g.columns]
        g.columns = columns
        bbal_features += columns
        bure_df = bure_df.merge(g, on='SK_ID_BUREAU', how='left')

    # bureau
    grp = bure_df.groupby('SK_ID_CURR')
    for agg, columns in {
        'count': [],
        'mean': [
            'DAYS_CREDIT',  # How many days before current application did client apply for Credit Bureau credit,time only relative to the application  # noqa
            'DAYS_CREDIT_ENDDATE',  # Remaining duration of CB credit (in days) at the time of application in Home Credit,time only relative to the application # noqa
            'DAYS_ENDDATE_FACT',  # Days since CB credit ended at the time of application in Home Credit (only for closed credit),time only relative to the application # noqa
            'DAYS_CREDIT_UPDATE',  # How many days before loan application did last information about the Credit Bureau credit come,time only relative to the application # noqa
            'CREDIT_DAY_OVERDUE',  # Number of days past due on CB credit at the time of application for related loan in our sample # noqa
            'AMT_CREDIT_SUM',  # Current credit amount for the Credit Bureau credit  # noqa
            'AMT_CREDIT_SUM_DEBT',  # Current debt on Credit Bureau credit
            'AMT_ANNUITY',  # Annuity of the Credit Bureau credit,
            'AMT_CREDIT_MAX_OVERDUE',  # Maximal amount overdue on the Credit Bureau credit so far (at application date of loan in our sample), # noqa
        ],
        'sum': [
            'AMT_CREDIT_SUM_DEBT',  # Current debt on Credit Bureau credit
            'AMT_CREDIT_MAX_OVERDUE',  # Maximal amount overdue on the Credit Bureau credit so far (at application date of loan in our sample), # noqa
        ] + bbal_features,
        'max': [
            'DAYS_CREDIT',
            'DAYS_CREDIT_ENDDATE',  # Remaining duration of CB credit (in days) at the time of application in Home Credit,time only relative to the application # noqa
            'DAYS_ENDDATE_FACT',  # Days since CB credit ended at the time of application in Home Credit (only for closed credit),time only relative to the application # noqa
            'DAYS_CREDIT_UPDATE',  # How many days before loan application did last information about the Credit Bureau credit come,time only relative to the application # noqa
            'CREDIT_DAY_OVERDUE',  # Number of days past due on CB credit at the time of application for related loan in our sample # noqa
            'AMT_CREDIT_SUM',  # Current credit amount for the Credit Bureau credit  # noqa
            'AMT_ANNUITY',  # Annuity of the Credit Bureau credit,
            'AMT_CREDIT_MAX_OVERDUE',  # Maximal amount overdue on the Credit Bureau credit so far (at application date of loan in our sample), # noqa
        ],
        'min': [
            'DAYS_CREDIT',
            'DAYS_CREDIT_ENDDATE',  # Remaining duration of CB credit (in days) at the time of application in Home Credit,time only relative to the application # noqa
            'DAYS_ENDDATE_FACT',  # Days since CB credit ended at the time of application in Home Credit (only for closed credit),time only relative to the application # noqa
            'DAYS_CREDIT_UPDATE',  # How many days before loan application did last information about the Credit Bureau credit come,time only relative to the application # noqa
            'CREDIT_DAY_OVERDUE',  # Number of days past due on CB credit at the time of application for related loan in our sample # noqa
            'AMT_CREDIT_SUM',  # Current credit amount for the Credit Bureau credit  # noqa
            'AMT_ANNUITY',  # Annuity of the Credit Bureau credit,
            'AMT_CREDIT_MAX_OVERDUE',  # Maximal amount overdue on the Credit Bureau credit so far (at application date of loan in our sample), # noqa
        ],
    }.items():
        if agg == 'count':
            # description is wrong...
            g = grp[['SK_ID_BUREAU']].count()
        elif agg == 'mean':
            g = grp[columns].mean()
        elif agg == 'max':
            g = grp[columns].max()
        elif agg == 'min':
            g = grp[columns].min()
        elif agg == 'sum':
            g = grp[columns].sum()
        else:
            raise RuntimeError('agg is invalid {}'.format(agg))

        if agg == 'count':
            columns = ['bureau_COUNT']
        else:
            columns = ['bureau_{}_{}'.format(c, agg) for c in columns]
        g.columns = columns
        features += columns
        g = g.reset_index()
        df = df.merge(g, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(g, on='SK_ID_CURR', how='left')

    # categorical
    for f in [
        'CREDIT_ACTIVE',  # Status of the Credit Bureau (CB) reported credits
        'CREDIT_CURRENCY',  # Recoded currency of the Credit Bureau credit,recoded # noqa
    ]:
        g = bure_df.groupby(['SK_ID_CURR', f])['SK_ID_BUREAU'].count()
        g = g.unstack(1)
        columns = ['bureau_{}_{}_count'.format(f, c) for c in g.columns]
        g.columns = columns
        features += columns
        g = g.reset_index()
        df = df.merge(g, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(g, on='SK_ID_CURR', how='left')

    return df, test_df, features


def join_credit_df(df, test_df, credit_df, features):
    # TODO: recent credit
    grp = credit_df.groupby('SK_ID_CURR')
    for agg, columns in [
        [
            'mean', [
                'SK_DPD',
                'SK_DPD_DEF',
                'CNT_DRAWINGS_ATM_CURRENT',  # Number of drawings at ATM during this month on the previous credit # noqa
                'CNT_DRAWINGS_CURRENT',  # Number of drawings during this month on the previous credit # noqa
                'CNT_DRAWINGS_OTHER_CURRENT',  # Number of other drawings during this month on the previous credit # noqa
                'CNT_DRAWINGS_POS_CURRENT',  # Number of drawings for goods during this month on the previous credit # noqa
            ],
        ],
    ]:
        if agg == 'mean':
            g = grp[columns].mean()
        else:
            raise RuntimeError('agg is invalid {}'.format(agg))
        columns = ['credit_{}_{}'.format(c, agg) for c in columns]
        g.columns = columns
        features += columns
        g = g.reset_index()
        df = df.merge(g, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(g, on='SK_ID_CURR', how='left')

    # categorical
    for f in ['NAME_CONTRACT_STATUS']:
        g = credit_df.groupby(['SK_ID_CURR', f])['SK_ID_PREV'].count()
        g = g.unstack(1)
        columns = ['credit_{}_{}_count'.format(f, c) for c in g.columns]
        g.columns = columns
        features += columns
        g = g.reset_index()
        df = df.merge(g, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(g, on='SK_ID_CURR', how='left')

    return df, test_df, features


def join_prev_df(df, test_df, prev_df, features):
    # TODO: increase annuity?
    # TODO: recent application
    mapping = {
        'POS mobile with interest': 'POS',
        'POS household without interest': 'POS',
        'POS household with interest': 'POS',
        'POS other with interest': 'POS',
        'POS mobile without interest': 'POS',
        'POS industry with interest': 'POS',
        'POS industry without interest': 'POS',
        'POS others without interest': 'POS',
        'Cash X-Sell: low': 'Cash',
        'Cash X-Sell: high': 'Cash',
        'Cash X-Sell: middle': 'Cash',
        'Cash Street: high': 'Cash',
        'Cash Street: middle'
        'Cash Street: low': 'Cash',
        'Cash': 'Cash',
        'Card Street': 'Card',
        'Card X-Sell': 'Card',
    }
    prev_df['PRODUCT_COMBINATION_PREFIX'] = prev_df[
        'PRODUCT_COMBINATION'].map(mapping)
    grp = prev_df.groupby('SK_ID_CURR')
    for agg, columns in [
        [
            'mean', [
                'AMT_ANNUITY',  # Annuity of the Credit Bureau credit,
                'AMT_APPLICATION',  # For how much credit did client ask on the previous application, # noqa
                'AMT_CREDIT',  # "Final credit amount on the previous application. This differs from AMT_APPLICATION in a way that the AMT_APPLICATION is the amount for which the client initially applied for, but during our e received different amount - AMT_CREDIT", # noqa
                'AMT_DOWN_PAYMENT',  # Down payment on the previous application, # noqa
                'AMT_GOODS_PRICE',  # Goods price of good that client asked for (if applicable) on the previous application, # noqa
                'RATE_DOWN_PAYMENT',  # Down payment rate normalized on previous credit,normalized # noqa
                'RATE_INTEREST_PRIMARY',  # Interest rate normalized on previous credit,normalized # noqa
                'RATE_INTEREST_PRIVILEGED',  # Interest rate normalized on previous credit,normalized # noqa
                'DAYS_DECISION',  # Relative to current application when was the decision about previous application made,time only relative to the application # noqa
                'CNT_PAYMENT',  # Term of previous credit at application of the previous application, # noqa
                'DAYS_FIRST_DRAWING',  # Relative to application date of current application when was the first disbursement of the previous application,time only relative to the application # noqa
                'DAYS_FIRST_DUE',  # Relative to application date of current application when was the first due supposed to be of the previous application,time only relative to the application # noqa
                'DAYS_LAST_DUE_1ST_VERSION',  # Relative to application date of current application when was the first due of the previous application,time only relative to the application # noqa
                'DAYS_LAST_DUE',  # Relative to application date of current application when was the last due date of the previous application,time only relative to the application # noqa
                'DAYS_TERMINATION',  # Relative to application date of current application when was the expected termination of the previous application,time only relative to the application # noqa
                # encoded
                'ENCODED_NAME_CONTRACT_TYPE',
                'ENCODED_WEEKDAY_APPR_PROCESS_START',  # On which day of the week did the client apply for previous application, # noqa
                'ENCODED_HOUR_APPR_PROCESS_START',  # Approximately at what day hour did the client apply for the previous application,rounded # noqa
                'ENCODED_FLAG_LAST_APPL_PER_CONTRACT',  # Flag if it was last application for the previous contract. Sometimes by mistake of client or our clerk there could be more applications for one single contract, # noqa
                'ENCODED_NFLAG_LAST_APPL_IN_DAY',  # Flag if the application was the last application per day of the client. Sometimes clients apply for more applications a day. Rarely it could also be error in our system that one application is in the database twice, # noqa
                # 'ENCODED_NFLAG_MICRO_CASH',  # Flag Micro finance loan, # missing? # noqa
                'ENCODED_NAME_CASH_LOAN_PURPOSE',  # Purpose of the cash loan, # noqa
                'ENCODED_NAME_CONTRACT_STATUS',
                'ENCODED_NAME_PAYMENT_TYPE',  # Payment method that client chose to pay for the previous application, # noqa
                'ENCODED_CODE_REJECT_REASON',  # Why was the previous application rejected, # noqa
                'ENCODED_NAME_TYPE_SUITE',  # Who accompanied client when applying for the previous application, # noqa
                'ENCODED_NAME_CLIENT_TYPE',  # Was the client old or new client when applying for the previous application, # noqa
                'ENCODED_NAME_GOODS_CATEGORY',  # What kind of goods did the client apply for in the previous application, # noqa
                'ENCODED_NAME_PORTFOLIO',  # "Was the previous application for CASH, POS, CAR, <85>", # noqa
                'ENCODED_NAME_PRODUCT_TYPE',  # Was the previous application x-sell o walk-in, # noqa
                'ENCODED_CHANNEL_TYPE',  # Through which channel we acquired the client on the previous application, # noqa
                'ENCODED_SELLERPLACE_AREA',  # Selling area of seller place of the previous application, # encoded # noqa
                'ENCODED_NAME_SELLER_INDUSTRY',  # The industry of the seller, # noqa
                'ENCODED_NAME_YIELD_GROUP',  # Grouped interest rate into small medium and high of the previous application,grouped # noqa
                'ENCODED_PRODUCT_COMBINATION',  # Detailed product combination of the previous application, # noqa
                'ENCODED_NFLAG_INSURED_ON_APPROVAL',  # Did the client requested insurance during the previous application, # noqa
            ],
        ],
    ]:
        if agg == 'mean':
            g = grp[columns].mean()
        else:
            raise RuntimeError('agg is invalid {}'.format(agg))
        columns = ['prev_{}_{}'.format(c, agg) for c in columns]
        g.columns = columns
        features += columns
        g = g.reset_index()
        df = df.merge(g, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(g, on='SK_ID_CURR', how='left')

    # categorical
    for f in [
        # 'NAME_CONTRACT_TYPE',
        # 'WEEKDAY_APPR_PROCESS_START',  # On which day of the week did the client apply for previous application, # noqa
        # 'HOUR_APPR_PROCESS_START',  # Approximately at what day hour did the client apply for the previous application,rounded # noqa
        # 'FLAG_LAST_APPL_PER_CONTRACT',  # Flag if it was last application for the previous contract. Sometimes by mistake of client or our clerk there could be more applications for one single contract, # noqa
        # 'NFLAG_LAST_APPL_IN_DAY',  # Flag if the application was the last application per day of the client. Sometimes clients apply for more applications a day. Rarely it could also be error in our system that one application is in the database twice, # noqa
        # # 'NFLAG_MICRO_CASH',  # Flag Micro finance loan, # missing? # noqa
        # 'NAME_CASH_LOAN_PURPOSE',  # Purpose of the cash loan, # noqa
        # 'NAME_CONTRACT_STATUS',
        # 'NAME_PAYMENT_TYPE',  # Payment method that client chose to pay for the previous application, # noqa
        # 'CODE_REJECT_REASON',  # Why was the previous application rejected, # noqa
        # 'NAME_TYPE_SUITE',  # Who accompanied client when applying for the previous application, # noqa
        # 'NAME_CLIENT_TYPE',  # Was the client old or new client when applying for the previous application, # noqa
        # 'NAME_GOODS_CATEGORY',  # What kind of goods did the client apply for in the previous application, # noqa
        # 'NAME_PORTFOLIO',  # "Was the previous application for CASH, POS, CAR, <85>", # noqa
        # 'NAME_PRODUCT_TYPE',  # Was the previous application x-sell o walk-in, # noqa
        # 'CHANNEL_TYPE',  # Through which channel we acquired the client on the previous application, # noqa
        # # 'SELLERPLACE_AREA',  # Selling area of seller place of the previous application, # encoded # noqa
        # 'NAME_SELLER_INDUSTRY',  # The industry of the seller, # noqa
        # 'NAME_YIELD_GROUP',  # Grouped interest rate into small medium and high of the previous application,grouped # noqa
        # 'PRODUCT_COMBINATION',  # Detailed product combination of the previous application, # noqa
        # 'NFLAG_INSURED_ON_APPROVAL',  # Did the client requested insurance during the previous application, # noqa
        'PRODUCT_COMBINATION_PREFIX',
    ]:
        g = prev_df.groupby(['SK_ID_CURR', f])['SK_ID_PREV'].count()
        g = g.unstack(1)
        columns = ['prev_{}_{}_count'.format(f, c) for c in g.columns]
        g.columns = columns
        features += columns
        g = g.reset_index()
        df = df.merge(g, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(g, on='SK_ID_CURR', how='left')

    return df, test_df, features


def join_inst_df(df, test_df, inst_df, features):
    grp = inst_df.groupby('SK_ID_CURR')
    for agg, columns in [
        [
            'mean', [
                'NUM_INSTALMENT_NUMBER',  # On which installment we observe payment,  # noqa
                'DAYS_INSTALMENT',  # When the installment of previous credit was supposed to be paid (relative to application date of current loan),time only relative to the application  # noqa
                'DAYS_ENTRY_PAYMENT',  # When was the installments of previous credit paid actually (relative to application date of current loan),time only relative to the application  # noqa
                'AMT_INSTALMENT',  # What was the prescribed installment amount of previous credit on this installment,  # noqa
                'AMT_PAYMENT',  # What the client actually paid on previous credit on this installment,  # noqa
            ],
        ],
        [
            'nunique', [
                'NUM_INSTALMENT_VERSION',  # Version of installment calendar (0 is for credit card) of previous credit. Change of installment version from month to month signifies that some parameter of payment calendar has changed,  # noqa
            ],
        ],
    ]:
        if agg == 'mean':
            g = grp[columns].mean()
        elif agg == 'nunique':
            g = grp[columns].nunique()
        else:
            raise RuntimeError('agg is invalid {}'.format(agg))
        columns = ['inst_{}_{}'.format(c, agg) for c in columns]
        g.columns = columns
        features += columns
        g = g.reset_index()
        df = df.merge(g, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(g, on='SK_ID_CURR', how='left')

    # categorical
    for f in []:
        g = inst_df.groupby(['SK_ID_CURR', f])['SK_ID_PREV'].count()
        g = g.unstack(1)
        columns = ['prev_{}_{}_count'.format(f, c) for c in g.columns]
        g.columns = columns
        features += columns
        g = g.reset_index()
        df = df.merge(g, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(g, on='SK_ID_CURR', how='left')

    return df, test_df, features


def train(
    df, test_df, pos_df, credit_df, prev_df, inst_df,
    bure_df, bbal_df,
    validate, importance_summay,
):
    # filter by sample id
    sk_id_curr = pd.concat([df['SK_ID_CURR'], test_df['SK_ID_CURR']]).unique()
    pos_df = pos_df[pos_df['SK_ID_CURR'].isin(sk_id_curr)]
    credit_df = credit_df[credit_df['SK_ID_CURR'].isin(sk_id_curr)]
    prev_df = prev_df[prev_df['SK_ID_CURR'].isin(sk_id_curr)]
    inst_df = inst_df[inst_df['SK_ID_CURR'].isin(sk_id_curr)]

    bure_df = bure_df[bure_df['SK_ID_CURR'].isin(sk_id_curr)]
    sk_id_bure = bure_df['SK_ID_BUREAU'].unique()
    bbal_df = bbal_df[bbal_df['SK_ID_BUREAU'].isin(sk_id_bure)]

    features = [
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',
        'CNT_CHILDREN',
        'AMT_INCOME_TOTAL',
        'AMT_CREDIT',
        'AMT_ANNUITY',
        'AMT_GOODS_PRICE',  # For consumer loans it is the price of the goods for which the loan is given  # noqa
        'REGION_POPULATION_RELATIVE',
        'DAYS_BIRTH',   # Client's age in days at the time of application,time only relative to the application  # noqa
        'DAYS_EMPLOYED',  # How many days before the application the person started current employment,time only relative to the application  # noqa
        'DAYS_REGISTRATION',
        'DAYS_ID_PUBLISH',   # How many days before the application did client change the identity document with which he applied for the loan,time only relative to the application # noqa
        'CNT_FAM_MEMBERS',
        'REGION_RATING_CLIENT',
        'REGION_RATING_CLIENT_W_CITY',
        'APARTMENTS_AVG',
        'BASEMENTAREA_AVG',
        'YEARS_BEGINEXPLUATATION_AVG',
        'YEARS_BUILD_AVG',
        'COMMONAREA_AVG',
        'ELEVATORS_AVG',
        'ENTRANCES_AVG',
        'FLOORSMAX_AVG',
        'FLOORSMIN_AVG',
        'LANDAREA_AVG',
        'LIVINGAPARTMENTS_AVG',
        'LIVINGAREA_AVG',
        'NONLIVINGAPARTMENTS_AVG',
        'NONLIVINGAREA_AVG',
        'APARTMENTS_MODE',
        'BASEMENTAREA_MODE',
        'YEARS_BEGINEXPLUATATION_MODE',
        'YEARS_BUILD_MODE',
        'COMMONAREA_MODE',
        'ELEVATORS_MODE',
        'ENTRANCES_MODE',
        'FLOORSMAX_MODE',
        'FLOORSMIN_MODE',
        'LANDAREA_MODE',
        'LIVINGAPARTMENTS_MODE',
        'LIVINGAREA_MODE',
        'NONLIVINGAPARTMENTS_MODE',
        'NONLIVINGAREA_MODE',
        'APARTMENTS_MEDI',
        'BASEMENTAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MEDI',
        'YEARS_BUILD_MEDI',
        'COMMONAREA_MEDI',
        'ELEVATORS_MEDI',
        'ENTRANCES_MEDI',
        'FLOORSMAX_MEDI',
        'FLOORSMIN_MEDI',
        'LANDAREA_MEDI',
        'LIVINGAPARTMENTS_MEDI',
        'LIVINGAREA_MEDI',
        'NONLIVINGAPARTMENTS_MEDI',
        'NONLIVINGAREA_MEDI',
        'TOTALAREA_MODE',
        'OBS_30_CNT_SOCIAL_CIRCLE',
        'DEF_30_CNT_SOCIAL_CIRCLE',
        'OBS_60_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE',
        'DAYS_LAST_PHONE_CHANGE',
        'OWN_CAR_AGE',  # Age of client's car,
        # encoded
        'ENCODED_CODE_GENDER',
        'ENCODED_FLAG_OWN_CAR',
        'ENCODED_FLAG_OWN_REALTY',
        'ENCODED_NAME_TYPE_SUITE',
        'ENCODED_NAME_INCOME_TYPE',
        'ENCODED_NAME_EDUCATION_TYPE',
        'ENCODED_NAME_FAMILY_STATUS',
        'ENCODED_NAME_HOUSING_TYPE',
        'ENCODED_FLAG_MOBIL',
        'ENCODED_FLAG_EMP_PHONE',
        'ENCODED_FLAG_WORK_PHONE',
        'ENCODED_FLAG_CONT_MOBILE',
        'ENCODED_FLAG_PHONE',
        'ENCODED_FLAG_EMAIL',
        'ENCODED_OCCUPATION_TYPE',
        'ENCODED_WEEKDAY_APPR_PROCESS_START',
        'ENCODED_HOUR_APPR_PROCESS_START',
        'ENCODED_REG_REGION_NOT_LIVE_REGION',
        'ENCODED_REG_REGION_NOT_WORK_REGION',
        'ENCODED_LIVE_REGION_NOT_WORK_REGION',
        'ENCODED_REG_CITY_NOT_LIVE_CITY',
        'ENCODED_REG_CITY_NOT_WORK_CITY',
        'ENCODED_LIVE_CITY_NOT_WORK_CITY',
        'ENCODED_ORGANIZATION_TYPE',
        'ENCODED_FONDKAPREMONT_MODE',
        'ENCODED_HOUSETYPE_MODE',
        'ENCODED_WALLSMATERIAL_MODE',
        'ENCODED_EMERGENCYSTATE_MODE',
        'ENCODED_NAME_CONTRACT_TYPE',
    ]

    cat_feature = [
        # 'CODE_GENDER',   # Gender of the client
        # 'FLAG_OWN_CAR',
        # 'FLAG_OWN_REALTY',
        # 'NAME_TYPE_SUITE',
        # 'NAME_INCOME_TYPE',
        # 'NAME_EDUCATION_TYPE',  # Level of highest education the client achieved,  # noqa
        # 'NAME_FAMILY_STATUS',
        # 'NAME_HOUSING_TYPE',
        # 'FLAG_MOBIL',
        # 'FLAG_EMP_PHONE',
        # 'FLAG_WORK_PHONE',
        # 'FLAG_CONT_MOBILE',
        # 'FLAG_PHONE',
        # 'FLAG_EMAIL',
        # 'OCCUPATION_TYPE',
        # 'WEEKDAY_APPR_PROCESS_START',
        # 'HOUR_APPR_PROCESS_START',
        # 'REG_REGION_NOT_LIVE_REGION',
        # 'REG_REGION_NOT_WORK_REGION',
        # 'LIVE_REGION_NOT_WORK_REGION',
        # 'REG_CITY_NOT_LIVE_CITY',
        # 'REG_CITY_NOT_WORK_CITY',
        # 'LIVE_CITY_NOT_WORK_CITY',
        # # 'ORGANIZATION_TYPE',
        # 'FONDKAPREMONT_MODE',
        # 'HOUSETYPE_MODE',
        # 'WALLSMATERIAL_MODE',
        # 'EMERGENCYSTATE_MODE',
        # 'NAME_CONTRACT_TYPE',  # Identification if loan is cash or revolving,
    ]

    # POS
    df, test_df, features = join_pos_df(df, test_df, pos_df, features)

    # credit bureau
    df, test_df, features = join_bure_df(
        df, test_df, bure_df, bbal_df, features)

    # credit card
    df, test_df, features = join_credit_df(df, test_df, credit_df, features)

    # prev_df
    df, test_df, features = join_prev_df(df, test_df, prev_df, features)

    # inst_df
    df, test_df, features = join_inst_df(df, test_df, inst_df, features)

    # cat features
    if len(cat_feature) > 0:
        n_train = len(df)
        df = pd.concat([df, test_df]).reset_index(drop=True)
        df[cat_feature] = df[cat_feature].fillna('NaN')
        df[cat_feature] = df[cat_feature].astype('category')
        df[cat_feature] = df[cat_feature].apply(lambda x: x.cat.codes)
        test_df = df[n_train:].reset_index(drop=True)
        df = df[:n_train].reset_index(drop=True)

    # train
    if validate:
        n_train = len(df)
        train_df = df
        valid_df = test_df
    else:
        n_train = int(len(df) * 0.85)
        train_df = df[:n_train]
        valid_df = df[n_train:]

    features += cat_feature
    xgtrain = lgb.Dataset(
        train_df[features].values, label=train_df['TARGET'].values,
        feature_name=features,
        categorical_feature=cat_feature,
    )
    xgvalid = lgb.Dataset(
        valid_df[features].values, label=valid_df['TARGET'].values,
        feature_name=features,
        categorical_feature=cat_feature,
    )
    evals_result = {}
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 15,
        'max_depth': -1,  # -1 means no limit
        'min_data_in_leaf': 40,
        'max_bin': 64,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'subsample_for_bin': 10000000,
        'min_split_gain': 0.01,
        'reg_alpha': 0.001,
        'reg_lambda': 0.001,
        'nthread': 12,
        'verbose': 0,
    }
    bst = lgb.train(
        lgb_params,
        xgtrain,
        valid_sets=[xgtrain, xgvalid],
        valid_names=['train', 'valid'],
        evals_result=evals_result,
        num_boost_round=2000,
        early_stopping_rounds=100,
        verbose_eval=50,
        categorical_feature=cat_feature,
        # feval=feval,
    )

    print("\nModel Report")
    print("bst1.best_iteration: ", bst.best_iteration)
    print("auc:", evals_result['valid']['auc'][bst.best_iteration-1])

    importance = bst.feature_importance(iteration=bst.best_iteration)
    feature_name = bst.feature_name()

    for key, value in zip(feature_name, importance):
        importance_summay[key] += value / sum(importance)

    return bst.predict(test_df[features], bst.best_iteration)


def split(df):
    pos_df = df[df['TARGET'] == 1].sample(frac=1)
    neg_df = df[df['TARGET'] == 0].sample(frac=1)
    n_pos = pos_df.shape[0]
    n_neg = neg_df.shape[0]
    n_pos_train = int(0.85*n_pos)
    n_neg_train = int(0.85*n_neg)
    train_df = pd.concat([pos_df[:n_pos_train], neg_df[:n_neg_train]])
    train_df = train_df.sample(frac=1).reset_index()
    test_df = pd.concat([pos_df[n_pos_train:], neg_df[n_neg_train:]])
    test_df = test_df.sample(frac=1).reset_index()
    return train_df, test_df


def main():
    np.random.seed(215)
    now = datetime.now().strftime('%m%d-%H%M')
    validate = True
    print('validate: {}'.format(validate))
    print('load data...')
    df = pd.read_feather('./data/application_train.csv.encoded.feather')
    print('n_train: {}'.format(len(df)))
    pos_df = pd.read_feather('./data/POS_CASH_balance.csv.feather')
    credit_df = pd.read_feather('./data/credit_card_balance.csv.feather')
    prev_df = pd.read_feather(
        './data/previous_application.csv.encoded.feather')
    inst_df = pd.read_feather('./data/installments_payments.csv.feather')

    # bureau
    bure_df = pd.read_feather('./data/bureau.csv.feather')
    bbal_df = pd.read_feather('./data/bureau_balance.csv.feather')

    if validate:
        n_bagging = 5
    else:
        n_bagging = 5
        test_df = pd.read_feather(
            './data/application_test.csv.encoded.feather')
        print('n_test: {}'.format(len(test_df)))

    importance_summay = defaultdict(lambda: 0)
    auc_summary = []

    for i in range(n_bagging):
        if validate:
            train_df, test_df = split(df)
        else:
            train_df = df
        pos_train_df = train_df[train_df['TARGET'] == 1]
        neg_train_df = train_df[train_df['TARGET'] == 0]
        n_pos = len(pos_train_df)
        print('n_pos: {}'.format(n_pos))
        neg_part_train_df = neg_train_df.sample(n=n_pos)
        part_df = pd.concat([pos_train_df, neg_part_train_df])
        part_df = part_df.sample(frac=1)

        test_df['PRED_{}'.format(i)] = train(
            part_df, test_df, pos_df, credit_df, prev_df, inst_df,
            bure_df, bbal_df,
            validate, importance_summay,
        )
        if validate:
            auc_summary.append(roc_auc_score(
                test_df['TARGET'], test_df['PRED_{}'.format(i)]
            ))

    auc_summary = np.array(auc_summary)

    for key, value in sorted(importance_summay.items(), key=lambda x: -x[1]):
        print('{} {}'.format(key, value))

    if validate:
        print('validate auc: {} +- {}'.format(
            auc_summary.mean(), auc_summary.std()))
    else:
        test_df['TARGET'] = 0
        for i in range(n_bagging):
            test_df['TARGET'] += test_df['PRED_{}'.format(i)]
        test_df['TARGET'] /= test_df['TARGET'].max()
        test_df = test_df[['SK_ID_CURR', 'TARGET']]
        os.makedirs('./output', exist_ok=True)
        path = os.path.join('./output', '{}.csv.gz'.format(now))
        test_df.to_csv(path, index=False, compression='gzip')


if __name__ == '__main__':
    main()
