import pandas as pd
import numpy as np
import gc
import pickle
import matplotlib.pylab as plt
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
import joblib
import warnings
warnings.filterwarnings("ignore")
gc.enable()

### Generate dataset ####
def difference(groups,num_features,shift):
    data=(groups[num_features].nth(-1)-groups[num_features].nth(-1*shift)).rename(columns={f: f"{f}_diff{shift-1}" for f in num_features})
    return data
def get_difference(data,num_features, lag):
    groups=data.groupby('customer_ID')
    df1=difference(groups,num_features,lag)
    df1 = df1.reset_index(drop=False)
    return df1

def get_skew(data, num_features):
    df1 = []
    customer_ids = []
    groups = data.groupby('customer_ID')
    for customer_id, df in tqdm(groups):
        diff_df1 = df[num_features].skew().values.astype(np.float32)
        df1.append(np.expand_dims(diff_df1, axis=0))
        customer_ids.append(customer_id)
    df1 = np.concatenate(df1, axis = 0)
    df1 = pd.DataFrame(df1, columns = [col + f'_skew' for col in df[num_features].columns])
    df1['customer_ID'] = customer_ids
    return df1

def get_moving_average(data, num_features):

    groups = data.groupby('customer_ID')
    df1 = ( groups[num_features].nth(-1) +
            groups[num_features].nth(-2) +
            groups[num_features].nth(-3)).rename(columns={f: f"{f}_ma3" for f in num_features})

    df1[df1.columns] = df1[df1.columns].values/3
    df1 = df1.reset_index(drop=False)

    return df1

def get_ewm(data, num_features, alpha=2/3):
    df1 = []
    customer_ids = []
    groups = data.groupby('customer_ID')
    for customer_id, df in tqdm(groups):
        diff_df1 = df[num_features].ewm(alpha=alpha).mean().iloc[[-1]].values.astype(np.float32)
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    df1 = np.concatenate(df1, axis=0)
    df1 = pd.DataFrame(df1, columns=[col + '_ewm' for col in df[num_features].columns])
    df1['customer_ID'] = customer_ids
    return df1

def get_mode(data, cat_cols):
    df1 = []
    customer_ids = []
    groups = data.groupby('customer_ID')
    for customer_id, df in tqdm(groups):
        diff_df1 = df[cat_cols].mode(dropna=False)
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    df1 = np.concatenate(df1, axis=0)
    df1 = pd.DataFrame(df1, columns=[col + '_mode' for col in df[cat_cols].columns])
    df1['customer_ID'] = customer_ids
    return df1

def preprocess_cat_data(train, cat_cols):

    print('Categorical variables...')
    train_cat_agg = train.groupby("customer_ID")[cat_cols].agg(['last', 'first', 'median', 'nunique'])
    train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns]
    train_cat_agg = train_cat_agg.reset_index(drop=False)

    # Previous features
    temp = train.groupby(by=['customer_ID'])[cat_cols].nth(-2)
    temp.columns = [f'{x}_lag' for x in cat_cols]
    temp = temp.reset_index(drop=False)
    train_cat_agg = train_cat_agg.merge(temp, on=['customer_ID'], how='left', copy=False)

    del temp
    gc.collect()

    # Previous features
    temp = train.groupby(by=['customer_ID'])[cat_cols].nth(-3)
    temp.columns = [f'{x}_lag2' for x in cat_cols]
    temp = temp.reset_index(drop=False)
    train_cat_agg = train_cat_agg.merge(temp, on=['customer_ID'], how='left', copy=False)

    del temp
    gc.collect()

    # # Previous features
    # df_mode = get_mode(train, cat_cols)
    # train_cat_agg = train_cat_agg.merge(df_mode, on=['customer_ID'], how='left', copy=False)
    #
    # del temp
    # gc.collect()

    # Transform int64 columns to int32
    cols = list(train_cat_agg.dtypes[train_cat_agg.dtypes == 'int64'].index)
    for col in tqdm(cols):
        train_cat_agg[col] = train_cat_agg[col].astype(np.int32)

    return train_cat_agg

def preprocess_num_data(train, num_cols, feature_type='high'):

    print('Numerical variables...')
    if feature_type=='high':
        agg_func = ['last', 'mean']
    elif feature_type=='binary':
        agg_func = ['last', 'first', 'mean']
    elif feature_type == 'ordinal':
        agg_func = ['last', 'first', 'mean', 'max', 'min']
    else:
        agg_func = ['last', 'first', 'mean', 'max', 'min', 'std']

    train_num_agg = train.groupby("customer_ID")[num_cols].agg(agg_func)
    train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
    train_num_agg = train_num_agg.reset_index(drop=False)

    # Transform float64 columns to float32
    cols = list(train_num_agg.dtypes[train_num_agg.dtypes == 'float64'].index)
    for col in tqdm(cols):
        train_num_agg.loc[~train_num_agg[col].isnull(), col] = train_num_agg.loc[~train_num_agg[col].isnull(), col].astype(np.float32)

    if feature_type in ['ordinal', 'normal']:
        for col in train_num_agg:
            if 'last' in col and col.replace('last', 'first') in train_num_agg:
                train_num_agg[col + '_lag_sub'] = (train_num_agg[col] - train_num_agg[col.replace('last', 'first')])
                train_num_agg[col + '_lag_div'] = train_num_agg[col] / (train_num_agg[col.replace('last', 'first')])

        train_diff = get_difference(train, num_cols, lag=2)
        train_num_agg = train_num_agg.merge(train_diff, on=['customer_ID'], how='left', copy=False)

        del train_diff
        gc.collect()

        train_diff = get_difference(train, num_cols, lag=3)
        train_num_agg = train_num_agg.merge(train_diff, on=['customer_ID'], how='left', copy=False)

        del train_diff
        gc.collect()

        # Previous features
        temp = train.groupby(by=['customer_ID'])[num_cols].nth(-2)
        temp.columns = [f'{x}_lag1' for x in num_cols]
        temp = temp.reset_index(drop=False)
        train_num_agg = train_num_agg.merge(temp, on=['customer_ID'], how='left', copy=False)

        del temp
        gc.collect()

        # Previous features
        temp = train.groupby(by=['customer_ID'])[num_cols].nth(-6)
        temp.columns = [f'{x}_lag2' for x in num_cols]
        temp = temp.reset_index(drop=False)
        train_num_agg = train_num_agg.merge(temp, on=['customer_ID'], how='left', copy=False)

        del temp
        gc.collect()

    if feature_type in ['ordinal', 'normal', 'binary']:
        train_rolling = get_moving_average(train, num_cols)
        train_num_agg = train_num_agg.merge(train_rolling, on=['customer_ID'], how='left', copy=False)

        del train_rolling
        gc.collect()

    if feature_type == 'normal':
        train_skew = get_skew(train, num_cols)
        train_num_agg = train_num_agg.merge(train_skew, on=['customer_ID'], how='left', copy=False)

        del train_skew
        gc.collect()

        train_ewm = get_ewm(train, num_cols, alpha= 2/3)
        train_num_agg = train_num_agg.merge(train_ewm, on=['customer_ID'], how='left', copy=False)

        del train_ewm
        gc.collect()

    if feature_type in ['ordinal', 'normal']:
        for col in train_num_agg:
            if 'max' in col and col.replace('max', 'last') in train_num_agg:
                train_num_agg[col + '_last_sub'] = train_num_agg[col] - (train_num_agg[col.replace('max', 'last')])

            # if 'max' in col and col.replace('min', 'max') in train_num_agg:
            #     train_num_agg[col + '_min_sub'] = (train_num_agg[col] / (train_num_agg[col.replace('min', 'max')]))

            if 'last' in col and col.replace('last', 'mean') in train_num_agg:
                train_num_agg[col + '_mean_div'] = train_num_agg[col] / (train_num_agg[col.replace('last', 'mean')])

    return train_num_agg

### Generate dataset ####
def get_month_at_value(data, cols, max_value=True):
    groups = data.groupby('customer_ID')
    df1 = []
    customer_ids = []
    if max_value:
        for customer_id, df in tqdm(groups):
            temp = df[cols].idxmax()
            df1.append([temp.values])
            customer_ids.append(customer_id)
    else:
        for customer_id, df in tqdm(groups):
            temp = df[cols].idxmin()
            df1.append([temp.values])
            customer_ids.append(customer_id)

    df1 = np.concatenate(df1, axis=0)
    df1 = pd.DataFrame(df1, columns=[col for col in df[cols].columns])

    df1['customer_ID'] = customer_ids
    for col in df1.columns[:-1]:
        df1.loc[~df1[col].isnull(), col] = data.loc[df1.loc[~df1[col].isnull(), col].values, 'month'].values

    if max_value:
        df1.columns = [col + '_month_at_max' for col in df[cols].columns] + ['customer_ID']
    else:
        df1.columns = [col + '_month_at_min' for col in df[cols].columns] + ['customer_ID']

    return df1

### Generate dataset ####
def get_month_at_weighted_value(data, cols, max_value=True):
    groups = data.groupby('customer_ID')
    df1 = []
    customer_ids = []
    if max_value:
        for customer_id, df in tqdm(groups):
            temp = df[cols].idxmax()
            df1.append([temp.values])
            customer_ids.append(customer_id)
    else:
        for customer_id, df in tqdm(groups):
            temp = df[cols].idxmin()
            df1.append([temp.values])
            customer_ids.append(customer_id)

    df1 = np.concatenate(df1, axis=0)
    df1 = pd.DataFrame(df1, columns=[col for col in df[cols].columns])

    df1['customer_ID'] = customer_ids
    for col in df1.columns[:-1]:
        df1.loc[~df1[col].isnull(), col] = data.loc[df1.loc[~df1[col].isnull(), col].values, 'month'].values*data.loc[df1.loc[~df1[col].isnull(), col].values, col].values
        df1.loc[~df1[col].isnull(), col] = df1.loc[~df1[col].isnull(), col].values/15

    if max_value:
        df1.columns = [col + '_month_at_max_weighted' for col in df[cols].columns] + ['customer_ID']
    else:
        df1.columns = [col + '_month_at_min_weighted' for col in df[cols].columns] + ['customer_ID']

    return df1



DATA_DIR = 'preprocessed_data/raw_data/'
DATA_DIR2 = 'preprocessed_data/'
DATA_DIR_FINAL = 'preprocessed_data/test_preprocessed/'
SPLIT = False
cat_cols = ["B_30", "B_38","D_114", "D_117", "D_120", "D_126",
            "D_63", "D_64", "D_68"]

drop_cols = ['D_88', 'D_87', 'D_108', 'D_110', 'D_111', 'B_39',
             'D_73', 'B_42', 'B_31', 'D_116', 'D_66', 'D_109', 'R_23',
             'R_28', 'R_22','R_25','D_93','D_137', 'D_94','D_135', 'D_86',
             'D_96']

NPARTS = 2

if SPLIT:
    test = []
    for i in tqdm(range(12)):
        temp = pd.read_pickle(DATA_DIR + f'test_{i}.pkl')
        test.append(temp)

        del temp
        gc.collect()

    gc.collect()

    test = pd.concat(test, axis=0)
    gc.collect()

    cols = test.columns[2:-1]

    for col in tqdm(cols):
        test.loc[test[col] == -999, col] = np.nan
        test.loc[test[col] == -1, col] = np.nan

    test['max_month'] = test.groupby('customer_ID')['month'].transform('max')

    # test['target'] = 0
    # stats = get_stats(test.loc[test['month'] == test['month'].max()], target=False)
    # stats.to_csv('df_summary_test.csv', index=False)

    print(test.loc[test['max_month']==28].shape, test.loc[test['max_month']==34].shape)
    test.loc[test['max_month'] == 28].to_pickle(DATA_DIR_FINAL + 'test_0.pkl')
    test.loc[test['max_month'] == 34].to_pickle(DATA_DIR_FINAL + 'test_1.pkl')


    del test
    gc.collect()


for p in range(NPARTS):

    binary_cols = joblib.load(DATA_DIR2 + 'binary_cols.joblib')
    high_missing_cols = joblib.load(DATA_DIR2 + 'high_missing_cols.joblib')
    cat_cols = joblib.load(DATA_DIR2 + 'cat_cols.joblib')
    ordinal_cols = joblib.load(DATA_DIR2 + 'ordinal_cols.joblib')
    normal_cols = joblib.load(DATA_DIR2 + 'normal_cols.joblib')

    train = pd.read_pickle(DATA_DIR_FINAL + f'test_{p}.pkl')
    train = train.drop(['max_month'], axis=1)
    if p==0:
        train['month'] = train['month'].values - 13
    else:
        train['month'] = train['month'].values - 19

    train.loc[train['B_2'] > 0.8, 'B_2'] = np.round(2 * train.loc[train['B_2'] > 0.8, 'B_2'].values) / 2
    train.loc[train['B_2'] < 0.8, 'B_2'] = np.round(36 * train.loc[train['B_2'] < 0.8, 'B_2'].values) / 36
    train['B_4'] = np.round(100 * train['B_4']) / 100
    train['B_10'] = np.round(50 * train['B_10']) / 50
    train['B_16'] = np.round(13 * train['B_16']) / 13
    train.loc[train['B_17'] > 0.99, 'B_17'] = 1
    train.loc[train['B_17'] < 0.01, 'B_17'] = 0
    train['B_18'] = np.round(15 * train['B_18']) / 15
    train['B_19'] = np.round(30 * train['B_19']) / 30
    train['B_20'] = np.round(18 * train['B_20']) / 18
    train['B_21'] = np.round(12 * train['B_21']) / 12
    train['B_22'] = np.round(4 * train['B_22']) / 4
    train.loc[train['B_25'] > 0.99, 'B_25'] = 1
    train.loc[train['B_25'] < 0.01, 'B_25'] = 0
    train.loc[train['B_29'] < 0.01, 'B_29'] = 0
    train['B_31'] = np.round(2 * train['B_31']) / 2
    train['B_32'] = np.round(2 * train['B_32']) / 2
    train['B_33'] = np.round(2 * train['B_33']) / 2
    train.loc[train['B_36'] > 0.99, 'B_36'] = 1
    train.loc[train['B_36'] < 0.01, 'B_36'] = 0
    train.loc[train['B_39'] > 0.99, 'B_39'] = 1
    train.loc[train['B_39'] < 0.01, 'B_39'] = 0
    train['B_41'] = np.round(7 * train['B_41']) / 7
    train['B_8'] = np.round(2 * train['B_8']) / 2

    train['D_39'] = np.round(34 * train['D_39']) / 34
    train.loc[train['B_41'] < 0.01, 'B_41'] = 0
    train['D_44'] = np.round(9 * train['D_44']) / 9
    train['D_51'] = np.round(6 * train['D_51']) / 6
    train.loc[train['D_54'] > 0.99, 'D_54'] = 1
    train.loc[train['D_58'] < 0.01, 'D_58'] = 0
    train['D_59'] = np.round(100 * train['D_59']) / 100
    train.loc[train['D_61'] < 0.01, 'D_61'] = 0
    train['D_62'] = np.round(200 * train['D_62']) / 200
    train['D_65'] = np.round(24 * train['D_65']) / 24
    train.loc[train['D_69'] < 0.01, 'D_69'] = 0
    train['D_70'] = np.round(5 * train['D_70']) / 5
    train['D_72'] = np.round(9 * train['D_72']) / 9
    train['D_74'] = np.round(14 * train['D_74']) / 14
    train['D_75'] = np.round(16 * train['D_75']) / 16
    train['D_78'] = np.round(4 * train['D_78']) / 4
    train['D_79'] = np.round(4 * train['D_79']) / 4
    train['D_80'] = np.round(6 * train['D_80']) / 6
    train['D_81'] = np.round(8 * train['D_81']) / 8
    train['D_81'] = np.round(3 * train['D_81']) / 3
    train['D_82'] = np.round(3 * train['D_82']) / 3
    train.loc[train['D_83'] > 0.99, 'D_83'] = 1
    train.loc[train['D_83'] < 0.2, 'D_83'] = np.round(8 * train.loc[train['D_83'] < 0.2, 'D_83']) / 8
    train.loc[train['D_84'] > 0.4, 'D_84'] = np.round(2 * train.loc[train['D_84'] > 0.4, 'D_84']) / 2
    train.loc[train['D_84'] < 0.4, 'D_84'] = np.round(10 * train.loc[train['D_84'] < 0.4, 'D_84']) / 10
    train['D_86'] = np.round(2 * train['D_86']) / 2
    train['D_87'] = train['D_87'].fillna(0)
    train.loc[train['D_87'] > 0, 'D_87'] = 1
    train['D_89'] = np.round(8 * train['D_89']) / 8
    train.loc[train['D_91'] > 0.4, 'D_84'] = np.round(3 * train.loc[train['D_91'] > 0.4, 'D_91']) / 3
    train.loc[train['D_91'] < 0.01, 'D_91'] = 0
    train['D_92'] = np.round(3 * train['D_92']) / 3
    train['D_93'] = np.round(2 * train['D_93']) / 2
    train['D_94'] = np.round(2 * train['D_94']) / 2
    train['D_96'] = np.round(2 * train['D_96']) / 2
    train.loc[train['D_102'] < 0.01, 'D_102'] = 0
    train['D_103'] = np.round(2 * train['D_103']) / 2
    train.loc[train['D_104'] < 0.01, 'D_104'] = 0
    train['D_106'] = np.round(24 * train['D_106']) / 24
    train['D_107'] = np.round(4 * train['D_107']) / 4
    train['D_108'] = np.round(8 * train['D_108']) / 8
    train['D_109'] = np.round(2 * train['D_109']) / 2
    train.loc[train['D_110'] > 0.99, 'D_110'] = 1
    train.loc[train['D_110'] < 0.01, 'D_110'] = 0
    train['D_111'] = np.round(3 * train['D_111']) / 3
    train.loc[train['D_112'] > 0.99, 'D_112'] = 1
    train['D_113'] = np.round(6 * train['D_113']) / 6
    train['D_116'] = np.round(2 * train['D_116']) / 2
    train['D_122'] = np.round(12 * train['D_122']) / 12
    train['D_123'] = np.round(2 * train['D_123']) / 2
    train['D_124'] = np.round(24 * train['D_124']) / 24
    train['D_125'] = np.round(2 * train['D_125']) / 2
    train['D_127'] = np.round(2 * train['D_127']) / 2
    train['D_128'] = np.round(2 * train['D_128']) / 2
    train['D_129'] = np.round(2 * train['D_129']) / 2
    train['D_130'] = np.round(2 * train['D_130']) / 2
    train.loc[train['D_131'] < 0.01, 'D_131'] = 0
    train.loc[train['D_133'] < 0.01, 'D_133'] = 0
    train.loc[train['D_134'] > 0.99, 'D_134'] = 1
    train['D_135'] = np.round(2 * train['D_135']) / 2
    train['D_136'] = np.round(8 * train['D_136']) / 8
    train['D_137'] = np.round(2 * train['D_137']) / 2
    train['D_138'] = np.round(4 * train['D_138']) / 4
    train['D_139'] = np.round(2 * train['D_139']) / 2
    train['D_140'] = np.round(2 * train['D_140']) / 2
    train.loc[train['D_141'] < 0.01, 'D_141'] = 0
    train['D_143'] = np.round(2 * train['D_143']) / 2
    train.loc[train['D_144'] < 0.01, 'D_144'] = 0
    train['D_145'] = np.round(12 * train['D_145']) / 12

    train.loc[train['P_2'] > 0.99, 'P_2'] = 1
    train.loc[train['P_4'] < 0.01, 'P_4'] = 0

    train['R_1'] = np.round(10 * train['R_1']) / 10
    train['R_2'] = np.round(2 * train['R_2']) / 2
    train['R_3'] = np.round(10 * train['R_3']) / 10
    train['R_4'] = np.round(2 * train['R_4']) / 2
    train.loc[train['R_5'] > 0.4, 'R_5'] = np.round(2 * train.loc[train['R_5'] > 0.4, 'R_5']) / 2
    train.loc[train['R_5'] < 0.01, 'R_5'] = 0
    train.loc[train['R_6'] < 0.01, 'R_6'] = 0
    train.loc[train['R_7'] < 0.01, 'R_7'] = 0
    train.loc[train['R_8'] > 0.90, 'R_8'] = 1
    train.loc[train['R_8'] < 0.3, 'R_8'] = np.round(9 * train.loc[train['R_8'] < 0.3, 'R_8']) / 9
    train['R_9'] = np.round(10 * train['R_9']) / 10
    train.loc[train['R_10'] > 0.90, 'R_10'] = 1
    train.loc[train['R_10'] < 0.3, 'R_10'] = np.round(5 * train.loc[train['R_10'] < 0.3, 'R_10']) / 5
    train.loc[train['R_11'] > 0.4, 'R_11'] = np.round(2 * train.loc[train['R_11'] > 0.4, 'R_11']) / 2
    train.loc[train['R_11'] < 0.3, 'R_11'] = np.round(4 * train.loc[train['R_11'] < 0.3, 'R_11']) / 4
    train.loc[train['R_12'] > 0.90, 'R_12'] = 1
    train.loc[train['R_13'] > 0.40, 'R_13'] = 1
    train.loc[train['R_13'] < 0.4, 'R_13'] = np.round(9 * train.loc[train['R_13'] < 0.4, 'R_13']) / 9
    train.loc[train['R_14'] < 0.01, 'R_14'] = 0
    train['R_15'] = np.round(2 * train['R_15']) / 2
    train.loc[train['R_16'] > 0.4, 'R_16'] = np.round(2 * train.loc[train['R_16'] > 0.4, 'R_16']) / 2
    train.loc[train['R_16'] < 0.4, 'R_16'] = np.round(7 * train.loc[train['R_16'] < 0.4, 'R_16']) / 7
    train.loc[train['R_17'] > 0.40, 'R_17'] = 1
    train.loc[train['R_17'] < 0.4, 'R_17'] = np.round(12 * train.loc[train['R_17'] < 0.4, 'R_17']) / 12
    train['R_18'] = np.round(12 * train['R_18']) / 12
    train['R_19'] = np.round(2 * train['R_19']) / 2
    train.loc[train['R_20'] > 0.90, 'R_20'] = 1
    train.loc[train['R_20'] < 0.4, 'R_20'] = np.round(16 * train.loc[train['R_20'] < 0.4, 'R_20']) / 16
    train['R_21'] = np.round(2 * train['R_21']) / 2
    train['R_22'] = np.round(2 * train['R_22']) / 2
    train['R_23'] = np.round(2 * train['R_23']) / 2
    train['R_24'] = np.round(2 * train['R_24']) / 2
    train['R_25'] = np.round(2 * train['R_25']) / 2
    train['R_26'] = np.round(24 * train['R_26']) / 24
    train.loc[train['R_27'] < 0.20, 'R_27'] = np.round(10 * train.loc[train['R_27'] < 0.20, 'R_27']) / 10
    train.loc[train['R_27'] > 0.60, 'R_27'] = np.round(10 * train.loc[train['R_27'] > 0.60, 'R_27']) / 10
    train['R_28'] = np.round(2 * train['R_28']) / 2

    train['S_6'] = np.round(2 * train['S_6']) / 2
    train['S_8'] = np.round(20 * train['S_8']) / 20
    train['S_11'] = np.round(24 * train['S_11']) / 24
    train['S_13'] = np.round(10 * train['S_13']) / 10
    train['S_15'] = np.round(12 * train['S_15']) / 12
    train['S_16'] = np.round(18 * train['S_16']) / 18
    train.loc[train['S_17'] < 0.01, 'S_17'] = 0
    train.loc[train['S_19'] < 0.01, 'S_19'] = 0
    train['S_20'] = np.round(2 * train['S_20']) / 2
    train.loc[train['S_22'] < 0.02, 'S_22'] = 0
    train.loc[train['S_23'] > 0.90, 'S_23'] = 1
    train.loc[train['S_23'] < 0.00, 'S_23'] = 0
    train.loc[train['S_25'] > 0.99, 'S_25'] = 1
    train.loc[train['S_25'] < 0.01, 'S_25'] = 0
    train.loc[train['S_26'] < 0.01, 'S_26'] = 0
    train.loc[train['S_27'] < 0.01, 'S_27'] = 0

    ## new variables
    train['R_binary'] = train[['R_15', 'R_19', 'R_2', 'R_21', 'R_22', 'R_23', 'R_24', 'R_25', 'R_28', 'R_4']].mean(axis=1)
    train['R_binary_good'] = train[['R_15', 'R_19', 'R_21', 'R_24']].mean(axis=1)
    train['B_binary'] = train[['B_32', 'B_8']].mean(axis=1)
    train['D_binary_neg'] = train[['D_109', 'D_135', 'D_137', 'D_86', 'D_93', 'D_94', 'D_96', 'D_127']].mean(axis=1)
    train['D_binary_pos'] = train[['D_116', 'D_140', 'D_139', 'D_143', 'D_87']].mean(axis=1)
    train['binary_pos'] = train[['R_2', 'R_4', 'B_8', 'D_130', 'R_24', 'R_15', 'R_21', 'S_20', 'B_32', 'R_19', 'D_143',
                                 'D_139', 'D_140', 'D_103', 'R_25', 'R_22', 'D_87', 'D_116', 'R_28', 'R_23']].mean(
        axis=1)
    train['binary_neg'] = train[
        ['D_135', 'D_137', 'D_109', 'D_93', 'D_86', 'D_96', 'D_94', 'B_31', 'S_6', 'D_129', 'D_128', 'D_127',
         'B_33']].mean(axis=1)

    train['D_high_pos'] = train[['D_88', 'D_110', 'D_42']].mean(axis=1)
    train['D_high_neg'] = train[['D_73', 'D_134', 'D_76', 'D_132']].mean(axis=1)

    train['B_ordinal'] = train[['B_21', 'B_41']].mean(axis=1)
    train['D_ordinal'] = train[
        ['D_108', 'D_123', 'D_123', 'D_124', 'D_125', 'D_136', 'D_138', 'D_80', 'D_82', 'D_84']].mean(axis=1)
    train['R_ordinal'] = train[['R_11', 'R_17', 'R_18', 'R_9']].mean(axis=1)

    train['B_normal'] = train[['B_15', 'B_27', 'B_36']].mean(axis=1)
    train['D_normal_pos'] = train[['D_102', 'D_69', 'D_83']].mean(axis=1)
    train['D_normal_neg'] = train[['D_105', 'D_144', 'D_60']].mean(axis=1)
    train['S_normal_neg'] = train[['S_12', 'S_18']].mean(axis=1)
    train['S_normal_pos'] = train[['S_17', 'S_19', 'S_9']].mean(axis=1)

    train = train.drop(drop_cols, axis=1)

    for bcol in [f'B_{i}' for i in [11, 14, 17]] + ['D_39', 'D_131'] + [f'S_{i}' for i in [16, 23]]:
        for pcol in ['P_2', 'P_3']:
            if bcol in normal_cols and pcol in normal_cols:
                train[f'{bcol}-{pcol}'] = train[bcol].values - train[pcol].values
                train[f'{bcol}-{pcol}'] = train[f'{bcol}-{pcol}'].astype(np.float32)
                normal_cols.append(f'{bcol}-{pcol}')


    ## Preprocessing Categorical features ##########
    print('Starting data preprocess...')
    train_cat_agg = preprocess_cat_data(train, cat_cols)

    print('Number of cat features:', train_cat_agg.shape[1])
    train_cat_agg.to_pickle(DATA_DIR_FINAL + f'test_cat_{p}.pkl')

    del train_cat_agg
    gc.collect()

    train = train.drop(cat_cols, axis=1)
    gc.collect()

    ## Preprocessing High missing features ##########
    train_num_agg = preprocess_num_data(train, high_missing_cols, feature_type='high')

    temp = train.groupby("customer_ID")['month'].agg(['count', 'min'])
    temp.columns = ['month_count', 'month_min']
    temp = temp.reset_index(drop=False)
    train_num_agg = train_num_agg.merge(temp, on=['customer_ID'], how='left', copy=False)

    del temp
    gc.collect()

    temp = train[['customer_ID', 'S_2']]
    temp['S_2'] = pd.to_datetime(temp['S_2'])
    temp['number_of_days'] = 365 * (temp['S_2'].dt.year - min(temp['S_2'].dt.year)) + 30 * temp['S_2'].dt.month + temp[
        'S_2'].dt.day
    temp['number_of_days_shift'] = temp.groupby("customer_ID")['number_of_days'].shift(1)
    temp['diff_day'] = temp['number_of_days'].values - temp['number_of_days_shift'].values

    temp = temp.groupby("customer_ID")['diff_day'].agg(['mean', 'last', 'max', 'min'])
    temp.columns = ['diff_day_mean', 'diff_day_last', 'diff_day_max', 'diff_day_min']
    temp = temp.reset_index(drop=False)
    train_num_agg = train_num_agg.merge(temp, on=['customer_ID'], how='left', copy=False)

    del temp
    gc.collect()

    print('Number of high missing features:', train_num_agg.shape[1])
    train_num_agg.to_pickle(DATA_DIR_FINAL + f'test_num_high_{p}.pkl')

    del train_num_agg
    gc.collect()

    train = train.drop(high_missing_cols, axis=1)
    gc.collect()

    ## Preprocessing binary features ##########
    train_num_agg = preprocess_num_data(train, binary_cols, feature_type='binary')

    print('Number of binary features:', train_num_agg.shape[1])
    train_num_agg.to_pickle(DATA_DIR_FINAL + f'test_num_binary_{p}.pkl')

    del train_num_agg
    gc.collect()

    train = train.drop(binary_cols, axis=1)
    gc.collect()

    ## Preprocessing ordinal features ##########
    train_num_agg = preprocess_num_data(train, ordinal_cols, feature_type='ordinal')

    print('Number of ordinal features:', train_num_agg.shape[1])
    train_num_agg.to_pickle(DATA_DIR_FINAL + f'test_num_ordinal_{p}.pkl')

    del train_num_agg,
    gc.collect()

    ## Preprocessing good features ##########
    train_num_agg = preprocess_num_data(train, normal_cols, feature_type='normal')

    print('Number of good features:', train_num_agg.shape[1])
    train_num_agg.to_pickle(DATA_DIR_FINAL + f'test_num_good_{p}.pkl')

    del train_num_agg
    gc.collect()

    ## Preprocessing month features ##########
    train_num = get_month_at_value(train[['customer_ID', 'month'] + normal_cols + ordinal_cols],
                                   normal_cols + ordinal_cols, max_value=True)

    train_num.to_pickle(DATA_DIR_FINAL + f'test_num_month_at_max_{p}.pkl')

    del train_num
    gc.collect()

    train_num = get_month_at_value(train[['customer_ID', 'month'] + normal_cols + ordinal_cols],
                                   normal_cols + ordinal_cols, max_value=False)

    train_num.to_pickle(DATA_DIR_FINAL + f'test_num_month_at_min_{p}.pkl')

    del train_num, train
    gc.collect()


    ## Preprocessing month features ##########
    train = train.drop(cat_cols+high_missing_cols+binary_cols, axis=1)

    train_num = get_month_at_weighted_value(train[['customer_ID', 'month'] + normal_cols + ordinal_cols],
                                   normal_cols + ordinal_cols, max_value=True)

    train_num.to_pickle(DATA_DIR_FINAL + f'test_num_month_at_max_weighted_{p}.pkl')

    del train_num
    gc.collect()

    train_num = get_month_at_weighted_value(train[['customer_ID', 'month'] + normal_cols + ordinal_cols],
                                   normal_cols + ordinal_cols, max_value=False)

    train_num.to_pickle(DATA_DIR_FINAL + f'test_num_month_at_min_weighted_{p}.pkl')

    del train_num, train
    gc.collect()