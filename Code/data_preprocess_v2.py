import pandas as pd
import numpy as np
import gc
import pickle
import matplotlib.pylab as plt
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
import warnings
import joblib
warnings.filterwarnings("ignore")
gc.enable()

def outlier_fixer_v1(df, cols):

    outlier_cols = {'columns':[],
                    'max_value':[],
                    'min_value':[]}
    for col in tqdm(cols):
        if col not in ['customer_ID','S_2', 'month', 'target', 'type']:
            max_value = df[col].max()
            min_value = df[col].min()
            q1 = df[col].quantile(0.005)
            q2 = df[col].quantile(0.995)
            outlier_cols['columns'].append(col)
            if max_value/q2 > 3 and max_value>=2:
                max_p = np.nanquantile(df[col].values, 0.99)
                outlier_cols['max_value'].append(max_p)
                df[col] = np.where(df[col].values > max_p, max_p, df[col].values)
            else:
                outlier_cols['max_value'].append(None)

            if np.abs(min_value/q1) > 3.0 and min_value<0:
                min_p = np.nanquantile(df[col].values, 0.01)
                df[col] = np.where(df[col].values < min_p, min_p, df[col].values)
                outlier_cols['min_value'].append(min_p)
            else:
                outlier_cols['min_value'].append(None)

    return  df, outlier_cols

def get_stats(train, test_lb, test_pb):

    stats = {'columns':[],
             'max_value_train':[],
             'max_value_lb': [],
             'max_value_pb': [],
             'min_value_train':[],
             'min_value_lb': [],
             'min_value_pb': [],
             'mean_value_train':[],
             'mean_value_lb': [],
             'mean_value_pb': [],
             'std_value_train': [],
             'std_value_lb': [],
             'std_value_pb': [],
             'number_of_nan_train':[],
             'number_of_nan_lb': [],
             'number_of_nan_pb': []
             }


    for col in train.columns:
        if col not in ['customer_ID','S_2', 'month', 'target', 'type']:
            stats['columns'].append(col)
            stats['max_value_train'].append(train[col].max())
            stats['max_value_lb'].append(test_lb[col].max())
            stats['max_value_pb'].append(test_pb[col].max())
            stats['min_value_train'].append(train[col].min())
            stats['min_value_lb'].append(test_lb[col].min())
            stats['min_value_pb'].append(test_pb[col].min())
            stats['mean_value_train'].append(train[col].mean())
            stats['mean_value_lb'].append(test_lb[col].mean())
            stats['mean_value_pb'].append(test_pb[col].mean())
            stats['std_value_train'].append(train[col].std())
            stats['std_value_lb'].append(test_lb[col].std())
            stats['std_value_pb'].append(test_pb[col].std())
            stats['number_of_nan_train'].append(train[col].isnull().sum()/train.shape[0])
            stats['number_of_nan_lb'].append(test_lb[col].isnull().sum() / test_lb.shape[0])
            stats['number_of_nan_pb'].append(test_pb[col].isnull().sum() / test_pb.shape[0])
    return pd.DataFrame.from_dict(stats)

DATA_DIR = 'preprocessed_data/'

cat_cols = ["B_30", "B_38","D_114", "D_117", "D_120", "D_126",
            "D_63", "D_64", "D_68"]

drop_cols = ['D_88', 'D_87', 'D_108', 'D_110', 'D_111', 'B_39',
             'D_73', 'B_42', 'B_31', 'D_116', 'D_66', 'D_109', 'R_23',
             'R_28', 'R_22','R_25','D_93','D_137', 'D_94','D_135', 'D_86',
             'D_96']

summary = joblib.load('figures/selected_features_v1.joblib')
train_cols_1200 = summary['selected_features_names']
train_cols_1374 = summary['selected_features_names'] + summary['eliminated_features_names'][426:]

DATA_DIR = 'preprocessed_data/test_preprocessed/'
test_set = []
for k in range(2):

    print(f'Reading {k}-th test data...')
    df_cat = pd.read_pickle(DATA_DIR + f'test_cat_{k}.pkl')
    temp = pd.read_pickle(DATA_DIR + f'test_num_high_{k}.pkl')
    temp = temp.merge(df_cat, on=['customer_ID'], how='left')

    del df_cat
    gc.collect()

    df_med = pd.read_pickle(DATA_DIR + f'test_num_binary_{k}.pkl')
    temp = temp.merge(df_med, on=['customer_ID'], how='left')

    del df_med
    gc.collect()

    df_skew = pd.read_pickle(DATA_DIR + f'test_num_ordinal_{k}.pkl')
    temp = temp.merge(df_skew, on=['customer_ID'], how='left')

    del df_skew
    gc.collect()

    print('merging max month dataset...')
    df_month_max = pd.read_pickle(DATA_DIR + f'test_num_month_at_max_{k}.pkl')
    temp = temp.merge(df_month_max, on=['customer_ID'], how='left')

    del df_month_max
    gc.collect()

    print('merging min month dataset...')
    df_month_min = pd.read_pickle(DATA_DIR + f'test_num_month_at_min_{k}.pkl')
    temp = temp.merge(df_month_min, on=['customer_ID'], how='left')

    del df_month_min
    gc.collect()

    print('merging weighted max month dataset...')
    df_month_max = pd.read_pickle(DATA_DIR + f'test_num_month_at_max_weighted_{k}.pkl')
    temp = temp.merge(df_month_max, on=['customer_ID'], how='left')

    del df_month_max
    gc.collect()

    print('merging weighted min month dataset...')
    df_month_min = pd.read_pickle(DATA_DIR + f'test_num_month_at_min_weighted_{k}.pkl')
    temp = temp.merge(df_month_min, on=['customer_ID'], how='left')

    del df_month_min
    gc.collect()

    df = pd.read_pickle(DATA_DIR + f'test_num_good_{k}.pkl')
    df = df.merge(temp, on=['customer_ID'], how='left')

    del temp
    gc.collect()

    df = df.replace([np.inf, -np.inf], np.nan)

    df[['customer_ID']+train_cols_1374].to_parquet(f'test_{k}.parquet')




# train = pd.read_parquet(DATA_DIR + 'train.parquet')
# test_lb = pd.read_pickle(DATA_DIR + 'test_preprocessed/test_0.pkl')
# test_pb = pd.read_pickle(DATA_DIR + 'test_preprocessed/test_1.pkl')
#
# test_lb = test_lb.drop(['max_month'], axis=1)
# test_pb = test_pb.drop(['max_month'], axis=1)
# test_lb['month'] = test_lb['month'].values - 13
# test_pb['month'] = test_pb['month'].values - 19
#
# train = train.drop(cat_cols+drop_cols, axis=1)
# test_lb = test_lb.drop(cat_cols+drop_cols, axis=1)
# test_pb = test_pb.drop(cat_cols+drop_cols, axis=1)
#
# train['type'] = 'train'
# test_lb['type'] = 'lb'
# test_pb['type'] = 'pb'
#
# df = pd.concat([train, test_lb[train.columns], test_pb[train.columns]], axis=0)
#
# del train, test_lb, test_pb
# gc.collect()
#
# df, outlier_cols = outlier_fixer_v1(df, df.columns)
#
# train = df.loc[df['type']=='train']
# test_lb = df.loc[df['type']=='lb']
# test_pb = df.loc[df['type']=='pb']
#
# del df
# gc.collect()

# print('merging target...')
# with open(DATA_DIR + 'encoder.pkl', 'rb') as fin:
#     encoder = pickle.load(fin)
#
# labels = pd.read_csv(DATA_DIR+ 'train_labels.csv')
# labels['customer_ID'] = encoder.transform(labels['customer_ID'])
#
# train = train.merge(labels, on=['customer_ID'], how='left')
#
# del encoder, labels
# gc.collect()


# stats = get_stats(train, test_lb, test_pb)
# stats.to_csv('stats_removed_outlies.csv', index=False)
# exit()

# stats = pd.read_csv('stats_removed_outlies.csv')
#
# binary_cols = joblib.load(DATA_DIR + 'binary_cols.joblib')
# high_missing_cols = joblib.load(DATA_DIR + 'high_missing_cols.joblib')
# ordinal_cols = joblib.load(DATA_DIR + 'ordinal_cols.joblib')
# normal_cols = joblib.load(DATA_DIR + 'normal_cols.joblib')
#
# stats['column_type'] = np.nan
# for col in stats['columns']:
#     if col in binary_cols:
#         stats.loc[stats['columns']==col, 'column_type'] = 'binary'
#     elif col in high_missing_cols:
#         stats.loc[stats['columns']==col, 'column_type'] = 'high'
#     elif col in ordinal_cols:
#         stats.loc[stats['columns'] == col, 'column_type'] = 'ordinal'
#     elif col in normal_cols:
#         stats.loc[stats['columns'] == col, 'column_type'] = 'normal'
#
# stats.to_csv('stats_with_type.csv', index=False)


