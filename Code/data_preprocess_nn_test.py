import pandas as pd
import numpy as np
import gc
import pickle
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
import lightgbm as lgb
import os
from scipy.misc import derivative
import joblib
import pathlib
import matplotlib.pylab as plt

import warnings
warnings.filterwarnings("ignore")

def get_stats(df, target=False):

    stats = {'columns':[],
             'max_value':[],
             'min_value':[],
             'mean_value':[],
             'quantile_1': [],
             'quantile_25': [],
             'quantile_75': [],
             'quantile_99': [],
             'skew':[],
             'std':[],
             'number_of_nan':[],
             'quantile_count1':[],
             'quantile_count99':[],
             'nan_npl': [],
             'not_nan_npl': [],
             'target':[]}


    for col in df.columns:
        if col not in ['customer_ID','S_2', 'month', 'target']:
            if target:
                for lb in [0, 1]:
                    stats['columns'].append(col)
                    stats['max_value'].append(df.loc[df['target']==lb, col].max())
                    stats['min_value'].append(df.loc[df['target'] == lb, col].min())
                    stats['mean_value'].append(df.loc[df['target'] == lb, col].mean())
                    stats['quantile_1'].append(df.loc[df['target'] == lb, col].quantile(0.01))
                    stats['quantile_25'].append(df.loc[df['target'] == lb, col].quantile(0.25))
                    stats['quantile_75'].append(df.loc[df['target'] == lb, col].quantile(0.75))
                    stats['quantile_99'].append(df.loc[df['target'] == lb, col].quantile(0.99))
                    stats['skew'].append(df.loc[df['target'] == lb, col].skew())
                    stats['std'].append(df.loc[df['target'] == lb, col].std())
                    stats['number_of_nan'].append(df.loc[df['target'] == lb, col].isnull().sum()/df.loc[(df['target'] == lb)].shape[0])
                    stats['quantile_count1'].append(df.loc[(df['target'] == lb) & (df[col]<=df[col].quantile(0.01)), col].shape[0] / df.loc[(df['target'] == lb)].shape[0])
                    stats['quantile_count99'].append(df.loc[(df['target'] == lb) & (df[col] >= df[col].quantile(0.99)), col].shape[0] /df.loc[(df['target'] == lb)].shape[0])
                    stats['nan_npl'].append(df.loc[df[col].isnull(), 'target'].sum() / df.loc[df[col].isnull(), 'target'].shape[0])
                    stats['not_nan_npl'].append(df.loc[~df[col].isnull(), 'target'].sum() / df.loc[~df[col].isnull(), 'target'].shape[0])
                    stats['target'].append(lb)
            else:
                stats['columns'].append(col)
                stats['max_value'].append(df[col].max())
                stats['min_value'].append(df[col].min())
                stats['mean_value'].append(df[col].mean())
                stats['quantile_1'].append(df[col].quantile(0.01))
                stats['quantile_25'].append(df[col].quantile(0.25))
                stats['quantile_75'].append(df[col].quantile(0.75))
                stats['quantile_99'].append(df[col].quantile(0.99))
                stats['skew'].append(df[col].skew())
                stats['std'].append(df[col].std())
                stats['number_of_nan'].append(df[col].isnull().sum()/df.shape[0])
                stats['quantile_count1'].append(df.loc[(df[col] <= df[col].quantile(0.01)), col].shape[0] / df.shape[0])
                stats['quantile_count99'].append(df.loc[(df[col] >= df[col].quantile(0.99)), col].shape[0] / df.shape[0])
                stats['nan_npl'].append(df.loc[df[col].isnull(), 'target'].sum() / df.loc[df[col].isnull(), 'target'].shape[0])
                stats['not_nan_npl'].append(df.loc[~df[col].isnull(), 'target'].sum() / df.loc[~df[col].isnull(), 'target'].shape[0])
                stats['target'].append('no')

    return pd.DataFrame.from_dict(stats)

def one_hot_encoding(train, variables):
    cat_cols = []
    one_hot_dict = {}
    for var in tqdm(variables):
        one_hot_dict[var] = []
        train[var] = train[var].fillna(-9)
        values = train[var].unique()
        if len(values)>2:
            for val in values:
                train[f'{var}_{int(val)}'] = np.where(train[var]==val, 1, 0)
                train[f'{var}_{int(val)}'] = train[f'{var}_{int(val)}'].astype('int8')
                cat_cols.append(f'{var}_{int(val)}')
                one_hot_dict[var].append(val)
            train.drop([var], axis=1, inplace=True)
        else:
            cat_cols.append(var)

    return train, cat_cols, one_hot_dict


def preprocess(fold, df_temp, high_missing_cols, binary_cols, ordinal_cols, normal_cols, cat_cols):

    print('New dataset')
    df_num = pd.DataFrame()
    df_num['customer_ID'] = np.repeat(list(np.sort(df_temp['customer_ID'].unique())), 13)
    df_num['month'] = len(df_temp['customer_ID'].unique()) * list(df_temp['month'].unique())
    df_num = df_num.merge(df_temp, on=['customer_ID', 'month'], how='left', copy=False)

    df_num = df_num.sort_values(['customer_ID', 'month']).reset_index(drop=True)
    tar = df_num[['customer_ID']].drop_duplicates().sort_index().reset_index(drop=True)

    cols = list(df_num.dtypes[df_num.dtypes == 'float64'].index)
    for col in tqdm(cols):
        df_num[col] = df_num[col].astype(np.float32)

    df_arr = np.reshape(df_num[normal_cols + ordinal_cols].fillna(0).values,
                        (int(df_num.shape[0] / 13), 13, len(normal_cols + ordinal_cols)))
    print('3D input:', df_arr.shape)
    np.save(f'neural_network/test/train_num_{fold}.npy', df_arr)

    df_temp.drop(normal_cols + ordinal_cols, axis=1, inplace=True)

    del df_arr, df_num
    gc.collect()

    train_num_agg = df_temp.groupby("customer_ID")[high_missing_cols].agg(['last', 'mean'])
    train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
    train_num_agg[train_num_agg.columns] = train_num_agg[train_num_agg.columns].astype(np.float32)
    train_num_agg = train_num_agg.fillna(0)
    train_num_agg = train_num_agg.loc[tar['customer_ID']].reset_index(drop=True)
    print('High missing columns:', train_num_agg.shape)
    np.save(f'neural_network/test/train_high_{fold}.npy', train_num_agg.values)

    df_temp.drop(high_missing_cols, axis=1, inplace=True)

    del train_num_agg
    gc.collect()

    train_num_agg = df_temp.groupby("customer_ID")[cat_cols + binary_cols].agg(['last', 'first', 'mean'])
    train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
    train_num_agg = train_num_agg.fillna(0)
    train_num_agg = train_num_agg.loc[tar['customer_ID']].reset_index(drop=True)
    print('Binary columns:', train_num_agg.shape)
    np.save(f'neural_network/test/train_skew_{fold}.npy', train_num_agg.values)

    df_temp.drop(cat_cols, axis=1, inplace=True)

    del train_num_agg
    gc.collect()

    print('Target:', tar.shape)
    target = df_temp.groupby("customer_ID")['target'].last()
    target = target.loc[tar['customer_ID']].reset_index(drop=False)
    target.to_pickle(f'neural_network/test/targets_{fold}.pkl')

    del df_temp, tar
    gc.collect()

    return f'{fold} is done..'


DATA_DIR2 = 'preprocessed_data/test_preprocessed/'


cat_cols = ["B_30", "B_38","D_114", "D_117", "D_120", "D_126",
            "D_63", "D_64", "D_68"]

drop_cols = ['D_88', 'D_87', 'D_108', 'D_110', 'D_111', 'B_39',
             'D_73', 'B_42', 'B_31', 'D_116', 'D_66', 'D_109', 'R_23',
             'R_28', 'R_22','R_25','D_93','D_137', 'D_94','D_135', 'D_86',
             'D_96']

for fold in range(2):

    train = pd.read_pickle(DATA_DIR2 + f'test_{fold}.pkl')

    train = train.drop(['max_month'], axis=1)
    if fold == 0:
        train['month'] = train['month'].values - 13
    else:
        train['month'] = train['month'].values - 19

    train.loc[train['B_2']>0.8, 'B_2']=np.round(2*train.loc[train['B_2']>0.8, 'B_2'].values)/2
    train.loc[train['B_2']<0.8, 'B_2']=np.round(36*train.loc[train['B_2']<0.8, 'B_2'].values)/36
    train['B_4'] = np.round(100*train['B_4'])/100
    train['B_10'] = np.round(50*train['B_10'])/50
    train['B_16'] = np.round(13*train['B_16'])/13
    train.loc[train['B_17']>0.99, 'B_17'] = 1
    train.loc[train['B_17']<0.01, 'B_17'] = 0
    train['B_18'] = np.round(15*train['B_18'])/15
    train['B_19'] = np.round(30*train['B_19'])/30
    train['B_20'] = np.round(18*train['B_20'])/18
    train['B_21'] = np.round(12*train['B_21'])/12
    train['B_22'] = np.round(4*train['B_22'])/4
    train.loc[train['B_25']>0.99, 'B_25'] = 1
    train.loc[train['B_25']<0.01, 'B_25'] = 0
    train.loc[train['B_29']<0.01, 'B_29'] = 0
    train['B_31'] = np.round(2*train['B_31'])/2
    train['B_32'] = np.round(2*train['B_32'])/2
    train['B_33'] = np.round(2*train['B_33'])/2
    train.loc[train['B_36']>0.99, 'B_36'] = 1
    train.loc[train['B_36']<0.01, 'B_36'] = 0
    train.loc[train['B_39']>0.99, 'B_39'] = 1
    train.loc[train['B_39']<0.01, 'B_39'] = 0
    train['B_41'] = np.round(7*train['B_41'])/7
    train['B_8'] = np.round(2*train['B_8'])/2

    train['D_39'] = np.round(34*train['D_39'])/34
    train.loc[train['B_41']<0.01, 'B_41'] = 0
    train['D_44'] = np.round(9*train['D_44'])/9
    train['D_51'] = np.round(6*train['D_51'])/6
    train.loc[train['D_54']>0.99, 'D_54'] = 1
    train.loc[train['D_58']<0.01, 'D_58'] = 0
    train['D_59'] = np.round(100*train['D_59'])/100
    train.loc[train['D_61']<0.01, 'D_61'] = 0
    train['D_62'] = np.round(200*train['D_62'])/200
    train['D_65'] = np.round(24*train['D_65'])/24
    train.loc[train['D_69']<0.01, 'D_69'] = 0
    train['D_70'] = np.round(5*train['D_70'])/5
    train['D_72'] = np.round(9*train['D_72'])/9
    train['D_74'] = np.round(14*train['D_74'])/14
    train['D_75'] = np.round(16*train['D_75'])/16
    train['D_78'] = np.round(4*train['D_78'])/4
    train['D_79'] = np.round(4*train['D_79'])/4
    train['D_80'] = np.round(6*train['D_80'])/6
    train['D_81'] = np.round(8*train['D_81'])/8
    train['D_81'] = np.round(3*train['D_81'])/3
    train['D_82'] = np.round(3*train['D_82'])/3
    train.loc[train['D_83']>0.99, 'D_83'] = 1
    train.loc[train['D_83']<0.2, 'D_83'] = np.round(8*train.loc[train['D_83']<0.2, 'D_83'])/8
    train.loc[train['D_84']>0.4, 'D_84'] = np.round(2*train.loc[train['D_84']>0.4, 'D_84'])/2
    train.loc[train['D_84']<0.4, 'D_84'] = np.round(10*train.loc[train['D_84']<0.4, 'D_84'])/10
    train['D_86'] = np.round(2*train['D_86'])/2
    train['D_87'] = train['D_87'].fillna(0)
    train.loc[train['D_87']>0, 'D_87'] = 1
    train['D_89'] = np.round(8*train['D_89'])/8
    train.loc[train['D_91']>0.4, 'D_84'] = np.round(3*train.loc[train['D_91']>0.4, 'D_91'])/3
    train.loc[train['D_91']<0.01, 'D_91'] = 0
    train['D_92'] = np.round(3*train['D_92'])/3
    train['D_93'] = np.round(2*train['D_93'])/2
    train['D_94'] = np.round(2*train['D_94'])/2
    train['D_96'] = np.round(2*train['D_96'])/2
    train.loc[train['D_102']<0.01, 'D_102'] = 0
    train['D_103'] = np.round(2*train['D_103'])/2
    train.loc[train['D_104']<0.01, 'D_104']=0
    train['D_106'] = np.round(24*train['D_106'])/24
    train['D_107'] = np.round(4*train['D_107'])/4
    train['D_108'] = np.round(8*train['D_108'])/8
    train['D_109'] = np.round(2*train['D_109'])/2
    train.loc[train['D_110']>0.99, 'D_110'] = 1
    train.loc[train['D_110']<0.01, 'D_110']=0
    train['D_111'] = np.round(3*train['D_111'])/3
    train.loc[train['D_112']>0.99, 'D_112'] = 1
    train['D_113'] = np.round(6*train['D_113'])/6
    train['D_116'] = np.round(2*train['D_116'])/2
    train['D_122'] = np.round(12*train['D_122'])/12
    train['D_123'] = np.round(2*train['D_123'])/2
    train['D_124'] = np.round(24*train['D_124'])/24
    train['D_125'] = np.round(2*train['D_125'])/2
    train['D_127'] = np.round(2*train['D_127'])/2
    train['D_128'] = np.round(2*train['D_128'])/2
    train['D_129'] = np.round(2*train['D_129'])/2
    train['D_130'] = np.round(2*train['D_130'])/2
    train.loc[train['D_131']<0.01, 'D_131']=0
    train.loc[train['D_133']<0.01, 'D_133']=0
    train.loc[train['D_134']>0.99, 'D_134'] = 1
    train['D_135'] = np.round(2*train['D_135'])/2
    train['D_136'] = np.round(8*train['D_136'])/8
    train['D_137'] = np.round(2*train['D_137'])/2
    train['D_138'] = np.round(4*train['D_138'])/4
    train['D_139'] = np.round(2*train['D_139'])/2
    train['D_140'] = np.round(2*train['D_140'])/2
    train.loc[train['D_141']<0.01, 'D_141']=0
    train['D_143'] = np.round(2*train['D_143'])/2
    train.loc[train['D_144']<0.01, 'D_144']=0
    train['D_145'] = np.round(12*train['D_145'])/12

    train.loc[train['P_2']>0.99, 'P_2'] = 1
    train.loc[train['P_4']<0.01, 'P_4']=0

    train['R_1'] = np.round(10*train['R_1'])/10
    train['R_2'] = np.round(2*train['R_2'])/2
    train['R_3'] = np.round(10*train['R_3'])/10
    train['R_4'] = np.round(2*train['R_4'])/2
    train.loc[train['R_5']>0.4, 'R_5'] = np.round(2*train.loc[train['R_5']>0.4, 'R_5'])/2
    train.loc[train['R_5']<0.01, 'R_5']=0
    train.loc[train['R_6']<0.01, 'R_6']=0
    train.loc[train['R_7']<0.01, 'R_7']=0
    train.loc[train['R_8']>0.90, 'R_8'] = 1
    train.loc[train['R_8']<0.3, 'R_8'] = np.round(9*train.loc[train['R_8']<0.3, 'R_8'])/9
    train['R_9'] = np.round(10*train['R_9'])/10
    train.loc[train['R_10']>0.90, 'R_10'] = 1
    train.loc[train['R_10']<0.3, 'R_10'] = np.round(5*train.loc[train['R_10']<0.3, 'R_10'])/5
    train.loc[train['R_11']>0.4, 'R_11'] = np.round(2*train.loc[train['R_11']>0.4, 'R_11'])/2
    train.loc[train['R_11']<0.3, 'R_11'] = np.round(4*train.loc[train['R_11']<0.3, 'R_11'])/4
    train.loc[train['R_12']>0.90, 'R_12'] = 1
    train.loc[train['R_13']>0.40, 'R_13'] = 1
    train.loc[train['R_13']<0.4, 'R_13'] = np.round(9*train.loc[train['R_13']<0.4, 'R_13'])/9
    train.loc[train['R_14']<0.01, 'R_14']=0
    train['R_15'] = np.round(2*train['R_15'])/2
    train.loc[train['R_16']>0.4, 'R_16'] = np.round(2*train.loc[train['R_16']>0.4, 'R_16'])/2
    train.loc[train['R_16']<0.4, 'R_16'] = np.round(7*train.loc[train['R_16']<0.4, 'R_16'])/7
    train.loc[train['R_17']>0.40, 'R_17'] = 1
    train.loc[train['R_17']<0.4, 'R_17'] = np.round(12*train.loc[train['R_17']<0.4, 'R_17'])/12
    train['R_18'] = np.round(12*train['R_18'])/12
    train['R_19'] = np.round(2*train['R_19'])/2
    train.loc[train['R_20']>0.90, 'R_20'] = 1
    train.loc[train['R_20']<0.4, 'R_20'] = np.round(16*train.loc[train['R_20']<0.4, 'R_20'])/16
    train['R_21'] = np.round(2*train['R_21'])/2
    train['R_22'] = np.round(2*train['R_22'])/2
    train['R_23'] = np.round(2*train['R_23'])/2
    train['R_24'] = np.round(2*train['R_24'])/2
    train['R_25'] = np.round(2*train['R_25'])/2
    train['R_26'] = np.round(24*train['R_26'])/24
    train.loc[train['R_27']<0.20, 'R_27'] = np.round(10*train.loc[train['R_27']<0.20, 'R_27'])/10
    train.loc[train['R_27']>0.60, 'R_27'] = np.round(10*train.loc[train['R_27']>0.60, 'R_27'])/10
    train['R_28'] = np.round(2*train['R_28'])/2

    train['S_6'] = np.round(2*train['S_6'])/2
    train['S_8'] = np.round(20*train['S_8'])/20
    train['S_11'] = np.round(24*train['S_11'])/24
    train['S_13'] = np.round(10*train['S_13'])/10
    train['S_15'] = np.round(12*train['S_15'])/12
    train['S_16'] = np.round(18*train['S_16'])/18
    train.loc[train['S_17']<0.01, 'S_17']=0
    train.loc[train['S_19']<0.01, 'S_19']=0
    train['S_20'] = np.round(2*train['S_20'])/2
    train.loc[train['S_22']<0.02, 'S_22']=0
    train.loc[train['S_23']>0.90, 'S_23'] = 1
    train.loc[train['S_23']<0.00, 'S_23'] = 0
    train.loc[train['S_25']>0.99, 'S_25'] = 1
    train.loc[train['S_25']<0.01, 'S_25'] = 0
    train.loc[train['S_26']<0.01, 'S_26'] = 0
    train.loc[train['S_27']<0.01, 'S_27'] = 0

    ## new variables
    train['R_binary'] = train[['R_15', 'R_19', 'R_2', 'R_21', 'R_22', 'R_23', 'R_24', 'R_25', 'R_28', 'R_4']].mean(axis=1)
    train['R_binary_good'] = train[['R_15', 'R_19', 'R_21', 'R_24']].mean(axis=1)
    train['B_binary'] = train[['B_32', 'B_8']].mean(axis=1)
    train['D_binary_neg'] = train[['D_109', 'D_135', 'D_137', 'D_86', 'D_93', 'D_94', 'D_96', 'D_127']].mean(axis=1)
    train['D_binary_pos'] = train[['D_116', 'D_140', 'D_139', 'D_143', 'D_87']].mean(axis=1)
    train['binary_pos'] = train[['R_2','R_4','B_8','D_130','R_24','R_15','R_21','S_20','B_32','R_19','D_143',
                                 'D_139','D_140','D_103','R_25','R_22','D_87','D_116','R_28','R_23']].mean(axis=1)
    train['binary_neg'] = train[['D_135','D_137','D_109','D_93','D_86','D_96','D_94','B_31','S_6','D_129','D_128','D_127','B_33']].mean(axis=1)

    train['D_high_pos'] = train[['D_88', 'D_110', 'D_42']].mean(axis=1)
    train['D_high_neg'] = train[['D_73', 'D_134', 'D_76', 'D_132']].mean(axis=1)

    train['B_ordinal'] = train[['B_21', 'B_41']].mean(axis=1)
    train['D_ordinal'] = train[['D_108', 'D_123', 'D_123','D_124','D_125','D_136','D_138', 'D_80', 'D_82','D_84']].mean(axis=1)
    train['R_ordinal'] = train[['R_11', 'R_17', 'R_18','R_9']].mean(axis=1)

    train['B_normal'] = train[['B_15', 'B_27', 'B_36']].mean(axis=1)
    train['D_normal_pos'] = train[['D_102', 'D_69', 'D_83']].mean(axis=1)
    train['D_normal_neg'] = train[['D_105', 'D_144', 'D_60']].mean(axis=1)
    train['S_normal_neg'] = train[['S_12', 'S_18']].mean(axis=1)
    train['S_normal_pos'] = train[['S_17', 'S_19', 'S_9']].mean(axis=1)

    train = train.drop(drop_cols, axis=1)

    binary_cols = joblib.load('preprocessed_data/binary_cols.joblib')
    high_missing_cols = joblib.load('preprocessed_data/high_missing_cols.joblib')
    cat_cols = joblib.load('preprocessed_data/cat_cols.joblib')
    ordinal_cols = joblib.load('preprocessed_data/ordinal_cols.joblib')
    normal_cols = joblib.load('preprocessed_data/normal_cols.joblib')
    one_hot_cols = joblib.load('neural_network/data/one_hot_cols.joblib')
    one_hot_dict = joblib.load('neural_network/data/one_hot_dict.joblib')
    outliers = joblib.load('neural_network/data/outliers.joblib')

    for idx in range(len(outliers['columns'])):
        col = outliers['columns'][idx]
        max_value = outliers['max_value'][idx]
        min_value = outliers['min_value'][idx]

        if max_value is not None:
            train[col] = np.where(train[col].values > max_value, max_value, train[col].values)
        if min_value is not None:
            train[col] = np.where(train[col].values < min_value, min_value, train[col].values)

    for col in tqdm(cat_cols):
        values = one_hot_dict[col]
        train[col] = train[col].fillna(train[col].min())
        if len(values)>2:
            for val in values:
                train[f'{col}_{int(val)}'] = np.where(train[col] == val, 1, 0)
                train[f'{col}_{int(val)}'] = train[f'{col}_{int(val)}'].astype('int8')

            train = train.drop([col], axis=1)

    train = train.replace([np.inf, -np.inf], np.nan)

    for bcol in [f'B_{i}' for i in [11,14,17]]+['D_39','D_131']+[f'S_{i}' for i in [16,23]]:
        for pcol in ['P_2','P_3']:
            if bcol in normal_cols and pcol in normal_cols:
                train[f'{bcol}-{pcol}'] = train[bcol].values - train[pcol].values
                train[f'{bcol}-{pcol}'] = train[f'{bcol}-{pcol}'].astype(np.float32)
                normal_cols.append(f'{bcol}-{pcol}')

    train['target'] = 0

    msg = preprocess(fold, train, high_missing_cols, binary_cols, ordinal_cols, normal_cols, one_hot_cols)
    print(msg)

    del train
    gc.collect()
