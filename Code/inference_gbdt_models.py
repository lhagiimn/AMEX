import pandas as pd
import numpy as np
import gc
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import os
from scipy.misc import derivative
import joblib
from scipy.stats import rankdata
import xgboost as xgb
import pathlib
import torch
import random
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from catboost import CatBoostClassifier, Pool

import warnings
warnings.filterwarnings("ignore")


# ### LGBM models All features are used #####
# Nfolds = 5
# seed = 421
# model_path = 'models_lgbm/no_feature_selection/'
# models_lgbm_all = []
#
# for fold in range(Nfolds):
#     with open(model_path + f'model_lgb_{fold}_{seed}.pkl', 'rb') as fin:
#         model = pickle.load(fin)
#     models_lgbm_all.append(model)
#
# ### LGBM models 2000 features are used #####
# Nfolds = 5
# seed = 421
# model_path = 'models_lgbm/2000_features/'
# models_lgbm_2000  = []
#
# for fold in range(Nfolds):
#     fin = model_path + f'model_lgb_{fold}_{seed}.pkl'
#     model = joblib.load(fin)
#     models_lgbm_2000.append(model)
#
# ### LGBM models 1500 features are used #####
# Nfolds = 5
# seed = 421
# model_path = 'models_lgbm/1500_features/'
# models_lgbm_1500  = []
#
# for fold in range(Nfolds):
#     with open(model_path + f'model_lgb_{fold}_{seed}.pkl', 'rb') as fin:
#         model = pickle.load(fin)
#     models_lgbm_1500.append(model)

### LGBM models 1374 features are used #####
# Nfolds = 5
# seed = 421
# model_path = 'models_lgbm/1374_features/'
# models_lgbm_1374  = []
#
# for fold in range(Nfolds):
#     with open(model_path + f'model_lgb_{fold}_{seed}.pkl', 'rb') as fin:
#         model = pickle.load(fin)
#     models_lgbm_1374.append(model)

# ### LGBM models 2000 features are used #####
# Nfolds = 5
# seed = 925
# model_path = 'models_lgbm_dart/2000_features/'
# models_lgbm_dart_2000  = []
#
# for fold in range(Nfolds):
#     fin = model_path + f'model_lgb_{fold}_{seed}.pkl'
#     model = joblib.load(fin)
#     models_lgbm_dart_2000.append(model)
#
# ### LGBM models 1500 features are used #####
# Nfolds = 5
# seed = 925
# model_path = 'models_lgbm_dart/1500_features/'
# models_lgbm_dart_1500  = []
#
# for fold in range(Nfolds):
#     fin = model_path + f'model_lgb_{fold}_{seed}.pkl'
#     model = joblib.load(fin)
#     models_lgbm_dart_1500.append(model)

### LGBM models 1374 features are used #####
Nfolds = 5
seed = 925
model_path = 'models_lgbm_dart/1374_features/'
models_lgbm_dart_1374  = []

for fold in range(Nfolds):
    fin = model_path + f'model_lgb_{fold}_{seed}.pkl'
    model = joblib.load(fin)
    models_lgbm_dart_1374.append(model)

# ### Catboost models  All features are used ####
# Nfolds = 10
# seed = 124
# model_path = 'models_cat/no_feature_selection/'
# models_cb_all  = []
#
# for fold in range(Nfolds):
#     with open(model_path + f'model_cb_{fold}_{seed}.pkl', 'rb') as fin:
#         model = pickle.load(fin)
#     models_cb_all.append(model)
#
# ### Catboost models  2000 features are used  ####
# Nfolds = 10
# seed = 124
# model_path = 'models_cat/2000_features/'
# models_cb_2000  = []
#
# for fold in range(Nfolds):
#     with open(model_path + f'model_cb_{fold}_{seed}.pkl', 'rb') as fin:
#         model = pickle.load(fin)
#     models_cb_2000.append(model)
#
# ### Catboost models 1500 features are used#####
# Nfolds = 10
# seed = 124
# model_path = 'models_cat/1500_features/'
# models_cb_1500  = []
#
# for fold in range(Nfolds):
#     with open(model_path + f'model_cb_{fold}_{seed}.pkl', 'rb') as fin:
#         model = pickle.load(fin)
#     models_cb_1500.append(model)

### Catboost models 1200 features are used#####
# Nfolds = 10
# seed = 124
# model_path = 'models_cat/1200_features/'
# models_cb_1200  = []
#
# for fold in range(Nfolds):
#     with open(model_path + f'model_cb_{fold}_{seed}.pkl', 'rb') as fin:
#         model = pickle.load(fin)
#     models_cb_1200.append(model)

### Catboost models 1374 features are used#####
# Nfolds = 10
# seed = 124
# model_path = 'models_cat/1374_features/'
# models_cb_1374  = []
#
# for fold in range(Nfolds):
#     with open(model_path + f'model_cb_{fold}_{seed}.pkl', 'rb') as fin:
#         model = pickle.load(fin)
#     models_cb_1374.append(model)

## Catboost models 1082 features are used#####
Nfolds = 10
seed = 124
model_path = 'models_cat/1082_features/'
models_cb_1082  = []

for fold in range(Nfolds):
    with open(model_path + f'model_cb_{fold}_{seed}.pkl', 'rb') as fin:
        model = pickle.load(fin)
    models_cb_1082.append(model)

# ### XGBoost models 2000 features are used ####
# Nfolds = 5
# seed = 42
# model_path = 'models_xgb/2000_features/'
# models_xg_2000  = []
#
# for fold in range(Nfolds):
#     with open(model_path + f'model_xg_{fold}.pkl', 'rb') as fin:
#         model = pickle.load(fin)
#     models_xg_2000.append(model)
#
# ### XGBoost models 1500 features are used ####
# Nfolds = 5
# seed = 42
# model_path = 'models_xgb/1500_features/'
# models_xg_1500  = []
#
# for fold in range(Nfolds):
#     with open(model_path + f'model_xg_{fold}.pkl', 'rb') as fin:
#         model = pickle.load(fin)
#     models_xg_1500.append(model)

### XGBoost models 1374 features are used ####
# Nfolds = 5
# seed = 42
# model_path = 'models_xgb/1374_features/'
# models_xg_1374  = []
#
# for fold in range(Nfolds):
#     with open(model_path + f'model_xg_{fold}.pkl', 'rb') as fin:
#         model = pickle.load(fin)
#     models_xg_1374.append(model)

### XGBoost models 1082 features are used ####
Nfolds = 5
seed = 42
model_path = 'models_xgb/1082_features/'
models_xg_1082  = []

for fold in range(Nfolds):
    with open(model_path + f'model_xg_{fold}.pkl', 'rb') as fin:
        model = pickle.load(fin)
    models_xg_1082.append(model)

train_cols_all = joblib.load('models_cat/no_feature_selection/train_cols.joblib')
train_cols_2000 = joblib.load('models_cat/2000_features/train_cols.joblib')
train_cols_1500 = joblib.load('models_cat/1500_features/train_cols.joblib')

summary = joblib.load('figures/selected_features_v1.joblib')
train_cols_1200 = summary['selected_features_names']
train_cols_1374 = summary['selected_features_names'] + summary['eliminated_features_names'][426:]

summary = joblib.load('figures/selected_features_v2.joblib')
train_cols_1082 = summary['selected_features_names'] + summary['eliminated_features_names'][118:]

### Prediction #############
preds_lgbm_all = []
preds_lgbm_2000 = []
preds_lgbm_1500 = []
preds_lgbm_1374 = []
preds_cb_all = []
preds_cb_2000 = []
preds_cb_1500 = []
preds_cb_1200 = []
preds_cb_1374 = []
preds_cb_1082 = []
preds_dart_2000 = []
preds_dart_1500 = []
preds_dart_1374 = []
preds_xg_2000 = []
preds_xg_1500 = []
preds_xg_1374 = []
preds_xg_1082 = []
customers = []

PREDICT = False

if PREDICT:
    DATA_DIR = 'preprocessed_data/test_preprocessed/'

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


        customers = customers + df['customer_ID'].to_list()

        # print('predicting LGBM models all...')
        # pred_lgbm_all = np.zeros(df.shape[0])
        # for model in tqdm(models_lgbm_all):
        #     pred_lgbm_all += model.predict(df[train_cols_all])/len(models_lgbm_all)
        #
        # print('predicting LGBM models 2000 features...')
        # pred_lgbm_2000 = np.zeros(df.shape[0])
        # for model in tqdm(models_lgbm_2000):
        #     pred_lgbm_2000 += model.predict(df[train_cols_2000]) / len(models_lgbm_2000)
        #
        # print('predicting LGBM models 1500 features...')
        # pred_lgbm_1500 = np.zeros(df.shape[0])
        # for model in tqdm(models_lgbm_1500):
        #     pred_lgbm_1500 += model.predict(df[train_cols_1500]) / len(models_lgbm_1500)

        # print('predicting LGBM models 1374 features...')
        # pred_lgbm_1374 = np.zeros(df.shape[0])
        # for model in tqdm(models_lgbm_1374):
        #     pred_lgbm_1374 += model.predict(df[train_cols_1374]) / len(models_lgbm_1374)

        # print('predicting LGBM dart models 2000 features...')
        # pred_dart_2000 = np.zeros(df.shape[0])
        # for model in tqdm(models_lgbm_dart_2000):
        #     pred_dart_2000 += model.predict(df[train_cols_2000]) / len(models_lgbm_dart_2000)
        #
        # print('predicting LGBM dart models 1500 features...')
        # pred_dart_1500 = np.zeros(df.shape[0])
        # for model in tqdm(models_lgbm_dart_1500):
        #     pred_dart_1500 += model.predict(df[train_cols_1500]) / len(models_lgbm_dart_1500)

        print('predicting LGBM dart models 1374 features...')
        pred_dart_1374 = np.zeros(df.shape[0])
        num_iters = [7600, 7600, 7800, 7600, 7600]
        iteration = 0
        for model in tqdm(models_lgbm_dart_1374):
            pred_dart_1374 += model.predict(df[train_cols_1374], num_iteration = num_iters[iteration]) / len(models_lgbm_dart_1374)
            iteration +=1

        # print('predicting CATBOOST models all...')
        # pred_cat_all = np.zeros(df.shape[0])
        # for model in tqdm(models_cb_all):
        #     pred_cat_all += model.predict_proba(df[train_cols_all])[:, 1] / len(models_cb_all)
        #
        # print('predicting CATBOOST models 2000 features...')
        # pred_cat_2000 = np.zeros(df.shape[0])
        # for model in tqdm(models_cb_2000):
        #     pred_cat_2000 += model.predict_proba(df[train_cols_2000])[:, 1] / len(models_cb_2000)
        #
        # print('predicting CATBOOST models 1500 features...')
        # pred_cat_1500 = np.zeros(df.shape[0])
        # for model in tqdm(models_cb_1500):
        #     pred_cat_1500 += model.predict_proba(df[train_cols_1500])[:, 1] / len(models_cb_1500)

        # print('predicting CATBOOST models 1200 features...')
        # pred_cat_1200 = np.zeros(df.shape[0])
        # for model in tqdm(models_cb_1200):
        #     pred_cat_1200 += model.predict_proba(df[train_cols_1200])[:, 1] / len(models_cb_1200)
        #
        # print('predicting CATBOOST models 1374 features...')
        # pred_cat_1374 = np.zeros(df.shape[0])
        # for model in tqdm(models_cb_1374):
        #     pred_cat_1374 += model.predict_proba(df[train_cols_1374])[:, 1] / len(models_cb_1374)

        print('predicting CATBOOST models 1082 features...')
        pred_cat_1082 = np.zeros(df.shape[0])
        for model in tqdm(models_cb_1082):
            pred_cat_1082 += model.predict_proba(df[train_cols_1082])[:, 1] / len(models_cb_1082)
            
        # print('predicting xgboost models 2000 features...')
        # pred_xg_2000 = np.zeros(df.shape[0])
        # for model in tqdm(models_xg_2000):
        #     pred_xg_2000 += model.predict(xgb.DMatrix(df[train_cols_2000])) / len(models_xg_2000)
        #
        # print('predicting xgboost models 1500 features...')
        # pred_xg_1500 = np.zeros(df.shape[0])
        # for model in tqdm(models_xg_1500):
        #     pred_xg_1500 += model.predict(xgb.DMatrix(df[train_cols_1500])) / len(models_xg_1500)

        # print('predicting xgboost models 1374 features...')
        # pred_xg_1374 = np.zeros(df.shape[0])
        # for model in tqdm(models_xg_1374):
        #     pred_xg_1374 += model.predict(xgb.DMatrix(df[train_cols_1374])) / len(models_xg_1374)

        print('predicting xgboost models 1082 features...')
        pred_xg_1082 = np.zeros(df.shape[0])
        for model in tqdm(models_xg_1082):
            pred_xg_1082 += model.predict(xgb.DMatrix(df[train_cols_1082])) / len(models_xg_1082)

        # preds_lgbm_all.append(pred_lgbm_all)
        # preds_lgbm_2000.append(pred_lgbm_2000)
        # preds_lgbm_1500.append(pred_lgbm_1500)
        # preds_lgbm_1374.append(pred_lgbm_1374)
        # preds_cb_all.append(pred_cat_all)
        # preds_cb_2000.append(pred_cat_2000)
        # preds_cb_1500.append(pred_cat_1500)
        # preds_cb_1200.append(pred_cat_1200)
        # preds_cb_1374.append(pred_cat_1374)
        preds_cb_1082.append(pred_cat_1082)
        # preds_dart_2000.append(pred_dart_2000)
        # preds_dart_1500.append(pred_dart_1500)
        preds_dart_1374.append(pred_dart_1374)
        # preds_xg_2000.append(pred_xg_2000)
        # preds_xg_1500.append(pred_xg_1500)
        # preds_xg_1374.append(pred_xg_1374)
        preds_xg_1082.append(pred_xg_1082)



    # joblib.dump(customers, 'figures/customers.pkl')
    # joblib.dump(preds_lgbm_all, 'figures/preds_lgbm_all.pkl')
    # joblib.dump(preds_lgbm_2000, 'figures/preds_lgbm_2000.pkl')
    # joblib.dump(preds_lgbm_1500, 'figures/preds_lgbm_1500.pkl')
    # joblib.dump(preds_lgbm_1374, 'figures/preds_lgbm_1374.pkl')
    # joblib.dump(preds_cb_all, 'figures/preds_cb_all.pkl')
    # joblib.dump(preds_cb_2000, 'figures/preds_cb_2000.pkl')
    # joblib.dump(preds_cb_1500, 'figures/preds_cb_1500.pkl')
    # joblib.dump(preds_cb_1200, 'figures/preds_cb_1200.pkl')
    # joblib.dump(preds_cb_1374, 'figures/preds_cb_1374.pkl')
    joblib.dump(preds_cb_1082, 'figures/preds_cb_1082.pkl')
    # joblib.dump(preds_dart_2000, 'figures/preds_dart_2000.pkl')
    # joblib.dump(preds_dart_1500, 'figures/preds_dart_1500.pkl')
    joblib.dump(preds_dart_1374, 'figures/preds_dart_1374.pkl')
    # joblib.dump(preds_xg_2000, 'figures/preds_xg_2000.pkl')
    # joblib.dump(preds_xg_1500, 'figures/preds_xg_1500.pkl')
    # joblib.dump(preds_xg_1374, 'figures/preds_xg_1374.pkl')
    joblib.dump(preds_xg_1082, 'figures/preds_xg_1082.pkl')


else:

    customers = joblib.load('figures/customers.pkl')
    preds_lgbm_all = joblib.load('figures/preds_lgbm_all.pkl')
    preds_lgbm_2000 = joblib.load('figures/preds_lgbm_2000.pkl')
    preds_lgbm_1500 = joblib.load('figures/preds_lgbm_1500.pkl')
    preds_lgbm_1374 = joblib.load('figures/preds_lgbm_1374.pkl')
    preds_cb_all = joblib.load('figures/preds_cb_all.pkl')
    preds_cb_2000 = joblib.load('figures/preds_cb_2000.pkl')
    preds_cb_1500 = joblib.load('figures/preds_cb_1500.pkl')
    preds_cb_1200 = joblib.load('figures/preds_cb_1200.pkl')
    preds_cb_1374 = joblib.load('figures/preds_cb_1374.pkl')
    preds_cb_1082 = joblib.load('figures/preds_cb_1082.pkl')
    preds_dart_2000 = joblib.load('figures/preds_dart_2000.pkl')
    preds_dart_1500 = joblib.load('figures/preds_dart_1500.pkl')
    preds_dart_1374 = joblib.load('figures/preds_dart_1374.pkl')
    preds_xg_2000 = joblib.load('figures/preds_xg_2000.pkl')
    preds_xg_1500 = joblib.load('figures/preds_xg_1500.pkl')
    preds_xg_1374 = joblib.load('figures/preds_xg_1374.pkl')
    preds_xg_1082 = joblib.load('figures/preds_xg_1082.pkl')
    preds_tabnet = joblib.load('figures/preds_tabnet.pkl')
    preds_tabnet_v2 = joblib.load('figures/preds_tabnet_v2.pkl')
    preds_cnn = joblib.load('figures/preds_CNN.pkl')

    preds_lgbm_all = np.concatenate(preds_lgbm_all)
    preds_lgbm_2000 = np.concatenate(preds_lgbm_2000)
    preds_lgbm_1500 = np.concatenate(preds_lgbm_1500)
    preds_lgbm_1374 = np.concatenate(preds_lgbm_1374)
    preds_cb_all = np.concatenate(preds_cb_all)
    preds_cb_2000 = np.concatenate(preds_cb_2000)
    preds_cb_1500 = np.concatenate(preds_cb_1500)
    preds_cb_1200 = np.concatenate(preds_cb_1200)
    preds_cb_1374 = np.concatenate(preds_cb_1374)
    preds_cb_1082 = np.concatenate(preds_cb_1082)
    preds_dart_2000 = np.concatenate(preds_dart_2000)
    preds_dart_1500 = np.concatenate(preds_dart_1500)
    preds_dart_1374 = np.concatenate(preds_dart_1374)
    preds_xg_2000 = np.concatenate(preds_xg_2000)
    preds_xg_1500 = np.concatenate(preds_xg_1500)
    preds_xg_1374 = np.concatenate(preds_xg_1374)
    preds_xg_1082 = np.concatenate(preds_xg_1082)
    preds_tabnet_2000 = np.concatenate(preds_tabnet)
    preds_tabnet_1374 = np.concatenate(preds_tabnet_v2)
    preds_cnn = preds_cnn.set_index('customer_ID')



    test = pd.DataFrame(index=customers,data={ 'preds_lgbm_all': preds_lgbm_all,
                                               'preds_lgbm_2000': preds_lgbm_2000,
                                               'preds_lgbm_1500':preds_lgbm_1500,
                                               'preds_lgbm_1374': preds_lgbm_1374,
                                               'preds_cb_all':preds_cb_all,
                                               'preds_cb_2000': preds_cb_2000,
                                               'preds_cb_1500': preds_cb_1500,
                                               'preds_cb_1200': preds_cb_1200,
                                               'preds_cb_1374': preds_cb_1374,
                                               'preds_cb_1082': preds_cb_1082,
                                               'preds_dart_2000': preds_dart_2000,
                                               'preds_dart_1500': preds_dart_1500,
                                               'preds_dart_1374': preds_dart_1374,
                                               'preds_xg_2000': preds_xg_2000,
                                               'preds_xg_1500': preds_xg_1500,
                                               'preds_xg_1374': preds_xg_1374,
                                               'preds_xg_1082': preds_xg_1082,
                                               'preds_tabnet_2000': preds_tabnet_2000,
                                               'preds_tabnet_1374': preds_tabnet_1374,
                                               'preds_cnn':preds_cnn.loc[customers, 'prediction'].values,
                                              })


    # WRITE SUBMISSION FILE
    with open('preprocessed_data/encoder.pkl', 'rb') as fin:
        encoder = pickle.load(fin)

    ## joing xgboost pyramid models
    sub_pyramid = pd.read_csv('models_xgb/submission_pyramid_cv0.7987.csv')
    sub_pyramid['customer_ID'] = encoder.transform(sub_pyramid['customer_ID'])
    sub_pyramid = sub_pyramid.set_index('customer_ID')
    test['preds_xg_pyramid_v1'] = sub_pyramid.loc[test.index, 'prediction'].values

    sub_pyramid = pd.read_csv('models_xgb/submission_pyramid_cv0.7995.csv')
    sub_pyramid['customer_ID'] = encoder.transform(sub_pyramid['customer_ID'])
    sub_pyramid = sub_pyramid.set_index('customer_ID')
    test['preds_xg_pyramid_v2'] = sub_pyramid.loc[test.index, 'prediction'].values



    test['prediction'] = 0.00 * test['preds_lgbm_all'].values + \
                         0.00 * test['preds_lgbm_2000'].values + \
                         0.00 * test['preds_lgbm_1500'].values + \
                         0.05 * test['preds_lgbm_1374'].values + \
                         0.00 * test['preds_cb_all'].values + \
                         0.00 * test['preds_cb_2000'].values + \
                         0.00 * test['preds_cb_1500'].values + \
                         0.00 * test['preds_cb_1200'].values + \
                         0.05 * test['preds_cb_1374'].values + \
                         0.05 * test['preds_cb_1082'].values + \
                         0.25 * test['preds_dart_2000'].values + \
                         0.20 * test['preds_dart_1500'].values + \
                         0.25 * test['preds_dart_1374'].values + \
                         0.00 * test['preds_xg_2000'].values + \
                         0.00 * test['preds_xg_1500'].values + \
                         0.00 * test['preds_xg_1374'].values + \
                         0.05 * test['preds_xg_1082'].values + \
                         0.00 * test['preds_xg_pyramid_v1'].values + \
                         0.05 * test['preds_xg_pyramid_v2'].values + \
                         0.00 * test['preds_tabnet_2000'].values + \
                         0.025 * test['preds_tabnet_1374'].values + \
                         0.025 * test['preds_cnn'].values

    '''
    names = ['LGBM_2000_oof', 'LGBM_1500_oof', 'LGBM_1374_oof',
             'LGBM_dart_2000_oof', 'LGBM_dart_1500_oof', 'LGBM_dart_1374_oof',
             'CB_2000_oof', 'CB_1500_oof', 'CB_1200_oof', 'CB_1374_oof', 'CB_1082_oof',
             'XGBoost_2000_oof', 'XGBoost_1500_oof', 'XGBoost_1374_oof', 'XGBoost_1082_oof',
             'XGBoost_pyramid_v1_oof', 'XGBoost_pyramid_v2_oof',
             'tabnet_1374_oof', 'CNN_oof']


    cols = ['preds_lgbm_2000', 'preds_lgbm_1500', 'preds_lgbm_1374',
            'preds_dart_2000', 'preds_dart_1500', 'preds_dart_1374',
             'preds_cb_2000', 'preds_cb_1500', 'preds_cb_1200', 'preds_cb_1374', 'preds_cb_1082',
            'preds_xg_2000', 'preds_xg_1500', 'preds_xg_1374', 'preds_xg_1082',
            'preds_xg_pyramid_v1', 'preds_xg_pyramid_v2',
            'preds_tabnet_1374', 'preds_cnn']


    n_folds = 10
    seed_list = [i for i in range(2015, 2022)]

    preds_list = list()
    for seed in seed_list:
        pred = np.zeros(test.shape[0])
        for fold in range(n_folds):
            model = joblib.load(f'figures/nn/model_{fold}_{seed}.pkl')
            pred = pred + model.predict_proba(test[cols].values)[:, 1]/n_folds

        preds_list.append(pred)

    preds = np.mean(rankdata(preds_list, axis=1), axis=0) / test.shape[0]
    test['prediction'] = preds
    '''

    sub = pd.read_csv('preprocessed_data/sample_submission.csv')[['customer_ID']]
    sub['customer_ID_hash'] = encoder.transform(sub['customer_ID'])
    sub = sub.set_index('customer_ID_hash')
    sub['prediction'] = test.loc[sub.index, 'prediction']
    sub = sub.reset_index(drop=True)

    # DISPLAY PREDICTIONS
    sub.to_csv('submission_stacked.csv',index=False)
    print('Submission file shape is', sub.shape )
    print(sub.isnull().sum())
    print(sub.head())
    plt.hist(sub['prediction'])
    plt.show()
