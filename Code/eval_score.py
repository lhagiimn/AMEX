import pandas as pd
import numpy as np
import gc
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
import os
import optuna
from optuna.visualization import plot_optimization_history
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, Lasso
import joblib
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import random
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

sub_80 = pd.read_csv('figures/sub/submission_080.csv')
sub_mine = pd.read_csv('submission_stacked.csv')

sub_80 = sub_80.set_index(['customer_ID'])
sub_mine = sub_mine.set_index(['customer_ID'])
sub_80['prediction'] = rankdata(sub_80['prediction'])/sub_80.shape[0]
sub_80['prediction'] = 0.95*sub_80['prediction'].values + 0.05*sub_mine.loc[sub_80.index, 'prediction'].values
sub_80['prediction'] = rankdata(sub_80['prediction'])/sub_80.shape[0]

sub_80.to_csv('submission_ranked.csv')
exit()


def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    gini = gini[1] / gini[0]
    amex = 0.5 * (gini + top_four)


    #msg = f'Gini: {gini}', f'Top 4%: {top_four}', f'Amex metric: {amex}'

    return amex



def get_oof(data_path):
    oof = pd.read_csv(data_path)
    return oof

def get_df(paths, names):
    oofs = []
    for name in names:
        oofs.append(get_oof(paths[name]))

    i=0
    for name, oof in zip(names, oofs):

        if i==0:
            df_oof = oof[['customer_ID', 'target', 'oof']]
            df_oof.columns = ['customer_ID', 'target', f'{name}_oof']
            i=+1
        else:
            if 'XGBoost_pyramid' in name:
                oof = oof.rename(columns={'oof_pred': f'{name}_oof'})
                df_oof[f'{name}_oof'] = oof[f'{name}_oof'].values
            else:
                oof = oof.rename(columns={'oof': f'{name}_oof'})
                df_oof = df_oof.merge(oof[['customer_ID', f'{name}_oof']], on=['customer_ID'], how='left')

    return df_oof

def get_score(df_oof, names, weights):

    y_pred = np.zeros(df_oof.shape[0])
    for name, w in zip(names, weights):
        #df_oof[f'{name}_oof'] = rankdata(df_oof[f'{name}_oof'].values) / df_oof.shape[0]
        print(name, w, amex_metric(df_oof['target'], df_oof[f'{name}_oof']))
        y_pred += w*df_oof[f'{name}_oof'].values

    print('Ensemble:', amex_metric(df_oof['target'].values, y_pred))

    return y_pred



# df = pd.read_csv('neural_network/oof_CNN.csv')
# for fold in range(5):
#     target = pd.read_pickle(f'neural_network/data/targets_{fold}.pkl')
#     target = target.merge(df[['customer_ID', 'oof']], on=['customer_ID'], how='left', copy=False)
#     print(fold, amex_metric(target['target'], target['oof']))
#
# print(amex_metric(df['target'], df['oof']))
# exit()

oofs = {
        'CNN': 'neural_network/oof_CNN.csv',
        'MLP': 'neural_network/oof_MLP.csv',
        'LSTM': 'neural_network/oof_LSTM.csv',
        'LGBM_all': 'models_lgbm/no_feature_selection/df_oof_421.csv',
        'LGBM_2000': 'models_lgbm/2000_features/df_oof_421.csv',
        'LGBM_1500': 'models_lgbm/1500_features/df_oof_421.csv',
        'LGBM_1374': 'models_lgbm/1374_features/df_oof_421.csv',
        'CB_all': 'models_cat/no_feature_selection/df_oof_cb_124.csv',
        'CB_2000': 'models_cat/2000_features/df_oof_cb_124.csv',
        'CB_1500': 'models_cat/1500_features/df_oof_cb_124.csv',
        'CB_1200': 'models_cat/1200_features/df_oof_cb_124.csv',
        'CB_1374': 'models_cat/1374_features/df_oof_cb_124.csv',
        'CB_1082': 'models_cat/1082_features/df_oof_cb_124.csv',
        'LGBM_dart_2000': 'models_lgbm_dart/2000_features/df_oof_925.csv',
        'LGBM_dart_1500': 'models_lgbm_dart/1500_features/df_oof_925.csv',
        'LGBM_dart_1374': 'models_lgbm_dart/1374_features/df_oof_925.csv',
        'XGBoost_2000': 'models_xgb/2000_features/df_oof_cb_42.csv',
        'XGBoost_1500': 'models_xgb/1500_features/df_oof_42.csv',
        'XGBoost_1374': 'models_xgb/1374_features/df_oof_xgboost.csv',
        'XGBoost_1082': 'models_xgb/1082_features/df_oof_xgboost.csv',
        'XGBoost_pyramid_v1': 'models_xgb/oof_xgb_v1.csv',
        'XGBoost_pyramid_v2': 'models_xgb/oof_xgb_v3.csv',
        "tabnet_2000": 'models_tabnet/v1/df_oof_42.csv',
        "tabnet_1374": 'models_tabnet/df_oof_42.csv'
        }

names = ['LGBM_2000',  'LGBM_1500', 'LGBM_1374',
         'LGBM_dart_2000', 'LGBM_dart_1500', 'LGBM_dart_1374',
          'CB_2000',  'CB_1500', 'CB_1200', 'CB_1374', 'CB_1082',
         'XGBoost_2000', 'XGBoost_1500', 'XGBoost_1374', 'XGBoost_1082',
         'XGBoost_pyramid_v1', 'XGBoost_pyramid_v2',
          'tabnet_1374', 'CNN']

weights = [0.00, 0.00, 0.05,
           0.15, 0.00, 0.15,
           0.00, 0.00, 0.00, 0.10, 0.10,
           0.00, 0.00, 0.05, 0.05,
           0.00, 0.30,
           0.025, 0.025]

print(np.sum(weights))

df_oof = get_df(oofs, names)
y_ensemble = (get_score(df_oof, names, weights))

df_oof['oof'] = y_ensemble

df_oof[['customer_ID', 'oof']].to_csv('oof.csv', index=False)


n_folds = 10
seed_list = [i for i in range(2015, 2022)]


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


set_seed(seed_list[0])

oof_list = list()
preds_list = list()
cols = []
for n in names:
    cols.append(f'{n}_oof')

print(cols)
#df_oof[cols].corr().to_csv('corr.csv')

for seed in seed_list:
    oof = np.zeros(df_oof.shape[0])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(df_oof, df_oof['target'])):
        X_train, y_train = df_oof[cols].to_numpy()[train_idx], df_oof['target'].to_numpy()[train_idx]
        X_valid, y_valid = df_oof[cols].to_numpy()[valid_idx], df_oof['target'].to_numpy()[valid_idx]

        model = CalibratedClassifierCV(
            RidgeClassifier(alpha=1, random_state=42),
            cv=10
        )
        model.fit(
            X_train,
            y_train,
        )

        joblib.dump(model, f'figures/nn/model_{fold}_{seed}.pkl')

        oof[valid_idx] = model.predict_proba(X_valid)[:,-1]

    auc = amex_metric(df_oof['target'], oof)
    print(f"SEED {seed}: AMEX {auc:.6f}")

    oof_list.append(oof)


auc = amex_metric(df_oof['target'], np.mean(rankdata(oof_list, axis=1), axis=0))
print(f"SEED AVERAGING AMEX {auc:.6f}")


# auc = amex_metric(df_oof['target'], 0.5*np.mean(oof_list, axis=0) + 0.5*y_ensemble)
# print(f"Ensemble {auc:.6f}")






