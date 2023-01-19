import pandas as pd
import numpy as np
import gc
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
import os
import optuna
from tqdm import tqdm
import joblib
from optuna.visualization import plot_optimization_history
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")




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
    return 0.5 * (gini[1]/gini[0] + top_four)

# from ambrosm notebook
def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(pd.DataFrame({'target': y_true}), pd.Series(y_pred, name='prediction')),
            True)


DATA_DIR = 'preprocessed_data/'

print('merging categorical and high missing datasets...')
df_cat = pd.read_pickle(DATA_DIR + 'train_cat.pkl')
temp = pd.read_pickle(DATA_DIR + 'train_num_high.pkl')
temp = temp.merge(df_cat, on = ['customer_ID'], how='left')

del df_cat
gc.collect()

print('merging binary dataset...')
df_binary = pd.read_pickle(DATA_DIR + 'train_num_binary.pkl')
temp = temp.merge(df_binary, on = ['customer_ID'], how='left')

del df_binary
gc.collect()

print('merging ordinal dataset...')
df_ordinal = pd.read_pickle(DATA_DIR + 'train_num_ordinal.pkl')
temp = temp.merge(df_ordinal, on = ['customer_ID'], how='left')

del df_ordinal
gc.collect()

print('merging max month dataset...')
df_month_max = pd.read_pickle(DATA_DIR + 'train_num_month_at_max.pkl')
temp = temp.merge(df_month_max, on = ['customer_ID'], how='left')

del df_month_max
gc.collect()

print('merging min month dataset...')
df_month_min = pd.read_pickle(DATA_DIR + 'train_num_month_at_min.pkl')
temp = temp.merge(df_month_min, on = ['customer_ID'], how='left')

del df_month_min
gc.collect()

print('merging weighted max month dataset...')
df_month_max = pd.read_pickle(DATA_DIR + 'train_num_month_at_max_weighted.pkl')
temp = temp.merge(df_month_max, on = ['customer_ID'], how='left')

del df_month_max
gc.collect()

print('merging weighted min month dataset...')
df_month_min = pd.read_pickle(DATA_DIR + 'train_num_month_at_min_weighted.pkl')
temp = temp.merge(df_month_min, on = ['customer_ID'], how='left')

del df_month_min
gc.collect()

print('merging normal dataset...')
df = pd.read_pickle(DATA_DIR + 'train_num_good.pkl')
df = df.merge(temp, on = ['customer_ID'], how='left')

del temp
gc.collect()

print(df.shape)

df = df.replace([np.inf, -np.inf], np.nan)

summary = joblib.load('figures/selected_features_v1.joblib')
train_cols = summary['selected_features_names'] + summary['eliminated_features_names'][426:]
print(len(train_cols))
df = df[['customer_ID']+train_cols]
gc.collect()

print('merging target...')
with open(DATA_DIR + 'encoder.pkl', 'rb') as fin:
    encoder = pickle.load(fin)

labels = pd.read_csv(DATA_DIR+ 'train_labels.csv')
labels['customer_ID'] = encoder.transform(labels['customer_ID'])

df = df.merge(labels, on=['customer_ID'], how='left')

del encoder, labels
gc.collect()

print(df.shape)

train_cols = list(df.columns)
train_cols.remove('customer_ID')
train_cols.remove('target')

TARGET='target'

'''
def objective(trial, df=df[train_cols],  target=df[TARGET]):

    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=24,
                                                        shuffle=True, stratify=target)

    train_data = lgb.Dataset(data=X_train,
                             label=y_train,
                             free_raw_data=False)

    valid_data = lgb.Dataset(data=X_test,
                             label=y_test,
                             free_raw_data=False)

    params = {
        'objective': 'binary',
        'boosting': 'dart',
        'metric': 'auc',
        'random_state': 24,
        'n_estimators': 2000,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.2,0.3,0.4,0.5,0.6,0.7,0.8]),
        'subsample': trial.suggest_categorical('subsample', [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]),
        'drop_rate': trial.suggest_float('drop_rate', 0.05, 0.50),
        'max_drop': trial.suggest_int('max_drop', 5, 100),
        'learning_rate': 0.025,
        'num_leaves': trial.suggest_int('num_leaves', 1, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
    }

    model = lgb.train(params, train_data, valid_sets=[train_data, valid_data],
                      num_boost_round=2000, early_stopping_rounds=200,
                      verbose_eval=100)

    preds = model.predict(X_test)

    score = amex_metric(y_test, preds)

    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
print('Best score:', study.best_trial.value)


def objective(trial, df=df[train_cols],  target=df[TARGET]):

    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=24,
                                                        shuffle=True, stratify=target)

    params = {"l2_leaf_reg": trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
              "random_strength": trial.suggest_loguniform('random_strength', 1e-3, 10.0),
              "bagging_temperature": trial.suggest_float("bagging_temperature", 1e-3, 10.0),
              "n_estimators": 2500,
              "learning_rate": 0.025,
              "task_type": 'GPU'}

    gbm = CatBoostClassifier(**params)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)],
            verbose=500, early_stopping_rounds=100)

    preds = gbm.predict_proba(X_test)[:, 1]

    score = amex_metric(y_test, preds)

    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
print('Best score:', study.best_trial.value)

'''


def objective(trial, df=df[train_cols],  target=df[TARGET]):

    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=24,
                                                        shuffle=True, stratify=target)

    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_test, y_test)
    
    del df, target, X_train, y_train
    gc.collect()
    
    watchlist = [(dvalid, 'valid')]
    
    param = {
        'tree_method':'gpu_hist', 
         'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'predictor': 'gpu_predictor',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': 0.025,
        'max_depth': 8,
        'random_state': trial.suggest_categorical('random_state', [42]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    }


    model = xgb.train(param, dtrain, 2000, watchlist, early_stopping_rounds=300, verbose_eval=100)

    preds = model.predict(xgb.DMatrix(X_test))

    score = amex_metric(y_test, preds)

    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
print('Best score:', study.best_trial.value)