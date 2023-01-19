import pandas as pd
import numpy as np
import gc
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from catboost import EShapCalcType, EFeaturesSelectionAlgorithm
import os
import optuna
from tqdm import tqdm
import joblib
from optuna.visualization import plot_optimization_history
import matplotlib.pyplot as plt


summary = joblib.load('figures/selected_features_v2.joblib')
print(summary['loss_graph']['main_indices'])


# plt.plot(summary['loss_graph'])
# plt.show()

exit()


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

summary = joblib.load('figures/selected_features_v1.joblib')
train_cols = summary['selected_features_names'] #+ summary['eliminated_features_names'][426:]
print(len(train_cols))

TARGET='target'
target = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(df[train_cols], target,
                                                    test_size=0.2, random_state=24,
                                                    shuffle=True, stratify=target)

model_cb = CatBoostClassifier(
            n_estimators=5000,
            learning_rate=0.02,
            loss_function='Logloss',
            eval_metric='Logloss',
            random_seed = 42,
            max_depth=8,
            task_type='GPU',
            verbose=500)


train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)
print(X_train.shape)

summary = model_cb.select_features(
                                    train_pool,
                                    eval_set=test_pool,
                                    features_for_select=list(range(X_train.shape[1])),
                                    num_features_to_select=975,
                                    steps=2,
                                    #verbose = True,
                                    algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
                                    shap_calc_type=EShapCalcType.Regular,
                                    train_final_model=True,
                                    logging_level='Verbose',
                                    plot=False
                                )

print(summary['selected_features_names'])
joblib.dump(summary, 'figures/selected_features_v2.joblib')