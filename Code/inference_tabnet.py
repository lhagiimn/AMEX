import gc
import glob
import os
import sys
import time
import traceback
from contextlib import contextmanager
from enum import Enum
from typing import List, Optional
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale
from sklearn import preprocessing
from tqdm import tqdm
from catboost import CatBoostRegressor, Pool
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.autograd import Variable
import torch.optim as optim
import joblib
import random
import pickle
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
import json

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

import warnings
warnings.simplefilter("ignore")


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = 'preprocessed_data/test_preprocessed/'
DATA_DIR_V1 = 'models_tabnet/'
#train_cols = joblib.load(DATA_DIR_V1 + 'train_cols.joblib')
summary = joblib.load('figures/selected_features_v1.joblib')
train_cols = summary['selected_features_names'] + summary['eliminated_features_names'][426:]

cat_cols = joblib.load('preprocessed_data/cat_cols.joblib')
tails = ['last', 'first', 'median', 'lag', 'lag2']
cat_feats = []
for col in cat_cols:
    for t in tails:
        cat_feats.append(f'{col}_{t}')

cat_feats = [f for f in cat_feats if f in train_cols]

preds_tabnet = []
customers = []
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

    customers = customers + df['customer_ID'].to_list()

    df = df.fillna(0)

    categorical_columns = []
    categorical_dims = {}

    for col in tqdm(train_cols):
        if col in cat_feats:
            le = joblib.load(DATA_DIR_V1 + f'le_{col}.pkl')
            df[col] = le.transform(df[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(list(df[col].unique()))

    cat_idxs = [i for i, f in enumerate(df[train_cols].columns.tolist()) if f in categorical_columns]
    cat_dims = [categorical_dims[f] for i, f in enumerate(df[train_cols].columns.tolist()) if f in categorical_columns]

    tabnet_params = dict(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=8,
        n_d=16,
        n_a=16,
        n_steps=5,
        # gamma = 2,
        n_independent=3,
        # n_shared = 2,
        # lambda_sparse = 0,
        optimizer_fn=AdamW,
        optimizer_params=dict(lr=(2e-2)),
        mask_type="entmax",
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=88,
        verbose=5,
        device_name=DEVICE)

    pred = np.zeros(df.shape[0])
    for fold in range(5):
        clf = TabNetClassifier(**tabnet_params)
        clf.load_model(DATA_DIR_V1+f'fold{fold}.zip')
        pred += clf.predict_proba(df[train_cols].values)[:, 1]/5

    preds_tabnet.append(pred)

joblib.dump(preds_tabnet, 'figures/preds_tabnet_v2.pkl')

