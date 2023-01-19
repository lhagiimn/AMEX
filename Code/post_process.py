import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import gc
import pickle
from tqdm import tqdm
from scipy import stats
from numpy.fft import *
from scipy.stats import skew
import joblib
import matplotlib.pylab as plt
from sklearn.neighbors import KNeighborsRegressor
from scipy import special
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from scipy.stats import rankdata
import numpy as np

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

DATA_DIR = 'preprocessed_data/'
train = pd.read_parquet(DATA_DIR + 'train.parquet')
train = train.loc[train['month']==train['month'].max()]

print('merging target...')
with open(DATA_DIR + 'encoder.pkl', 'rb') as fin:
    encoder = pickle.load(fin)

labels = pd.read_csv(DATA_DIR+ 'train_labels.csv')
labels['customer_ID'] = encoder.transform(labels['customer_ID'])
train = train.merge(labels, on=['customer_ID'], how='left')

print('merging oof...')
oof = pd.read_csv('oof.csv')
train = train.merge(oof[['customer_ID', 'oof']], on=['customer_ID'], how='left')

del encoder, labels, oof
gc.collect()

print(amex_metric(train['target'], train['oof']))

train['oof_rank'] = rankdata(train['oof'])

for col in ['D_39']:
    train.loc[train[col]>= train[col].quantile(0.99), 'oof_rank']=train.loc[train[col]>= train[col].quantile(0.99), 'oof_rank'] + 5000

    print(col, amex_metric(train['target'], train['oof_rank']/train.shape[0]))

for col in ['P_2']:
    train.loc[train[col]<= train[col].quantile(0.01), 'oof_rank']=train.loc[train[col]<= train[col].quantile(0.01), 'oof_rank'] + 500

    print(col, amex_metric(train['target'], train['oof_rank']/train.shape[0]))

print(amex_metric(train['target'], train['oof_rank']/train.shape[0]))