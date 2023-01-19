import os, copy, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import random


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed=45)

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class AMEXLoader:

    def __init__(self, X_3D, X_2D, y, lag, shuffle=False, batch_size=1024):
        self.X_3D = X_3D
        self.X_2D = X_2D
        self.y = y
        self.lag = lag

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_conts = self.X_3D.shape[1]
        self.len = self.X_3D.shape[0]
        n_batches, remainder = divmod(self.len, self.batch_size)

        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        self.remainder = remainder  # for debugging

        self.idxes = np.array([i for i in range(self.len)])

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            ridxes = self.idxes
            np.random.shuffle(ridxes)
            self.X_3D = self.X_3D[ridxes]
            self.X_2D = self.X_2D[ridxes]
            if self.y is not None:
                self.y = self.y[ridxes]

        return self

    def __next__(self):
        if self.i >= self.len:
            raise StopIteration

        X_3D = torch.FloatTensor(self.X_3D[self.i:self.i + self.batch_size, self.lag:, :])
        X_2D = torch.FloatTensor(self.X_2D[self.i:self.i + self.batch_size, :])
        # idx = np.random.randint(self.len, size=xcont1.shape[0])
        if self.y is not None:
            y1 = self.y[self.i:self.i + self.batch_size]
            # y2 = self.y[idx]
            # y = torch.FloatTensor(np.where(y1==y2, 1, 0).astype(np.float32))
            y1 = torch.FloatTensor(y1.astype(np.float32))

        else:
            y1 = None
            # y = None

        # xcont2 = torch.FloatTensor(self.X_cont[idx, :, :])

        batch = (X_3D, X_2D, y1)  # , xcont2, y)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


# COMPETITION METRIC FROM Konstantin Yakovlev
# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
def amex_metric(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)
    #print("G: {:.6f}, D: {:.6f}, ALL: {:6f}".format(gini[1]/gini[0], top_four, 0.5*(gini[1]/gini[0] + top_four)))
    return 0.5 * (gini[1]/gini[0] + top_four)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


####### LSTM model for 3D input ############################
class LSTM_AMEX(nn.Module):

    def __init__(self, input_size, ffnn_input, hidden_size,
                 num_layers, seq_length, activation=nn.GELU(), device=device):
        super(LSTM_AMEX, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.device = device

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        inp_dim = hidden_size

        self.ffnn = nn.Sequential(nn.Linear(ffnn_input, 512),
                                  nn.Dropout(0.20),
                                  activation,
                                  nn.Linear(512, 512),
                                  nn.Dropout(0.10),
                                  activation,
                                  nn.Linear(512, 256)
                                  )

        self.classifier = nn.Sequential(nn.Linear(inp_dim + 256, 512),
                                        nn.Dropout(0.20),
                                        activation,
                                        nn.Linear(512, 256),
                                        # nn.Dropout(0.10),
                                        activation,
                                        nn.Linear(256, 1)
                                        )

        self.attention1 = nn.Sequential(
            nn.Linear(inp_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, inp3D, inp2D):  # , cont_x2):

        inp3D = inp3D.to(self.device)
        inp2D = inp2D.to(self.device)

        h_0 = Variable(torch.zeros(
            self.num_layers, inp3D.size(0), self.hidden_size)).to(device)

        c_0 = Variable(torch.zeros(
            self.num_layers, inp3D.size(0), self.hidden_size)).to(device)

        # Propagate input through LSTM
        x, _ = self.lstm(inp3D, (h_0, c_0))
        weights1 = self.attention1(x)
        x = torch.sum(weights1 * x, dim=1)

        x2 = self.ffnn(inp2D)
        x = torch.cat([x, x2], dim=1)

        out = self.classifier(x)

        return out


####### Custom Model ############################

# Fully connected neural network with one hidden layer
class MLP_MODEL(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers, num_output, activation, attention=False):
        super(MLP_MODEL, self).__init__()
        self.meta_model = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.meta_model.append(nn.Linear(input_size, hidden_size))
                # self.meta_model.append(nn.BatchNorm1d(hidden_size))
                self.meta_model.append(nn.Dropout(0.20))
                self.meta_model.append(activation)
            else:
                self.meta_model.append(nn.Linear(hidden_size, hidden_size))
                # self.meta_model.append(nn.BatchNorm1d(hidden_size))
                self.meta_model.append(nn.Dropout(0.20))
                self.meta_model.append(activation)

        self.linear = nn.Linear(hidden_size, num_output)

        self.attention = attention
        if self.attention == True:
            self.att = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                activation,
                nn.Linear(hidden_size, input_size),
                nn.Sigmoid()
            )

    def forward(self, x):

        if self.attention == True:
            w = self.att(x)
            x = w * x
        else:
            x = x

        global out
        for i in range(len(self.meta_model)):
            if i == 0:
                out = self.meta_model[0](x)
            else:
                out = self.meta_model[i](out)

        out = self.linear(out)

        return out


class AMEX_Model(nn.Module):

    def __init__(self, input_size, ffnn_input, hidden_size,
                 num_layers, seq_length, activation, device=device):

        super(AMEX_Model, self).__init__()

        self.device = device

        self.encoders = torch.nn.ModuleList()
        for i in range(seq_length):
            self.encoders.append(MLP_MODEL(input_size=input_size,
                                           hidden_size=hidden_size,
                                           num_layers=num_layers,
                                           num_output=64,
                                           activation=activation))

        self.ffnn = nn.Sequential(nn.Linear(ffnn_input, 512),
                                  nn.Dropout(0.20),
                                  activation,
                                  nn.Linear(512, 512),
                                  nn.Dropout(0.10),
                                  activation,
                                  nn.Linear(512, 256)
                                  )

        self.classifier = nn.Sequential(nn.Linear(64 * seq_length + 256 + input_size, 512),
                                        nn.Dropout(0.20),
                                        activation,
                                        nn.Linear(512, 256),
                                        # nn.Dropout(0.10),
                                        activation,
                                        nn.Linear(256, 1)
                                        )

    def forward(self, inp3D, inp2D):

        inp3D = inp3D.to(self.device)
        inp2D = inp2D.to(self.device)

        encoded_input = []
        for i in range(inp3D.shape[1]):
            encoded_input.append(self.encoders[i](inp3D[:, i, :]))

        encoded_input = torch.cat(encoded_input, dim=1)

        x = self.ffnn(inp2D)
        x = torch.cat([encoded_input, x, inp3D[:, -1, :]], dim=1)

        out = self.classifier(x)

        return out



### 1D CNN model ####
class CNN_AMEX(nn.Module):

    def __init__(self, input_size, ffnn_input, hidden_size, num_layers,
                 seq_length, activation, device=device):
        super().__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.device = device

        def _norm(layer, dim=None):
            return nn.utils.weight_norm(layer, dim=dim)

        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(seq_length),
            # nn.Dropout(0.10),
            nn.Conv1d(seq_length, 8, kernel_size=1, stride=1, bias=True),
            activation,
            # nn.AdaptiveAvgPool1d(output_size=128),
            nn.BatchNorm1d(8),
            # nn.Dropout(0.10),
            nn.Conv1d(8, 4, kernel_size=1, stride=1, bias=True),
            activation,
            # nn.AdaptiveAvgPool1d(output_size=64)
        )

        self.flt = nn.Flatten()

        self.ffnn = nn.Sequential(nn.Linear(ffnn_input, 512),
                                  nn.Dropout(0.20),
                                  activation,
                                  nn.Linear(512, 512),
                                  nn.Dropout(0.10),
                                  activation,
                                  nn.Linear(512, 256)
                                  )

        self.classifier = nn.Sequential(nn.Linear(5 * input_size + 256 + ffnn_input, 512),
                                        # nn.BatchNorm1d(512),
                                        nn.Dropout(0.30),
                                        activation,
                                        nn.Linear(512, 256),
                                        # nn.BatchNorm1d(256),
                                        # nn.Dropout(0.20),
                                        activation,
                                        nn.Linear(256, 1)
                                        )

    def forward(self, inp3D, inp2D):
        inp3D = inp3D.to(self.device)
        inp2D = inp2D.to(self.device)

        x1 = self.conv1(inp3D)
        x1 = self.flt(x1)

        x2 = self.ffnn(inp2D)
        x = torch.cat([x1, x2, inp3D[:, -1, :], inp2D], dim=1)

        out = self.classifier(x)

        return out

PATH_TO_DATA='neural_network/data/'


def training_nn(Nfolds, MODEL_TYPE, lag, seq_len, num_epoch=50, patience=10,
                num_layers=3, activation=nn.GELU(), MODEL_ROOT='models/lstm/',
                hidden_dim=512):
    uniques = {}

    if not os.path.exists(MODEL_ROOT):
        os.makedirs(MODEL_ROOT)

    scores = []
    oof = {'customer_ID': [], 'target': [], 'oof': []}

    for fold in range(Nfolds):

        model_path = MODEL_ROOT + f'/modeL_{fold}.pth'
        folds = [0, 1, 2, 3, 4]

        valid_idx = [fold]
        train_idx = [x for x in folds if x not in valid_idx]

        # READ TRAIN DATA FROM DISK
        X_train_3D = []
        X_train_2D = []
        y_train = []
        for k in train_idx:
            X_train_3D.append(np.load(f'{PATH_TO_DATA}train_num_{k}.npy'))
            X_train_2D.append(np.concatenate((np.load(f'{PATH_TO_DATA}train_high_{k}.npy'),
                                              np.load(f'{PATH_TO_DATA}train_skew_{k}.npy'),
                                              # np.load(f'{PATH_TO_DATA}train_meduim_{k}.npy')
                                              ), axis=1))
            y_train.append(pd.read_pickle(f'{PATH_TO_DATA}targets_{k}.pkl'))
        X_train_3D = np.concatenate(X_train_3D, axis=0)
        X_train_2D = np.concatenate(X_train_2D, axis=0)
        y_train = pd.concat(y_train).target.values

        # print('### Training data shapes', X_train_3D.shape, X_train_2D.shape, y_train.shape)

        # READ VALID DATA FROM DISK
        X_valid_3D = []
        X_valid_2D = []
        y_valid = []
        for k in valid_idx:
            X_valid_3D.append(np.load(f'{PATH_TO_DATA}train_num_{k}.npy'))
            X_valid_2D.append(np.concatenate((np.load(f'{PATH_TO_DATA}train_high_{k}.npy'),
                                              np.load(f'{PATH_TO_DATA}train_skew_{k}.npy'),
                                              # np.load(f'{PATH_TO_DATA}train_meduim_{k}.npy')
                                              ), axis=1))
            y_valid.append(pd.read_pickle(f'{PATH_TO_DATA}targets_{k}.pkl'))
        X_valid_3D = np.concatenate(X_valid_3D, axis=0)
        X_valid_2D = np.concatenate(X_valid_2D, axis=0)

        oof['customer_ID'] = oof['customer_ID'] + pd.concat(y_valid).customer_ID.to_list()
        oof['target'] = oof['target'] + pd.concat(y_valid).target.to_list()

        y_valid = pd.concat(y_valid).target.values
        # print('### Validation data shapes', X_valid_3D.shape, X_valid_2D.shape, y_valid.shape)

        train_loader = AMEXLoader(X_train_3D, X_train_2D, y_train, lag, batch_size=512, shuffle=True)
        val_loader = AMEXLoader(X_valid_3D, X_valid_2D, y_valid, lag, batch_size=2048, shuffle=False)

        del X_train_3D, X_train_2D, y_train
        gc.collect()

        if MODEL_TYPE == 'CNN':
            model = CNN_AMEX(input_size=X_valid_3D.shape[2], ffnn_input=X_valid_2D.shape[1],
                             hidden_size=hidden_dim, num_layers=num_layers, seq_length=seq_len,
                             activation=activation).to(device)

        elif MODEL_TYPE == 'LSTM':
            model = LSTM_AMEX(input_size=X_valid_3D.shape[2], ffnn_input=X_valid_2D.shape[1],
                              hidden_size=hidden_dim, num_layers=num_layers, seq_length=seq_len,
                              activation=activation).to(device)
        else:
            model = AMEX_Model(input_size=X_valid_3D.shape[2], ffnn_input=X_valid_2D.shape[1],
                               hidden_size=256, num_layers=num_layers, seq_length=seq_len,
                               activation=activation).to(device)

        criterion = nn.BCEWithLogitsLoss()

        torch.manual_seed(42)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                                  max_lr=2e-3, epochs=num_epoch, steps_per_epoch=len(train_loader))
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=2, verbose=True, factor=0.1)

        best_score = np.inf
        best_amex = 0
        best_y_pred = None
        best_model = None
        counter = 0
        for ep in range(num_epoch):

            train_loss, val_loss = 0, 0

            model.train()
            for X_3D, X_2D, y in (train_loader):
                optimizer.zero_grad()

                out = model(X_3D, X_2D)
                loss = criterion(out[:, 0], y.to(device))  # +criterion(sim, y1.to(device))

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    train_loss += loss.item() / len(train_loader)

            # Validation phase
            phase = 'Val'
            with torch.no_grad():
                model.eval()

                #                 y_disc_true = []
                #                 y_disc_pred = []

                y_true = []
                y_pred = []
                rloss = 0

                for X_3D, X_2D, y in (val_loader):
                    out = model(X_3D, X_2D)

                    loss = criterion(out[:, 0], y.to(device))  # +criterion(sim, y1.to(device))
                    rloss += loss.item() / len(val_loader)

                    #                     y_disc_pred += list(sim.sigmoid().detach().cpu().numpy().flatten())
                    #                     y_disc_true += list(y1.cpu().numpy())

                    y_pred += list(out.sigmoid().detach().cpu().numpy().flatten())
                    y_true += list(y.cpu().numpy())

                #                 y_disc_pred = np.round(y_disc_pred)
                #                 score_sim = accuracy_score(y_disc_true, y_disc_pred)

                score = amex_metric(y_true, y_pred)
                if best_amex < score:
                    best_score = rloss
                    best_amex = score
                    best_y_pred = y_pred
                    best_model = model
                    torch.save(best_model, model_path)
                    counter = 0
                else:
                    counter = counter + 1

                print(
                    f"Fold-{fold} epoch: {ep} | Tain loss: {train_loss:.4f} | Val Loss: {rloss:.4f} | AMEX: {score:.4f} | Best AMEX: {best_amex:.4f}")

                # plt.plot(y_true)
                # plt.plot(y_pred)
                # plt.show()
                # scheduler.step(rloss)

            if counter >= patience:
                print("Early stopping")
                break

        print(f'The best score - {fold}:', np.round(best_amex, 6))
        scores.append(best_score)
        oof['oof'] = oof['oof'] + best_y_pred
        del train_loader, val_loader, X_valid_3D, X_valid_2D, y_valid, model, best_model
        gc.collect()

    return scores, oof

