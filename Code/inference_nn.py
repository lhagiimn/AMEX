from nn_model_utils import *
import joblib
import pickle

PATH_TO_DATA = 'neural_network/test/'

def inference(Nfolds, MODEL_TYPE, lag, seq_len, num_epoch=50, patience=10,
              num_layers=3, activation=nn.SiLU(), MODEL_ROOT='models/lstm/',
              hidden_dim=512):
    y_pred = []
    cids = []
    for k in range(2):

        # READ TEST DATA FROM DISK
        X_test_3D = []
        X_test_2D = []
        customers = []
        X_test_3D.append(np.load(f'{PATH_TO_DATA}train_num_{k}.npy'))
        X_test_2D.append(np.concatenate((np.load(f'{PATH_TO_DATA}train_high_{k}.npy'),
                                         np.load(f'{PATH_TO_DATA}train_skew_{k}.npy'),
                                         # np.load(f'{PATH_TO_DATA}test_meduim_{k}.npy')
                                         ), axis=1))
        customers.append(pd.read_pickle(f'{PATH_TO_DATA}targets_{k}.pkl'))

        X_test_3D = np.concatenate(X_test_3D, axis=0)
        X_test_2D = np.concatenate(X_test_2D, axis=0)
        cids = cids + list(pd.concat(customers).customer_ID.values)

        test_loader = AMEXLoader(X_test_3D, X_test_2D, None, lag, batch_size=8196 * 4, shuffle=False)

        models = []
        for fold in range(Nfolds):

            model_path = MODEL_ROOT + f'/modeL_{fold}.pth'

            if MODEL_TYPE == 'CNN':
                model = CNN_AMEX(input_size=X_test_3D.shape[2], ffnn_input=X_test_2D.shape[1],
                                 hidden_size=hidden_dim, num_layers=num_layers, seq_length=seq_len,
                                 activation=activation).to(device)

                model = torch.load(model_path)
                model.to(device)
                models.append(model)

            elif MODEL_TYPE == 'LSTM':
                model = LSTM_AMEX(input_size=X_test_3D.shape[2], ffnn_input=X_test_2D.shape[1],
                                  hidden_size=hidden_dim, num_layers=num_layers, seq_length=seq_len,
                                  activation=activation).to(device)
                model = torch.load(model_path)
                model.to(device)
                models.append(model)

            else:
                model = AMEX_Model(input_size=X_test_3D.shape[2], ffnn_input=X_test_2D.shape[1],
                                   hidden_size=256, num_layers=num_layers, seq_length=seq_len,
                                   activation=activation).to(device)
                model = torch.load(model_path)
                model.to(device)
                models.append(model)

        preds = np.zeros(X_test_2D.shape[0])
        for model in models:
            with torch.no_grad():
                model.eval()
                pred = []
                for X_3D, X_2D, y in tqdm(test_loader):
                    out = model(X_3D, X_2D)
                    pred += list(out.sigmoid().detach().cpu().numpy().flatten())
            preds += np.asarray(pred) / len(models)

        y_pred.append(preds)

    return cids, y_pred



lag = 0
for MODEL_TYPE in ['CNN']:
    cids, y_pred = inference(Nfolds=5, MODEL_TYPE=MODEL_TYPE, lag=lag,
                             seq_len=13 - lag, num_epoch=20, patience=5,
                             num_layers=2, activation=nn.CELU(),
                             MODEL_ROOT=f'neural_network/models/{MODEL_TYPE}/',
                             hidden_dim=512)

    pred_nn = pd.DataFrame()
    pred_nn['customer_ID'] = cids
    pred_nn['prediction'] = np.concatenate(y_pred)
    joblib.dump(pred_nn, f'figures/preds_{MODEL_TYPE}.pkl')

    pred_nn = pred_nn.set_index('customer_ID')

    # WRITE SUBMISSION FILE
    with open('preprocessed_data/encoder.pkl', 'rb') as fin:
        encoder = pickle.load(fin)

    sub = pd.read_csv('preprocessed_data/sample_submission.csv')[['customer_ID']]
    sub['customer_ID_hash'] = encoder.transform(sub['customer_ID'])
    sub = sub.set_index('customer_ID_hash')
    sub['prediction'] = pred_nn.loc[sub.index, 'prediction']
    sub = sub.reset_index(drop=True)

    # DISPLAY PREDICTIONS
    sub.to_csv('submission_nn.csv', index=False)
    print('Submission file shape is', sub.shape)
    print(sub.isnull().sum())
    print(sub.head())
