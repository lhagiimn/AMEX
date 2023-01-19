from nn_model_utils import *
import gc
gc.enable()

import warnings
warnings.simplefilter("ignore")


lag=0
for MODEL_TYPE in ['CNN']:#, 'MLP', 'LSTM']:
    scores, oof = training_nn(Nfolds=5, MODEL_TYPE=MODEL_TYPE, lag=lag,
                              seq_len=13-lag, num_epoch=20, patience=5,
                              num_layers=3, activation = nn.CELU(),
                              MODEL_ROOT=f'neural_network/models/{MODEL_TYPE}/',
                              hidden_dim=512)

    print('Average score:', np.mean(scores))
    print('OOF score:', amex_metric(oof['target'], oof['oof']))

    oof = pd.DataFrame.from_dict(oof)
    oof.to_csv(f'neural_network/oof_{MODEL_TYPE}.csv', index=False)






