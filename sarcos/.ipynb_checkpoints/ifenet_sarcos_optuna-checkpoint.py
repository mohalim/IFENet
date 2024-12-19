import os
import sys, getopt
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import optuna
from functools import partial

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ife import IFENetRegressor



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

checkpoint_path = 'checkpoints/weights.hdf5'
saved_model_path = 'saved_model/' 

def array_to_dataset(data, target, shuffle=True, batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices((data, target))
    if shuffle:
        ds = ds.shuffle(batch_size*2).batch(batch_size).prefetch(batch_size)
    else:
        ds = ds.batch(batch_size)
    return ds
     
def load_dataset(train_size=None, batch_size=512, is_target_norm=False, random_state=0):
    filepath_train = 'sarcos_inv.csv'
    filepath_test = 'sarcos_inv_test.csv'

    features = [c for c in range(0,21)]
    targets = [c for c in range(21,28)]
    
    # read training set
    df_train = pd.read_csv(filepath_train)
    #df_train = df_train.iloc[:,columns]
    
    # read test set
    df_test = pd.read_csv(filepath_test)
    #df_test = df_test.iloc[:,columns]

    y_train = df_train.iloc[:,targets].values
    X_train = df_train.iloc[:,features].values
    y_test = df_test.iloc[:,targets].values
    X_test = df_test.iloc[:,features].values
    
    X_train, X_vald, y_train, y_vald = train_test_split(X_train, y_train, test_size=4500, random_state=random_state)

    print(f'random_state: {random_state}')
    print(f'X_train: {X_train.shape}')
    print(f'X_vald: {X_vald.shape}')
    print(f'X_test: {X_test.shape}')
    print(f'y_train.shape: {y_train.shape}')
    print(f'y_vald.shape: {y_vald.shape}')
    print(f'y_test.shape: {y_test.shape}')

    if is_target_norm:
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train)
        y_vald = scaler.transform(y_vald)
        y_test = scaler.transform(y_test)

    # C = len([targets])
    
    print(f'batch_size: {batch_size}')
    batch_size = batch_size
    train_ds = array_to_dataset(X_train, y_train, batch_size=batch_size)
    vald_ds = array_to_dataset(X_vald, y_vald, shuffle=False, batch_size=batch_size)
    test_ds = array_to_dataset(X_test, y_test, shuffle=False, batch_size=batch_size)
    return train_ds, vald_ds, test_ds, X_train.shape[1], y_train.shape[1], list(df_train.columns)

def create_model(n_features, n_response, **kwargs):
    num_att = kwargs['num_att']
    r = kwargs['r']
    ife_num_layers = kwargs['ife_num_layers']
    clf_num_layers = kwargs['clf_num_layers']
    clf_hidden_units = kwargs['clf_hidden_units']
    reduction_layer = kwargs['reduction_layer']
    
    print(f'n_outputs: {n_response}')
    print(f'n_features: {n_features}')
    print(f'num_att: {num_att}')
    print(f'r: {r}')
    print(f'ife_num_layers: {ife_num_layers}')
    print(f'clf_num_layers: {clf_num_layers}')
    print(f'clf_hidden_units: {clf_hidden_units}')
    print(f'reduction_layer: {reduction_layer}')
    
    ife_params = {'n_features': n_features,
                  'n_outputs': n_response,
                  'num_att': num_att,
                  'r': r,
                  'ife_num_layers': ife_num_layers, 
                  'clf_num_layers': clf_num_layers,
                  'clf_hidden_units': clf_hidden_units,
                  'reduction_layer': reduction_layer
                 }
    model = IFENetRegressor(target_activation='linear', **ife_params)
    #model.summary()
    return model

def compile_model(model, lr=0.01, is_lr_scheduler=True):
    print(f'lr: {lr}')
    loss_fn = tf.keras.losses.MeanSquaredError()
    #lr = lr
    if is_lr_scheduler:
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=10000, decay_rate=0.95, staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])
    return model

def train_model(model, train_ds, vald_ds, epochs=100, patience=20):
    print(f'epochs: {epochs}')
    print(f'patience: {patience}')

    log_path = './logs'
    patience = patience
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss'),
                 tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True, monitor='val_loss', verbose=1),
                 tf.keras.callbacks.TensorBoard(log_dir='./logs')]

    epochs = epochs
    model.fit(train_ds, validation_data=vald_ds, epochs=epochs, callbacks=callbacks, verbose=2)
    model.load_weights(checkpoint_path)
    model.save_weights(saved_model_path)                       

def evaluate_model(model, test_ds, columns):
    model.load_weights(saved_model_path)
    
    y_pred = np.empty((0,))
    y_test = np.empty((0,))

    for data,label in test_ds:
        y_hat = model(data)
        y_pred = np.append(y_pred, y_hat.numpy().ravel())

        label = label.numpy()
        y_test = np.append(y_test, label.ravel())

    print(f'R2 Score: {r2_score(y_test, y_pred)}')
    print(f'MSE: {mean_squared_error(y_test, y_pred)}')
    print(f'MAE: {mean_absolute_error(y_test, y_pred)}')

    feat_scores = model.input_scores
    for col,score in zip(columns,feat_scores):
        print(f'{col}: {score}')

def save_model(model, path_file): # to save the model's weights in h5 format
    model.save_weights(path_file)

# utility function to create and optimize model trials
def create_model_for_optimization(trial, n_features, n_response):
    # We optimize the parameters
    num_att = trial.suggest_int('num_att', 16, 150)
    r = trial.suggest_float('r', 1.5, 6.0)
    clf_num_layers = trial.suggest_int('clf_num_layers', 1, 2)
    clf_hidden_units1 = trial.suggest_int('clf_hidden_units1', 32, 150)
    clf_hidden_units2 = trial.suggest_int('clf_hidden_units2', 10, 85)
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.02)
    reduction_layer_id = trial.suggest_int('reduction_layer_id', 0, 1)

    if reduction_layer_id:
        reduction_layer = 'flatten'
    else:
        reduction_layer = 'average'
        
    if clf_num_layers == 1:
        clf_hidden_units = [clf_hidden_units1]
    else:
        clf_hidden_units = [clf_hidden_units1, clf_hidden_units2]
    ife_num_layers = 1
    ife_params = {'n_features': n_features,
                  'n_outputs': n_response,
                  'num_att': num_att,
                  'r': r,
                  'ife_num_layers': ife_num_layers, 
                  'clf_num_layers': clf_num_layers,
                  'clf_hidden_units': clf_hidden_units,
                  'reduction_layer': reduction_layer}

    model = IFENetRegressor(target_activation='linear', **ife_params)
    
    # compile the model
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=0.95, staircase=True)
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler))
    
    return model

def objective(trial, train_ds, vald_ds, test_ds, epochs, n_features, n_response):
    # instantiate model
    model_opt = create_model_for_optimization(trial, n_features, n_response)
    model_opt.fit(train_ds, validation_data=vald_ds, epochs=epochs, verbose=0)
    # calculate MSE score
    mse_score = model_opt.evaluate(test_ds, verbose=0)
    return mse_score

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help="Mode of execution: train, evaluate or optimize")
    parser.add_argument('-d', '--data_params', nargs='*', help="Dataset parameters: batch_size train_set_percentage is_target_normalized random_state : 512 0.7 0 0")
    parser.add_argument('-t', '--train_params', nargs='*', help="Training parameters: learning_rate is_lr_scheduler epochs patience (e.g. -t 0.01 1 100 20)")
    parser.add_argument('-cl', '--clf_num_layers', type=int, help="Classifier/Regressor parameters: num_hidden_layers (e.g. -cl 2)")
    parser.add_argument('-cu', '--clf_hidden_units', nargs='+', type=int, help="Classifier/Regressor hidden units: first_hidden_units second_hidden_units (e.g. -cu 128 64")
    parser.add_argument('-f', '--ife_params', nargs='*', help="IFENet parameters: num_attentions r num_ife_layers reduction_layer (e.g. -f 8 5.6498 1 flatten)")
    parser.add_argument('-s', '--save_filename', type=str, help="Filename for saving the model's weights (e.g. -s ifeNet_cover)")
    parser.add_argument('-o', '--optimize_n_trials', help="Optimization number of trials: 50")
    
    args = parser.parse_args()

    if args.data_params:
        batch_size = int(args.data_params[0])
        train_size = float(args.data_params[1])
        is_target_norm = int(args.data_params[2])
        if is_target_norm == 1:
            is_target_norm = True
        else:
            is_target_norm = False
        random_state = int(args.data_params[3])
    else:
        batch_size = 512
        train_size = 0.7
        random_state = 0
        is_target_norm = False
    
    if args.train_params:
        train_params = args.train_params
        lr = float(train_params[0])
        is_lr_scheduler = bool(train_params[1])
        epochs = int(train_params[2])
        patience = int(train_params[3])
    else:
        lr = 0.015
        is_lr_scheduler = True
        epochs = 100
        patience = 20

    if args.clf_num_layers:
        clf_num_layers = args.clf_num_layers
    else:
        clf_num_layers = 1

    if args.clf_hidden_units:
        clf_hidden_units = args.clf_hidden_units
    else:
        clf_hidden_units = [128]
    
    if args.ife_params:
        ife_params = args.ife_params
        num_att = int(ife_params[0])
        r = float(ife_params[1])
        ife_num_layers = int(ife_params[2])
        reduction_layer = ife_params[3]
    else:
        num_att = 8
        r = 5.6498
        ife_num_layers = 1
        reduction_layer = 'flatten'

    filename_params = '_'+str(clf_hidden_units)+'_'+str(num_att)+'_'+str(r)
    if args.save_filename:
        filename = args.save_filename
        path_filename = saved_model_path + filename + filename_params + '.h5'
    else:
        path_filename = saved_model_path + 'ifeNet_sarcos' + filename_params + '.h5'

    if args.optimize_n_trials:
        n_trials = int(args.optimize_n_trials)
    else:
        n_trials = 50

    model_params = {'num_att': num_att,
                    'r': r,
                    'ife_num_layers': ife_num_layers,
                    'clf_num_layers': clf_num_layers,
                    'clf_hidden_units': clf_hidden_units,
                    'reduction_layer': reduction_layer}
    
    train_ds, vald_ds, test_ds, n_features, n_response, columns = load_dataset(train_size=train_size, batch_size=batch_size, is_target_norm=is_target_norm, random_state=random_state)
    model = create_model(n_features, n_response, **model_params)
    model = compile_model(model, lr=lr, is_lr_scheduler=is_lr_scheduler)
    
    if args.mode == 'train':    
        train_model(model, train_ds, vald_ds, epochs=epochs, patience=patience)
        evaluate_model(model, test_ds, columns)
        save_model(model, path_filename)
    elif args.mode == 'evaluate':
        evaluate_model(model, test_ds, columns)
    elif args.mode == 'optimize':
        # perform the optimization
        partial_objective = partial(objective, train_ds=train_ds, vald_ds=vald_ds, test_ds=test_ds, epochs=epochs, n_features=n_features, n_response=n_response)
        print("Starting Optuna study...")
        study = optuna.create_study(direction="minimize", study_name="model optimization")
        print("Study created. Starting optimization...")
        study.optimize(partial_objective, n_trials=n_trials, n_jobs=-1)
        print("Optimization completed!")
        
    
if __name__ == "__main__":
    print(tf.__version__)
    main(sys.argv[1:])