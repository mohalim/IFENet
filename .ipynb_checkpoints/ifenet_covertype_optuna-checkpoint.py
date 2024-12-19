import os
import sys, getopt
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import optuna
from functools import partial
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from IterativeFeatureExclusion import IFENetClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

checkpoint_path = 'checkpoints/'
saved_model_path = 'saved_model/' 

def array_to_dataset(data, target, shuffle=True, batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices((data, target))
    if shuffle:
        ds = ds.shuffle(batch_size*2).batch(batch_size).prefetch(batch_size)
    else:
        ds = ds.batch(batch_size)
    return ds
     
def load_dataset(train_size, batch_size=512, random_state=0):
    filepath = 'covtype.data.csv'
    df = pd.read_csv(filepath, header=None)

    columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 
               'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
               'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
               'Soil_Type1' ,'Soil_Type2' ,'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 
               'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 
               'Soil_Type11' ,'Soil_Type12' ,'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 
               'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 
               'Soil_Type21' ,'Soil_Type22' ,'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 
               'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 
               'Soil_Type31' ,'Soil_Type32' ,'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 
               'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type']
    target = 'Cover_Type'

    df.columns = columns
    df[target] = df[target] - 1 # recode the integer values
    df = df.astype(float) # convert all to float

    y = df[target].values.astype(np.float32)
    X = df.drop(columns=[target]).values
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
    X_train, X_vald, y_train, y_vald = train_test_split(X_tmp, y_tmp, test_size=0.1, random_state=random_state)

    print(f'random_state: {random_state}')
    print(f'X_train: {X_train.shape}')
    print(f'X_vald: {X_vald.shape}')
    print(f'X_test: {X_test.shape}')

    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_vald, return_counts=True))
    print(np.unique(y_test, return_counts=True))

    num_target_instances = np.bincount(y_train.astype(np.int64))
    N = np.sum(num_target_instances)
    C = len(num_target_instances)

    print(f'batch_size: {batch_size}')
    batch_size = batch_size
    train_ds = array_to_dataset(X_train, y_train, batch_size=batch_size)
    vald_ds = array_to_dataset(X_vald, y_vald, shuffle=False, batch_size=batch_size)
    test_ds = array_to_dataset(X_test, y_test, shuffle=False, batch_size=batch_size)
    return train_ds, vald_ds, test_ds, X_train.shape[1], C, columns

def create_model(n_features, n_classes, **kwargs):
    num_att = kwargs['num_att']
    r = kwargs['r']
    ife_num_layers = kwargs['ife_num_layers']
    clf_num_layers = kwargs['clf_num_layers']
    clf_hidden_units = kwargs['clf_hidden_units']
    
    print(f'n_classes: {n_classes}')
    print(f'n_features: {n_features}')
    print(f'num_att: {num_att}')
    print(f'r: {r}')
    print(f'ife_num_layers: {ife_num_layers}')
    print(f'clf_num_layers: {clf_num_layers}')
    print(f'clf_hidden_units: {clf_hidden_units}')
    
    ife_params = {'n_features': n_features,
                  'n_classes': n_classes,
                  'num_att': num_att,
                  'r': r,
                  'ife_num_layers': ife_num_layers, 
                  'clf_num_layers': clf_num_layers,
                  'clf_hidden_units': clf_hidden_units
                 }
    model = IFENetClassifier(**ife_params)
    #model.summary()
    return model

def compile_model(model, lr=0.01, is_lr_scheduler=True):
    print(f'lr: {lr}')
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    lr = lr
    if is_lr_scheduler:
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=10000, decay_rate=0.95, staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model

def train_model(model, train_ds, vald_ds, epochs=100, patience=20):
    print(f'epochs: {epochs}')
    print(f'patience: {patience}')

    log_path = './logs'
    patience = patience
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss'),
                 tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy'),
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
        y_hat = np.argmax(y_hat, axis=-1)
        y_pred = np.append(y_pred, y_hat.ravel())

        label = label.numpy()
        y_test = np.append(y_test, label.ravel())

    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    feat_scores = model.input_scores
    for col,score in zip(columns,feat_scores):
        print(f'{col}: {score}')

def save_model(model, path_file): # to save the model's weights in h5 format
    model.save_weights(path_file)

# utility function to create and optimize model trials
def create_model_for_optimization(trial, n_features, n_classes):
    # We optimize the parameters
    num_att = trial.suggest_int('num_att', 8, 24)
    r = trial.suggest_float('r', 2.0, 7.0)
    clf_num_layers = trial.suggest_int('clf_num_layers', 1, 2)
    clf_hidden_units1 = trial.suggest_int('clf_hidden_units1', 32, 150)
    clf_hidden_units2 = trial.suggest_int('clf_hidden_units2', 10, 85)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 0.02)

    clf_hidden_units = [clf_hidden_units1, clf_hidden_units2]
    ife_num_layers = 1
    ife_params = {'n_features': n_features,
                  'n_classes': n_classes,
                  'num_att': num_att,
                  'r': r,
                  'ife_num_layers': ife_num_layers, 
                  'clf_num_layers': clf_num_layers,
                  'clf_hidden_units': clf_hidden_units}

    model = IFENetClassifier(**ife_params)
    
    # compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    
    return model

def objective(trial, train_ds, vald_ds, test_ds, epochs, n_features, n_classes):
    # instantiate model
    epochs = epochs
    model_opt = create_model_for_optimization(trial, n_features, n_classes)
    model_opt.fit(train_ds, validation_data=vald_ds, epochs=epochs, verbose=0)
    # calculate accuracy score
    acc_score = model_opt.evaluate(test_ds, verbose=0)[1]
    return acc_score

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help="Mode of execution: train, evaluate or optimize")
    parser.add_argument('-d', '--data_params', nargs='*', help="Dataset parameters: batch_size train_set_percentage random_state : 512 0.7 0")
    parser.add_argument('-t', '--train_params', nargs='*', help="Training parameters: learning_rate is_lr_scheduler epochs patience (e.g. -t 0.01 1 100 20)")
    parser.add_argument('-cl', '--clf_num_layers', type=int, help="Classifier/Regressor parameters: num_hidden_layers (e.g. -cl 2)")
    parser.add_argument('-cu', '--clf_hidden_units', nargs='+', type=int, help="Classifier/Regressor hidden units: first_hidden_units second_hidden_units (e.g. -cu 128 64")
    parser.add_argument('-f', '--ife_params', nargs='*', help="IFENet parameters: num_attentions r num_ife_layers (e.g. -f 8 5.6498 1)")
    parser.add_argument('-s', '--save_filename', type=str, help="Filename for saving the model's weights (e.g. -s ifeNet_cover)")
    parser.add_argument('-o', '--optimize_n_trials', help="Optimization number of trials: 50")
    
    args = parser.parse_args()

    if args.data_params:
        batch_size = int(args.data_params[0])
        train_size = float(args.data_params[1])
        random_state = int(args.data_params[2])
    else:
        batch_size = 512
        train_size = 0.7
        random_state = 0
    
    if args.train_params:
        train_params = args.train_params
        lr = float(train_params[0])
        is_lr_scheduler = bool(train_params[1])
        epochs = int(train_params[1])
        patience = int(train_params[2])
    else:
        # batch_size = 512
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
    else:
        num_att = 8
        r = 5.6498
        ife_num_layers = 1

    if args.save_filename:
        filename = args.save_filename
        path_filename = saved_model_path + filename + '.h5'
    else:
        path_filename = saved_model_path + 'ifeNet_cover.h5'

    if args.optimize_n_trials:
        n_trials = int(args.optimize_n_trials)
    else:
        n_trials = 50

    model_params = {'num_att': num_att,
                    'r': r,
                    'ife_num_layers': ife_num_layers,
                    'clf_num_layers': clf_num_layers,
                    'clf_hidden_units': clf_hidden_units}
    
    train_ds, vald_ds, test_ds, n_features, n_classes, columns = load_dataset(train_size=train_size, batch_size=batch_size, random_state=random_state)
    model = create_model(n_features, n_classes, **model_params)
    model = compile_model(model, lr=lr, is_lr_scheduler=is_lr_scheduler)
    
    if args.mode == 'train':    
        train_model(model, train_ds, vald_ds, epochs=epochs, patience=patience)
        evaluate_model(model, test_ds, columns)
        save_model(model, path_filename)
    elif args.mode == 'evaluate':
        evaluate_model(model, test_ds, columns)
    elif args.mode == 'optimize':
        # perform the optimization
        partial_objective = partial(objective, train_ds=train_ds, vald_ds=vald_ds, test_ds=test_ds, epochs=epochs, n_features=n_features, n_classes=n_classes)
        print("Starting Optuna study...")
        study = optuna.create_study(direction="maximize", study_name="model optimization")
        print("Study created. Starting optimization...")
        study.optimize(partial_objective, n_trials=n_trials, n_jobs=-1)
        print("Optimization completed!")
        
    
if __name__ == "__main__":
    print(tf.__version__)
    main(sys.argv[1:])