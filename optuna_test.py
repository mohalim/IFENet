import os
import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import optuna
# from tqdm.keras import TqdmCallback
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print(tf.__version__)

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

df = df.sample(n=10000, random_state=0)

print(df.shape)

from sklearn.model_selection import train_test_split

y = df[target].values.astype(np.float32)
X = df.drop(columns=[target]).values

# This split is according to Tab Survey (Borisov et al., 2022)
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
X_train, X_vald, y_train, y_vald = train_test_split(X_tmp, y_tmp, train_size=0.9, random_state=0)

print(f'Training set: {X_train.shape}')
print(f'Validation set: {X_vald.shape}')
print(f'Test set: {X_test.shape}')

print(np.unique(y_train, return_counts=True))
print(np.unique(y_vald, return_counts=True))
print(np.unique(y_test, return_counts=True))

def array_to_dataset(data, target, shuffle=True, batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices((data, target))
    if shuffle:
        ds = ds.shuffle(batch_size*2).batch(batch_size).prefetch(batch_size)
    else:
        ds = ds.batch(batch_size)
    return ds

batch_size = 128
train_ds = array_to_dataset(X_train, y_train, batch_size=batch_size)
vald_ds = array_to_dataset(X_vald, y_vald, shuffle=False, batch_size=batch_size)
test_ds = array_to_dataset(X_test, y_test, shuffle=False, batch_size=batch_size)

n_features = X_train.shape[1]
_, counts = np.unique(y_train, return_counts=True)
n_classes = len(counts)

def build_model(batch_size=None, hidden_units=None, drop_rate=None, input_shape=None, output_units=None):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(batch_size, input_shape=(input_shape,), activation='relu'))
    model.add(tf.keras.layers.Dense(hidden_units, activation='relu'))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(output_units, activation='softmax'))
    return model

'''
model = build_model(batch_size=batch_size, hidden_units=64, drop_rate=0.3, input_shape=n_features, output_units=n_classes)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])
model.fit(train_ds, validation_data=vald_ds, epochs=10)
'''


# utility function to create model trials
def create_model(trial):
    # We optimize the numbers of layers, their units and learning rates 
    hidden_units = trial.suggest_int("hidden_size", 64,68)
    drop_rate = trial.suggest_float("drop_rate", 0.3, 0.4)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.002)

    model = build_model(batch_size=batch_size, hidden_units=hidden_units, drop_rate=drop_rate, input_shape=n_features, output_units=n_classes)
    
    # compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    
    return model

# Objective function
def objective(trial):
    # instantiate model
    model_opt = create_model(trial)
    
    # fit the model
    model_opt.fit(train_ds, validation_data=vald_ds, epochs=epochs, verbose=0)
    # calculate accuracy score
    acc_score = model_opt.evaluate(test_ds, verbose=0)[1]
    return acc_score

epochs = 10
n_trials = 10
# perform the optimization
print("Starting Optuna study...")
study = optuna.create_study(direction="maximize", study_name="baseline model optimization")
print("Study created. Starting optimization...")
study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)
print("Optimization completed!")
