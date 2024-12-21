import sys
sys.path.append("../tf_ifenet")

import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Import the library
from models import IFENetClassifier
from config import DataConfig, ModelConfig
from utility import dataframe_to_dataset

print(tf.__version__)

 # load the csv file into a DataFrame using pandas
filepath = '../datasets/heloc/heloc.data.csv'
df = pd.read_csv(filepath)

# target columns should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
target_columns = ['RiskPerformance']

# list of numerical column names
num_col_names = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen',
                 'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 
                 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq',
                 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades',
                 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 
                 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 
                 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance',
                 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']

# list of categorical column names (the dataset has no categorical columns)
cat_col_names = []

# split the dataset into training, validation and test sets.
# the dataset is in a single pandas DataFrame
train_size = 8359
tmp, test = train_test_split(df, train_size=train_size, random_state=0)
train, vald = train_test_split(tmp, train_size=7000, random_state=0)

# convert the training, validation and test sets DataFrame into a tf.data.Dataset
# use the utility function (dataframe_to_dataset) to convert the DataFrame
batch_size = 256
train_ds = dataframe_to_dataset(train, target_columns, shuffle=True, batch_size=batch_size)
vald_ds = dataframe_to_dataset(vald, target_columns, shuffle=False, batch_size=batch_size)
test_ds = dataframe_to_dataset(test, target_columns, shuffle=False, batch_size=batch_size)

# define configs using DataConfig and ModelConfig
# DataConfig is used to define data related configurations like the categorical columns, numerical columns, encoding output mode for categorical columns ('one_hot' or 'multi_hot') and normalization (True or False) for numerical columns.
data_config = DataConfig(categorical_column_names=cat_col_names, 
                         numerical_column_names=num_col_names,
                         category_output_mode='one_hot',
                         is_normalization=False)

# ModelConfig is used to define model related configurations. IFENet has five parameters:
# 1. number of attention heads (num_att): integer
# 2. scaling factor (r): float
# 3. number of hidden layers in preditive module (clf_num_layers): integer
# 4. number of units for each hidden layer (clf_hidden_units): list of integers
# 5. reduction layer type (reduction_layer): "flatten" or "average"
model_config = ModelConfig(num_att=16,
                           r=3.5,
                           clf_num_layers=1,
                           clf_hidden_units=[32],
                           reduction_layer='flatten')

# create the model by passing the configs
model = IFENetClassifier(data_config, model_config)

# build the layers by passing the training set tf.data.Dataset
model.build_model(train_ds)

# define the loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

# we can use callbacks to implement to save the model during training (modelcheckpoint) and early stopping (optional)
checkpoint_path = 'checkpoints/ifeNet_heloc.h5'
patience = 20
callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_accuracy'),
             tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy')]

# train and evaluate the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
model.fit(train_ds, validation_data=vald_ds, epochs=100, callbacks=callbacks)
model.evaluate(test_ds)

# perform a prediction on a batch of data. The model will compute the feature importance scores of the data
[(data_batch, label_batch)] = test_ds.take(1)
y_pred = model(data_batch)

# get feature scores and save it into a DataFrame
df_feat_scores = model.get_feature_importance()

# save the model
saved_model_path = 'saved_models/ifeNet_heloc.keras'
model.save(saved_model_path)

# load the model (for later use)
new_model = tf.keras.models.load_model(saved_model_path, safe_mode=False)
