# Iterative Feature Exclusion Ranking for Deep Tabular Learning

IFENet is a deep learning model designed to enhance feature importance ranking in tabular data. It introduces a novel iterative feature exclusion module that iteratively excludes each feature from the input data and computes attention scores, representing the impact of features on predictions. By aggregating these attention scores across iterations, IFENet generates a refined representation of feature importance that captures both global and local interactions between features. This repository provides the code and instructions for implementing and training IFENet using TensorFlow 2.14.

## Usage

```
from models import IFENetClassifier
from config import DataConfig, ModelConfig
from utility import dataframe_to_dataset

# convert the training set DataFrame to tf.data.Dataset
train_ds = dataframe_to_dataset(train, target_columns, shuffle=True, batch_size=batch_size)

data_config = DataConfig(categorical_column_names=cat_col_names, 
                         numerical_column_names=num_col_names,
                         category_output_mode='one_hot',
                         is_normalization=False)

model_config = ModelConfig(num_att=16,
                           r=3.5,
                           clf_num_layers=1,
                           clf_hidden_units=[32],
                           reduction_layer='flatten')

model = IFENetClassifier(data_config, model_config)
# or
model = IFENetRegressor(data_config, model_config)

model.build_model(train_ds)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

model.fit(train_ds, validation_data=vald_ds, epochs=100, callbacks=callbacks)
```
