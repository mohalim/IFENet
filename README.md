# Iterative Feature Exclusion Ranking for Deep Tabular Learning

Tabular data is prevalent is various domains such as finance, healthcare and education. IFENet is a deep learning model designed to enhance feature importance ranking in tabular data. It introduces a novel iterative feature exclusion module that iteratively excludes each feature from the input data and computes attention scores, representing the impact of features on predictions. By aggregating these attention scores across iterations, IFENet generates a refined representation of feature importance that captures both global and local interactions between features. This repository provides the code and instructions for implementing and training IFENet using TensorFlow 2.

## Usage

```python
from models import IFENetClassifier
from config import DataConfig, ModelConfig
from utility import dataframe_to_dataset

# Convert the training set DataFrame to tf.data.Dataset
train_ds = dataframe_to_dataset(train, target_columns, shuffle=True, batch_size=256)

data_config = DataConfig(categorical_column_names=cat_col_names, 
                         numerical_column_names=num_col_names,
                         encode_category = 'embedding',
                         embedding_output_dim = 'auto',
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

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model. A list of callbacks such as EarlyStopping, ModelCheckpoint can be passed to the .fit() method.
model.fit(train_ds, validation_data=vald_ds, epochs=100)
```

## Required dependencies
IFENet requires the following dependencies
- NumPy >= 1.23.5
- Pandas >= 2.0.3
- TensorFlow >= 2.13.1

## Citation
Shaninah, F.S.E., Baraka, A.M.A., Noor, M.H.M., 2024. Iterative Feature Exclusion Ranking for Deep Tabular Learning. [https://doi.org/10.48550/arXiv.2412.16442](https://arxiv.org/abs/2412.16442)
```
@misc{shaninah2024iterativefeatureexclusionranking,
      title={Iterative Feature Exclusion Ranking for Deep Tabular Learning}, 
      author={Fathi Said Emhemed Shaninah and AbdulRahman M. A. Baraka and Mohd Halim Mohd Noor},
      year={2024},
      eprint={2412.16442},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.16442}, 
}
```

