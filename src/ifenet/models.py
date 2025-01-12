"""
Created on Mon Nov 10 09:00:00 2024
@author: Mohd Halim Mohd Noor
"""

import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.saving import register_keras_serializable, serialize_keras_object, deserialize_keras_object
from .config import DataConfig, ModelConfig
from .layers import _Attention, _IterativeFeatureExclusion, _CategoricalEncodingLayer, _NumericalEncodingLayer


@register_keras_serializable(name="_ifeModule")
class _IFEModule(tf.keras.Model):
    def __init__(self, data_config, model_config, name="_ifeModule", **kwargs):
        super(_IFEModule, self).__init__()
        self._attn_norm_fn = None
        self.is_built = False

        self._data_config = data_config
        self._model_config = model_config

        self._categorical_column_names = self._data_config.categorical_column_names
        self._numerical_column_names = self._data_config.numerical_column_names
        self._encode_category = self._data_config.encode_category
        self._embedding_output_dim = self._data_config.embedding_output_dim
        self._category_output_mode = self._data_config.category_output_mode
        self._is_normalization = self._data_config.is_normalization
        
        self._num_att = self._model_config.num_att
        self._r = self._model_config.r

        self._n_features = 0
        self._encoder_layers = {}

        self.data_batch = None
        self.feature_indices = {}
        self.input_scores = None
    
    def _create_encoder_layers(self, dataset, feature_names, feature_dtypes):
        for name in feature_names:
            if name in self._categorical_column_names:
                # print(f'feature name: {name}')
                feature_ds = dataset.map(lambda x, y: x[name])
                layer = _CategoricalEncodingLayer(self._encode_category, self._embedding_output_dim, self._category_output_mode, 
                                                  feature_ds, feature_dtypes[name], name="_categoricalEncodingLayer_"+name)
                self._encoder_layers[name] = layer
            
            elif name in self._numerical_column_names:
                # print(f'feature name: {name}')
                feature_ds = dataset.map(lambda x, y: x[name])
                layer = _NumericalEncodingLayer(self._is_normalization, feature_ds, name="_numericalEncodingLayer_"+name)
                self._encoder_layers[name] = layer
                
        st = 0
        ed = 0
        n_features = 0
        for name, layer in self._encoder_layers.items():
            example_input = next(iter(dataset.map(lambda x, y: x[name]))).numpy()
            example_output = layer(example_input)
            feature_size = example_output.shape[-1]  # Store the size (last dimension)
            ed = st + feature_size
            n_features = ed 
            index = list([st, ed])
            st = ed
            self.feature_indices[name] = index

        return n_features

    def get_feature_importance(self):
        if not tf.is_symbolic_tensor(self.input_scores): # input(batch, n_features, 1)
            feature_scores = np.mean(self.input_scores, axis=0)
            score = 0
            feat_rank = {}
            for feature, indices in self.feature_indices.items():
                for j,i in enumerate(range(indices[0], indices[1])):
                    name = feature + '[' + str(j) + ']'
                    feat_rank[name] = feature_scores[i]
            
            df = pd.DataFrame(list(feat_rank.items()), columns=['Feature', 'Score'])
            return df.sort_values(by='Score', ascending=False)
        else:
            msg = "Please perform a prediction first to compute the feature importance scores."
            print("\033[91m {}\033[00m" .format(msg))

    def fit(self, train_ds, validation_data=None, epochs=1, batch_size=None, class_weight=None, sample_weight=None, verbose=2, callbacks=None):
        """
        Override fit to ensure model is built before training.
        """
        if not self.is_built:
            raise ValueError("Model has not been built. Please run `model.build_model(train_ds)` before calling `fit()`.")

        # Call the original fit() method (or perform custom training loop if needed)
        super(_IFEModule, self).fit(train_ds, 
                                    validation_data=validation_data, 
                                    epochs=epochs, 
                                    batch_size=batch_size, 
                                    verbose=verbose, 
                                    class_weight=class_weight,
                                    sample_weight=sample_weight,
                                    callbacks=callbacks)

    def get_config(self):
        base_config = super(_IFEModule, self).get_config()
        
        # Serialize the data_config and model_config
        data_config_dict = self._data_config.get_config()
        model_config_dict = self._model_config.get_config()
        
        # Return the complete configuration
        config = {
            **base_config,
            "data_config": data_config_dict,
            "model_config": model_config_dict,
        }
        
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize the DataConfig and ModelConfig
        data_config = DataConfig.from_config(config["data_config"])
        model_config = ModelConfig.from_config(config["model_config"])

        # Create an instance of _IFEModule
        instance = cls(data_config, model_config)
        
        # Return the reconstructed model
        return instance
    
@register_keras_serializable(name="ifeNetRegressor")
class IFENetRegressor(_IFEModule):
    def __init__(self, data_config, model_config, target_activation='linear', name="ifeNetRegressor", **kwargs):
        super(IFENetRegressor, self).__init__(data_config, model_config)

        self._attn_norm_fn = 'sigmoid'
        self.target_activation = target_activation
        self._model_config = model_config

        self._clf_num_layers = self._model_config.clf_num_layers
        self._clf_hidden_units = self._model_config.clf_hidden_units
        self._clf_dropout = self._model_config.clf_dropout

    def build_model(self, dataset):
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError(f"Input must be a tf.data.Dataset, got {type(dataset)}.")

        feature_dtypes = {key: spec.dtype for key, spec in dataset.element_spec[0].items()}
        feature_names = list(feature_dtypes.keys())
        
        self._n_features = self._create_encoder_layers(dataset, feature_names, feature_dtypes)

        self._preprocess = tf.keras.layers.BatchNormalization(name=f"{self.name}/preprocess_batch_norm")

        # Determine the number of responses
        targets = next(iter(dataset.map(lambda x,y: y))).numpy()
        n_outputs = targets.shape[1]

        self._ife_attn = _IterativeFeatureExclusion(self._n_features, n_outputs, self._attn_norm_fn, self._num_att, self._r)

        # Build the predictive layers
        clf_hidden_layers = []
        for l in range(0, self._clf_num_layers):
            clf_hidden_layers.append(tf.keras.layers.Dense(units=self._clf_hidden_units[l], activation='relu'))
            #clf_hidden_layers.append(tf.keras.layers.BatchNormalization())
            clf_hidden_layers.append(tf.keras.layers.Dropout(rate=self._clf_dropout))
        
        self.clf_hidden_layers = tf.keras.Sequential(clf_hidden_layers, name=f"{self.name}/fc_hidden_layers")
        self.fc_out = tf.keras.layers.Dense(units=n_outputs, activation=self.target_activation, name=f"{self.name}/fc_out")
        self.is_built = True
        
    def call(self, inputs, training=False): # (batch, n_features)
        # preprocessing the inputs
        features = [self._encoder_layers[name](inputs[name], training=training) for name in self._encoder_layers]
        features = tf.concat(features, axis=1)

        # features are the preprocessed inputs
        # batch_size = tf.shape(features)[0]
        x = self._preprocess(features, training=training) # (batch, n_features)
        
        self.input_scores = self._ife_attn(x, training=training)
        x = x * self.input_scores
        x = self.clf_hidden_layers(x, training=training)
        outputs = self.fc_out(x, training=training)
        
        return outputs

    def get_config(self):
        # Serialize configuration of parent class (_IFEModule)
        base_config = super(IFENetRegressor, self).get_config()

        # Serialize the layer configurations for the layers created in build_model
        preprocess_config = self._preprocess.get_config()
        ife_attn_config = self._ife_attn.get_config()  
        clf_hidden_layers_config = self.clf_hidden_layers.get_config()
        fc_out_config = self.fc_out.get_config()

        # Serialize the encoder layers (which are created dynamically)
        # encoder_layers_config = {name: serialize_keras_object(layer) for name, layer in self._encoder_layers.items()}
        encoder_layers_config = {name: layer.get_config() for name, layer in self._encoder_layers.items()}
        encoder_layers_classes = {
            name: layer.__class__.__name__ for name, layer in self._encoder_layers.items()
        }

        config = {
            **base_config,
            "n_features": self._n_features,
            "attn_norm_fn": self._attn_norm_fn,
            "target_activation": self.target_activation,
            "clf_num_layers": self._clf_num_layers,
            "clf_hidden_units": self._clf_hidden_units,
            "feature_indices": self.feature_indices,
            "preprocess_config": preprocess_config,
            "ife_attn_config": ife_attn_config,
            "clf_hidden_layers_config": clf_hidden_layers_config,
            "fc_out_config": fc_out_config,
            "encoder_layers_config": encoder_layers_config,
            "encoder_layers_classes": encoder_layers_classes
        }
        return config

    @classmethod
    def from_config(cls, config):
        # Restore the base configuration from the parent class (_IFEModule)
        data_config = DataConfig.from_config(config['data_config'])
        model_config = ModelConfig.from_config(config['model_config'])
        
        # Create an instance of IFENetRegressor with the restored configurations
        instance = cls(data_config, model_config)

        # Set the custom configurations for IFENetRegressor
        instance._attn_norm_fn = config["attn_norm_fn"]
        instance.target_activation = config["target_activation"]
        instance._clf_num_layers = config["clf_num_layers"]
        instance._clf_hidden_units = config["clf_hidden_units"]
        instance._n_features = config["n_features"]
        instance.feature_indices = config["feature_indices"]
    
        # Deserialize and set layers
        instance._preprocess = tf.keras.layers.BatchNormalization.from_config(config["preprocess_config"])
        instance._ife_attn = _IterativeFeatureExclusion.from_config(config["ife_attn_config"])
        instance.clf_hidden_layers = tf.keras.Sequential.from_config(config["clf_hidden_layers_config"])
        instance.fc_out = tf.keras.layers.Dense.from_config(config["fc_out_config"])

        # Deserialize the encoder layers and assign them to the model
        #encoder_layers = {name: deserialize_keras_object(layer_config) for name, layer_config in config["encoder_layers"].items()}
        #instance._encoder_layers = encoder_layers
        encoder_layers_config = config["encoder_layers_config"]
        encoder_layers_classes = config["encoder_layers_classes"]
        for name, layer_class_name in encoder_layers_classes.items():
            if layer_class_name == "_CategoricalEncodingLayer":
                layer_class = _CategoricalEncodingLayer
            elif layer_class_name == "_NumericalEncodingLayer":
                layer_class = _NumericalEncodingLayer
            else:
                raise ValueError(f"Unknown encoder layer class: {layer_class_name}")
    
            layer = layer_class.from_config(encoder_layers_config[name])
            instance._encoder_layers[name] = layer
    
        return instance

@register_keras_serializable(name="ifeNetClassifier")
class IFENetClassifier(_IFEModule):
    def __init__(self, data_config, model_config, target_activation='softmax', name="ifeNetClassifier", **kwargs):
        super(IFENetClassifier, self).__init__(data_config, model_config)

        self._attn_norm_fn = 'softmax'
        self.target_activation = target_activation
        self._model_config = model_config

        self._clf_num_layers = self._model_config.clf_num_layers
        self._clf_hidden_units = self._model_config.clf_hidden_units
        self._clf_dropout = self._model_config.clf_dropout

        self._n_features = 0

    def build_model(self, dataset):
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError(f"Input must be a tf.data.Dataset, got {type(dataset)}.")

        self.data_batch = next(iter(dataset.map(lambda x, y: x)))
        feature_dtypes = {key: spec.dtype for key, spec in dataset.element_spec[0].items()}
        feature_names = list(feature_dtypes.keys())
        
        self._n_features = self._create_encoder_layers(dataset, feature_names, feature_dtypes)

        self._preprocess = tf.keras.layers.BatchNormalization(name=f"{self.name}/preprocess_batch_norm")

        # Determine the number of classes
        labels = next(iter(dataset.map(lambda x,y: y))).numpy()
        n_outputs = np.size(np.unique(labels))

        self._ife_attn = _IterativeFeatureExclusion(self._n_features, n_outputs, self._attn_norm_fn, self._num_att, self._r)

        # Build the predictive layers
        clf_hidden_layers = []
        for l in range(0, self._clf_num_layers):
            clf_hidden_layers.append(tf.keras.layers.Dense(units=self._clf_hidden_units[l], activation='relu'))
            clf_hidden_layers.append(tf.keras.layers.Dropout(rate=self._clf_dropout))
        
        self.clf_hidden_layers = tf.keras.Sequential(clf_hidden_layers, name=f"{self.name}/fc_hidden_layers")
        self.fc_out = tf.keras.layers.Dense(units=n_outputs, activation=self.target_activation, name=f"{self.name}/fc_out")
        self.is_built = True
    
    def call(self, inputs, training=False): # (batch, n_features)
        # preprocessing the inputs
        features = [self._encoder_layers[name](inputs[name], training=training) for name in self._encoder_layers]
        features = tf.concat(features, axis=1)
        
        # features are the preprocessed inputs
        # batch_size = tf.shape(features)[0]      
        x = self._preprocess(features, training=training) # (batch, n_features)
        
        self.input_scores = self._ife_attn(x, training=training)
        x = x * self.input_scores
        x = self.clf_hidden_layers(x, training=training)
        outputs = self.fc_out(x, training=training)
        
        return outputs

    def get_config(self):
        # Serialize configuration of parent class (_IFEModule)
        base_config = super(IFENetClassifier, self).get_config()

        # Serialize the layer configurations for the layers created in build_model
        preprocess_config = self._preprocess.get_config()  
        ife_attn_config = self._ife_attn.get_config()  
        clf_hidden_layers_config = self.clf_hidden_layers.get_config()
        fc_out_config = self.fc_out.get_config()

        # Serialize the encoder layers (which are created dynamically)
        # encoder_layers_config = {name: serialize_keras_object(layer) for name, layer in self._encoder_layers.items()}
        encoder_layers_config = {name: layer.get_config() for name, layer in self._encoder_layers.items()}
        encoder_layers_classes = {
            name: layer.__class__.__name__ for name, layer in self._encoder_layers.items()
        }
        
        config = {
            **base_config,
            "n_features": self._n_features,
            "attn_norm_fn": self._attn_norm_fn,
            "target_activation": self.target_activation,
            "clf_num_layers": self._clf_num_layers,
            "clf_hidden_units": self._clf_hidden_units,
            "clf_dropout": self._clf_dropout,
            "feature_indices": self.feature_indices,
            "preprocess_config": preprocess_config,
            "ife_attn_config": ife_attn_config,
            "clf_hidden_layers_config": clf_hidden_layers_config,
            "fc_out_config": fc_out_config,
            "encoder_layers_config": encoder_layers_config,
            "encoder_layers_classes": encoder_layers_classes
        }
        return config

    @classmethod
    def from_config(cls, config):
        # Restore the base configuration from the parent class (_IFEModule)
        data_config = DataConfig.from_config(config['data_config'])
        model_config = ModelConfig.from_config(config['model_config'])
        
        # Create an instance of IFENetClassifier with the restored configurations
        instance = cls(data_config, model_config)

        # Set the custom configurations for IFENetClassifier
        instance._attn_norm_fn = config["attn_norm_fn"]
        instance.target_activation = config["target_activation"]
        instance._clf_num_layers = config["clf_num_layers"]
        instance._clf_hidden_units = config["clf_hidden_units"]
        instance._clf_dropout = config["clf_dropout"]
        instance._n_features = config["n_features"]        
        instance.feature_indices = config["feature_indices"]
    
        # Deserialize and set layers
        instance._preprocess = tf.keras.layers.BatchNormalization.from_config(config["preprocess_config"])
        instance._ife_attn = _IterativeFeatureExclusion.from_config(config["ife_attn_config"])
        instance.clf_hidden_layers = tf.keras.Sequential.from_config(config["clf_hidden_layers_config"])
        instance.fc_out = tf.keras.layers.Dense.from_config(config["fc_out_config"])

        # Deserialize the encoder layers and assign them to the model
        # encoder_layers = {name: deserialize_keras_object(layer_config) for name, layer_config in config["encoder_layers_config"].items()}
        encoder_layers_config = config["encoder_layers_config"]
        encoder_layers_classes = config["encoder_layers_classes"]
        for name, layer_class_name in encoder_layers_classes.items():
            if layer_class_name == "_CategoricalEncodingLayer":
                layer_class = _CategoricalEncodingLayer
            elif layer_class_name == "_NumericalEncodingLayer":
                layer_class = _NumericalEncodingLayer
            else:
                raise ValueError(f"Unknown encoder layer class: {layer_class_name}")
    
            layer = layer_class.from_config(encoder_layers_config[name])
            instance._encoder_layers[name] = layer
    
        return instance
        