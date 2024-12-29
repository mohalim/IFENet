"""
Created on Mon Nov 10 09:00:00 2024
@author: Mohd Halim Mohd Noor
"""

import tensorflow as tf
import numpy as np
import pandas as pd

from .config import DataConfig, ModelConfig
from tensorflow.keras.saving import register_keras_serializable, serialize_keras_object, deserialize_keras_object
from tensorflow.keras.layers import Lambda

@register_keras_serializable(name="_attention")
class _Attention(tf.keras.layers.Layer):
    """
    A custom attention layer in TensorFlow Keras that computes attention scores and weighted outputs 
    based on a kernel and an attention normalization function.

    This layer uses an attention mechanism to compute weighted feature outputs from input data based on 
    a learned kernel and a specified normalization function (sigmoid or softmax). The resulting attention-weighted
    outputs can be used in downstream tasks like classification or regression.

    Attributes:
        units (int): The number of units (e.g., classes or responses) in the output.
        attn_norm_fn (str): The normalization function for attention scores. Can be 'sigmoid' or 'softmax'.
        num_att (int): The number of attention heads or attention vectors.
        r (float): A scaling factor for amplifying the attention weights.
        initializer (str): The initializer to use for the kernel weights (default is 'glorot_uniform').
        kernel (tf.Tensor): The learned kernel weights that represent the attention mechanism.
        norm_function (tf.keras.layers.Layer): The attention normalization function (sigmoid or softmax).

    Methods:
        build(input_shape):
            Initializes the kernel weight matrix based on the input shape and the number of attention heads.
        
        call(inputs):
            Computes the attention-weighted output by applying the kernel to the inputs, normalizing the result 
            using the specified function, and amplifying the attention weights.

        get_config():
            Returns the configuration of the layer, which includes the parameters used to initialize the layer.
    """
    def __init__(self, units, attn_norm_fn, num_att, r=2, initializer="glorot_uniform", name="_attention", **kwargs):
        super(_Attention, self).__init__()
        self.units = units # number of classes/responses
        self.attn_norm_fn = attn_norm_fn
        self.num_att = num_att
        self.r = r
        self.initializer = initializer
        if self.attn_norm_fn == 'sigmoid':
            self.norm_function = tf.keras.layers.Activation(activation='sigmoid')
        else:
            self.norm_function = tf.keras.layers.Softmax()

        self.kernel = None

    def build(self, input_shape): # input_shape = (batch, n_features)
        self.kernel = self.add_weight(shape=(self.num_att, input_shape[-1], self.units),
                                      initializer=self.initializer,
                                      trainable=True,
                                      name=f"{self.name}/kernel") # shape = (num_att, n_features, n_outputs)

    def call(self, inputs): # input_shape = (batch, n_features)
        z = tf.matmul(inputs, self.kernel) # (batch, n_features) dot (num_att, n_features, n_outputs) = (num_att, batch, n_outputs)
        # z = tf.nn.softmax(z, axis=-1) # (num_att, batch, n_outputs)
        z = self.norm_function(z) # (num_att, batch, n_outputs)
        
        w = tf.math.exp(self.kernel * self.r) # amplify attention weights
        outputs = tf.matmul(z, tf.transpose(w, perm=(0,2,1)))  # (num_att, batch, n_outputs) dot (num_att, n_outputs, n_features) = (num_att, batch, n_features)
        # outputs = tf.reduce_mean(a, axis=[1])  # shape = (batch, n_features)
        return outputs # (num_att, batch, n_features)

    def get_config(self):
        # Return the configuration parameters as a dictionary
        config = super(_Attention, self).get_config()        
        config.update({
            "units": self.units,
            "attn_norm_fn": self.attn_norm_fn,
            "num_att": self.num_att,
            "r": self.r,
            "initializer": self.initializer,            
            })
        return config
     
    @classmethod
    def from_config(cls, config):
        #layer = cls(
        #    units = config["units"],
        #    attn_norm_fn = config["attn_norm_fn"],
        #    num_att = config["num_att"],
        #    r = config["r"],
        #    initializer = config["initializer"]
        #)
        return cls(**config)

@register_keras_serializable(name="_iterativeFeatureExclusion")
class _IterativeFeatureExclusion(tf.keras.layers.Layer):
    def __init__(self, n_features, n_outputs, attn_norm_fn, num_att=8, r=2, name="_iterativeFeatureExclusion", **kwargs):
        super(_IterativeFeatureExclusion, self).__init__()

        self.n_features = n_features
        self.n_outputs = n_outputs
        self.attn_norm_fn = attn_norm_fn
        self.num_att = num_att
        self.r = r
        
        self.attentions = [_Attention(self.n_outputs, self.attn_norm_fn, self.num_att, self.r) for i in range(self.n_features)]
        mask_ones = np.ones((n_features,), dtype=np.int8)
        self.masks = []
        for j in range(0,n_features):
            mask = mask_ones.copy()
            mask[j] = 0
            self.masks.append(tf.constant(mask, dtype=tf.float32))
        #self.masks = tf.stack(self.masks, axis=1)

    def call(self, inputs):       # input shape = (batch, n_features)
        input_scores = []
        for mask, attention in zip(self.masks,self.attentions):
            inputs_masked = inputs * mask # shape = (num_att, batch, n_features)
            z = tf.expand_dims(attention(inputs_masked), axis=-1) # (num_att, batch, n_features, 1)
            input_scores.append(z)
            
        input_scores = tf.concat(input_scores, axis=-1) # shape = (num_att, batch, n_features, n_features)
        input_scores = tf.reduce_mean(input_scores, axis=[-1]) # shape = (num_att, batch, n_features)
        input_scores = tf.nn.softmax(input_scores, axis=-1) # shape = (num_att, batch, n_features)
        return input_scores

    def get_config(self):
        # Serialize the list of attention layers using serialize_keras_object
        attention_configs = [serialize_keras_object(attn) for attn in self.attentions]

        # Convert masks into a list of arrays
        #masks = [mask.numpy() for mask in self.masks]

        # Return a configuration dictionary including parameters and serialized layers
        base_config = super(_IterativeFeatureExclusion, self).get_config()
        config = {
            **base_config,
            "n_features": self.n_features,
            "n_outputs": self.n_outputs,
            "attn_norm_fn": self.attn_norm_fn,
            "num_att": self.num_att,
            "r": self.r,
            "attentions": attention_configs,  # serialized attention layers
            #"masks": masks  # serialized masks
        }
        return config

    @classmethod
    def from_config(cls, config):
        # Reconstruct the attention layers from the serialized configurations
        attentions = [deserialize_keras_object(attn_config) for attn_config in config["attentions"]]

        # Reconstruct masks
        #masks = [tf.constant(mask, dtype=tf.float32) for mask in config["masks"]]

        # Reconstruct the layer
        layer = cls(
            n_features=config["n_features"],
            n_outputs=config["n_outputs"],
            attn_norm_fn=config["attn_norm_fn"],
            num_att=config["num_att"],
            r=config["r"]
        )
        # Assign the reconstructed attentions and masks to the layer
        layer.attentions = attentions
        #layer.masks = masks
        return layer

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
        self._category_output_mode = self._data_config.category_output_mode
        self._is_normalization = self._data_config.is_normalization
        
        self._num_att = self._model_config.num_att
        self._r = self._model_config.r
        # self._ife_num_layers = model_config.ife_num_layers

        self._n_features = 0
        self._encoder_layers = {}

        self.data_batch = None
        self.feature_indices = {}
        self.input_scores = None

    def _get_category_encoding_layer(self, name, dataset, dtype, max_tokens=None):
        feature_ds = dataset.map(lambda x, y: x[name])
        if dtype == tf.string:
            index = tf.keras.layers.StringLookup(max_tokens=max_tokens)
        elif dtype == tf.int64:
            index = tf.keras.layers.IntegerLookup(max_tokens=max_tokens)
        
        index.adapt(feature_ds)
        encoder = tf.keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size(), output_mode=self._category_output_mode, name=name)
        return lambda feature: encoder(index(feature))
    
    def _get_numerical_encoding_layer(self, name, dataset):
        feature_ds = dataset.map(lambda x, y: x[name])
        
        if self._is_normalization:
            encoder = tf.keras.layers.Normalization(axis=None)
            encoder.adapt(feature_ds)
            return lambda feature: encoder(feature)
            return encoder
        else:
            return lambda feature: tf.cast(feature, dtype=tf.float32)
        
    def _create_encoder_layers(self, dataset, feature_names, feature_dtypes):
        for name in feature_names:
            if name in self._categorical_column_names:
                layer = Lambda(self._get_category_encoding_layer(name, dataset, feature_dtypes[name]))
                self._encoder_layers[name] = layer
            elif name in self._numerical_column_names:
                layer = Lambda(self._get_numerical_encoding_layer(name, dataset))
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
        if not tf.is_symbolic_tensor(self.input_scores):
            #self(self.data_batch)
            reduction = (0, 1)
            feature_scores = np.mean(self.input_scores, axis=reduction)
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

    def fit(self, train_ds, validation_data=None, epochs=1, batch_size=None, verbose=1, callbacks=None):
        """
        Override fit to ensure model is built before training.
        """
        if not self.is_built:
            raise ValueError("Model has not been built. Please run `model.build_model(train_ds)` before calling `fit()`.")

        # Call the original fit() method (or perform custom training loop if needed)
        super(_IFEModule, self).fit(train_ds, validation_data=validation_data, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks)

    def get_config(self):
        base_config = super(_IFEModule, self).get_config()
        
        # Serialize the data_config and model_config
        data_config_dict = self._data_config.get_config()
        model_config_dict = self._model_config.get_config()
        
        # encoder_layers_config = {name: serialize_keras_object(layer) for name, layer in self._encoder_layers.items()}

        # Return the complete configuration
        config = {
            **base_config,
            "data_config": data_config_dict,
            "model_config": model_config_dict,
            #"encoder_layers": encoder_layers_config,
        }
        
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize the DataConfig and ModelConfig
        data_config = DataConfig.from_config(config['data_config'])
        model_config = ModelConfig.from_config(config['model_config'])

        # Create an instance of _IFEModule
        instance = cls(data_config, model_config)
        
        # Return the reconstructed model
        return instance
    
@register_keras_serializable(name="ifeNetRegressor")
class IFENetRegressor(_IFEModule):
    def __init__(self, data_config, model_config, name="ifeNetRegressor", **kwargs):
        super(IFENetRegressor, self).__init__(data_config, model_config)

        self._attn_norm_fn = 'sigmoid'
        self.target_activation='linear'
        self._model_config = model_config

        self._clf_num_layers = self._model_config.clf_num_layers
        self._clf_hidden_units = self._model_config.clf_hidden_units
        self._clf_dropout = self._model_config.clf_dropout
        self._reduction = self._model_config.reduction_layer

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

        if self._reduction == 'flatten':
            self._reduction_layer = tf.keras.layers.Flatten(name=f"{self.name}/flatten")
        elif self._reduction == 'average':
            self._reduction_layer = tf.keras.layers.GlobalAveragePooling1D(name=f"{self.name}/global_average_pooling")
        
        self.clf_hidden_layers = tf.keras.Sequential(clf_hidden_layers, name=f"{self.name}/fc_hidden_layers")
        self.fc_out = tf.keras.layers.Dense(units=n_outputs, activation=self.target_activation, name=f"{self.name}/fc_out")
        self.is_built = True
        
    def call(self, inputs): # (batch, n_features)
        # preprocessing the inputs
        features = [self._encoder_layers[name](inputs[name]) for name in self._encoder_layers]
        
        features = tf.concat(features, axis=1)

        # features are the preprocessed inputs
        batch_size = tf.shape(features)[0]
        x = self._preprocess(features) # (batch, n_features)
        norm_inputs = x
        norm_inputs = tf.broadcast_to(norm_inputs, [self._num_att, batch_size, self._n_features]) # expand and broadcast it to the shape of input_scores
        
        self.input_scores = self._ife_attn(x)
        x = norm_inputs * self.input_scores # (head, batch, n_features)

        x = tf.transpose(x, perm=(1,0,2)) # (batch, head, n_features)
        x = self._reduction_layer(x)

        x = self.clf_hidden_layers(x)
        outputs = self.fc_out(x)
        return outputs

    def get_config(self):
        # Serialize configuration of parent class (_IFEModule)
        base_config = super(IFENetRegressor, self).get_config()

        # Serialize the layer configurations for the layers created in build_model
        preprocess_config = self._preprocess.get_config()  
        ife_attn_config = self._ife_attn.get_config()  
        reduction_config = self._reduction_layer.get_config()
        # clf_hidden_layers_config = [layer.get_config() for layer in self.clf_hidden_layers.layers]
        clf_hidden_layers_config = self.clf_hidden_layers.get_config()
        fc_out_config = self.fc_out.get_config()

        # Serialize the encoder layers (which are created dynamically)
        encoder_layers_config = {name: serialize_keras_object(layer) for name, layer in self._encoder_layers.items()}

        config = {
            **base_config,
            "n_features": self._n_features,
            "attn_norm_fn": self._attn_norm_fn,
            "target_activation": self.target_activation,
            "clf_num_layers": self._clf_num_layers,
            "clf_hidden_units": self._clf_hidden_units,
            "reduction": self._reduction,
            "feature_indices": self.feature_indices,
            "reduction_layer": reduction_config,
            "preprocess_config": preprocess_config,
            "ife_attn_config": ife_attn_config,
            "clf_hidden_layers_config": clf_hidden_layers_config,
            "fc_out_config": fc_out_config,
            "encoder_layers": encoder_layers_config,
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
        instance._reduction = config["reduction"]
        instance._n_features = config["n_features"]
        instance.feature_indices = config["feature_indices"]
    
        # Deserialize and set layers
        instance._preprocess = tf.keras.layers.BatchNormalization.from_config(config["preprocess_config"])
        instance._ife_attn = _IterativeFeatureExclusion.from_config(config["ife_attn_config"])
        instance.clf_hidden_layers = tf.keras.Sequential.from_config(config["clf_hidden_layers_config"])
        instance.fc_out = tf.keras.layers.Dense.from_config(config["fc_out_config"])

        instance._reduction_layer = tf.keras.layers.Flatten() if instance._reduction == "flatten" else tf.keras.layers.GlobalAveragePooling1D()

        # Deserialize the encoder layers and assign them to the model
        encoder_layers = {name: deserialize_keras_object(layer_config) for name, layer_config in config["encoder_layers"].items()}
        instance._encoder_layers = encoder_layers
    
        return instance

@register_keras_serializable(name="ifeNetClassifier")
class IFENetClassifier(_IFEModule):
    def __init__(self, data_config, model_config, name="ifeNetClassifier", **kwargs):
        super(IFENetClassifier, self).__init__(data_config, model_config)

        self._attn_norm_fn = 'softmax'
        self.target_activation = 'softmax'
        self._model_config = model_config

        self._clf_num_layers = self._model_config.clf_num_layers
        self._clf_hidden_units = self._model_config.clf_hidden_units
        self._clf_dropout = self._model_config.clf_dropout
        self._reduction = self._model_config.reduction_layer

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

        if self._reduction == 'flatten':
            self._reduction_layer = tf.keras.layers.Flatten(name=f"{self.name}/flatten")
        elif self._reduction == 'average':
            self._reduction_layer = tf.keras.layers.GlobalAveragePooling1D(name=f"{self.name}/global_average_pooling")
        
        self.clf_hidden_layers = tf.keras.Sequential(clf_hidden_layers, name=f"{self.name}/fc_hidden_layers")
        self.fc_out = tf.keras.layers.Dense(units=n_outputs, activation=self.target_activation, name=f"{self.name}/fc_out")
        self.is_built = True
    
    def call(self, inputs): # (batch, n_features)
        # preprocessing the inputs
        features = [self._encoder_layers[name](inputs[name]) for name in self._encoder_layers]
        features = tf.concat(features, axis=1)
        
        # features are the preprocessed inputs
        batch_size = tf.shape(features)[0]
        x = self._preprocess(features) # (batch, n_features)
        norm_inputs = x
        norm_inputs = tf.broadcast_to(norm_inputs, [self._num_att, batch_size, self._n_features]) # expand and broadcast it to the shape of input_scores
        
        self.input_scores = self._ife_attn(x)
        x = norm_inputs * self.input_scores
        
        x = tf.transpose(x, perm=(1,0,2))
        x = self._reduction_layer(x)
        
        x = self.clf_hidden_layers(x)
        outputs = self.fc_out(x)
        return outputs

    def get_config(self):
        # Serialize configuration of parent class (_IFEModule)
        base_config = super(IFENetClassifier, self).get_config()

        # Serialize the layer configurations for the layers created in build_model
        preprocess_config = self._preprocess.get_config()  
        ife_attn_config = self._ife_attn.get_config()  
        reduction_config = self._reduction_layer.get_config()
        # clf_hidden_layers_config = [layer.get_config() for layer in self.clf_hidden_layers.layers]
        clf_hidden_layers_config = self.clf_hidden_layers.get_config()
        fc_out_config = self.fc_out.get_config()

        # Serialize the encoder layers (which are created dynamically)
        encoder_layers_config = {name: serialize_keras_object(layer) for name, layer in self._encoder_layers.items()}
        
        config = {
            **base_config,
            "n_features": self._n_features,
            "attn_norm_fn": self._attn_norm_fn,
            "target_activation": self.target_activation,
            "clf_num_layers": self._clf_num_layers,
            "clf_hidden_units": self._clf_hidden_units,
            "clf_dropout": self._clf_dropout,
            "reduction": self._reduction,
            "feature_indices": self.feature_indices,   
            "reduction_layer": reduction_config,
            "preprocess_config": preprocess_config,
            "ife_attn_config": ife_attn_config,
            "clf_hidden_layers_config": clf_hidden_layers_config,
            "fc_out_config": fc_out_config,
            "encoder_layers": encoder_layers_config,
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
        instance._reduction = config["reduction"]
        instance._n_features = config["n_features"]        
        instance.feature_indices = config["feature_indices"]
    
        # Deserialize and set layers
        instance._preprocess = tf.keras.layers.BatchNormalization.from_config(config["preprocess_config"])
        instance._ife_attn = _IterativeFeatureExclusion.from_config(config["ife_attn_config"])
        instance.clf_hidden_layers = tf.keras.Sequential.from_config(config["clf_hidden_layers_config"])
        instance.fc_out = tf.keras.layers.Dense.from_config(config["fc_out_config"])

        instance._reduction_layer = tf.keras.layers.Flatten() if instance._reduction == "flatten" else tf.keras.layers.GlobalAveragePooling1D()

        # Deserialize the encoder layers and assign them to the model
        encoder_layers = {name: deserialize_keras_object(layer_config) for name, layer_config in config["encoder_layers"].items()}
        instance._encoder_layers = encoder_layers
    
        return instance
        