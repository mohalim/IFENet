"""
Created on Mon Nov 10 09:00:00 2024
@author: Mohd Halim Mohd Noor
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import math

from tensorflow.keras.saving import register_keras_serializable, serialize_keras_object, deserialize_keras_object
from .utility import decode_list

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

    def call(self, inputs, training=False): # input_shape = (batch, n_features)
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

        self.fc_scores = tf.keras.layers.Dense(units=1, activation=None)
        # self.conv1d_scores = tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, activation=None, padding='same')
        self.bn_scores = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):       # input shape = (batch, n_features)
        scores_list = []
        for mask, attention in zip(self.masks,self.attentions):
            inputs_masked = inputs * mask # shape = (num_att, batch, n_features)
            z = tf.expand_dims(attention(inputs_masked), axis=-1) # (num_att, batch, n_features, 1)
            scores_list.append(z)
            
        x = tf.concat(scores_list, axis=-1) # shape = (num_att, batch, n_features, n_features)
        x = tf.reduce_mean(x, axis=[-1]) # shape = (num_att, batch, n_features)
        x = tf.transpose(x, perm=(1,2,0)) # (batch, n_features, num_att)
        # x = self.conv1d_scores(x) # (batch, n_features, 1)
        x = self.fc_scores(x, training=training) # (batch, n_features, 1)
        x = self.bn_scores(x, training=training)
        x = tf.squeeze(x, axis=-1)
        scores = tf.nn.softmax(x, axis=-1) # shape = (num_att, batch, n_features)
        
        return scores

    def get_config(self):
        # Serialize the list of attention layers using serialize_keras_object
        fc_scores_config = self.fc_scores.get_config()
        bn_scores_config = self.bn_scores.get_config()
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
            "fc_scores_config": fc_scores_config,
            "bn_scores_config": bn_scores_config
            # "masks": masks  # serialized masks
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
        layer.fc_scores = tf.keras.layers.Dense.from_config(config["fc_scores_config"])
        layer.bn_scores = tf.keras.layers.BatchNormalization.from_config(config["bn_scores_config"])
        #layer.masks = masks
        return layer

@register_keras_serializable(name="_stringLookup")
class _StringLookup(tf.keras.layers.Layer):
    def __init__(self, vocabulary=None, max_tokens=None, oov_token='[UNK]', name="_stringLookup", **kwargs):
        super(_StringLookup, self).__init__()
        self.layer_name = name
        self.max_tokens = max_tokens
        self.oov_token = oov_token
        self.vocabulary = None
        self.vocab_size = None

        if vocabulary is not None:
            self.vocabulary = vocabulary[:max_tokens] if max_tokens is not None else vocabulary
            if self.oov_token not in self.vocabulary:
                self.vocabulary.insert(0, self.oov_token)
            self.vocab_size = len(vocabulary)

        #print(f'_StringLookup (init): {self.vocabulary}')

    def adapt(self, data):
        if self.vocabulary is None:
            if isinstance(data, tf.data.Dataset):
                data = list(data.as_numpy_iterator())[0]
                if data.ndim > 1:
                    data = data.ravel()

            if data is not None:
                # Create a set of unique tokens
                unique_tokens = set(data)
            
                # Limit the vocabulary size if max_tokens is specified
                if self.max_tokens:
                    unique_tokens = list(unique_tokens)[:self.max_tokens]
        
                unique_tokens = list(unique_tokens)
                unique_tokens.insert(0, self.oov_token)    
                self.vocabulary = list(unique_tokens)
                self.vocab_size = len(self.vocabulary)

            #print(f'_StringLookup (adapt): {self.vocabulary}')

    def build(self, input_shape):
        if self.vocabulary is not None:            
            initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.constant(self.vocabulary, dtype=tf.string), 
                                                              values=tf.range(self.vocab_size, dtype=tf.int64))

            # default_value = self.vocabulary[-1]
            default_value = tf.constant(self.vocabulary.index(self.oov_token), dtype=tf.int64)
            self.string_table = tf.lookup.StaticHashTable(initializer=initializer, 
                                                          default_value=default_value)
            super(_StringLookup, self).build(input_shape)

            # print(f'_StringLookup (build): {self.vocabulary}')
            
        else:
            print(f'Table is not initialized. Run adapt to initialize the table.')

    def get_vocabulary(self):
        return self.vocabulary
    
    def vocabulary_size(self):
        return self.vocab_size

    def call(self, inputs, training=False):
        inputs = tf.strings.as_string(inputs)
        return self.string_table.lookup(inputs)

    def get_config(self):
        config = super(_StringLookup, self).get_config()
        config.update({
            "max_tokens": self.max_tokens,
            "oov_token": self.oov_token,
            #"vocabulary": [str(item) for item in self.vocabulary],
            #"vocabulary": [str(item.encode('utf-8)) for item in self.vocabulary],
            "vocabulary": self.vocabulary,
            "layer_name": self.layer_name
        })

        #print(f'_StringLookup (get_config): {self.vocabulary}')
        return config

    @classmethod
    def from_config(cls, config):
        # print(f'_StringLookup (from_config): {config["vocabulary"]}')
        decoded_vocab = decode_list(config["vocabulary"])
        # print(f'_StringLookup (from_config): {decoded_vocab}')
        instance = cls(
            vocabulary=decoded_vocab,
            max_tokens=config["max_tokens"],
            oov_token=config["oov_token"],
            name=config["layer_name"]    
        )
        
        # instance.vocabulary = [bytes(item, 'utf-8') for item in config['vocabulary']]
        # instance.vocabulary = config['vocabulary']
        return instance

@register_keras_serializable(name="_integerLookup")
class _IntegerLookup(tf.keras.layers.Layer):
    def __init__(self, vocabulary=None, max_tokens=None, oov_token=-1, name="_integerLookup", **kwargs):
        super(_IntegerLookup, self).__init__()
        self.layer_name = name
        self.max_tokens = max_tokens
        self.oov_token = oov_token
        self.vocabulary = None
        self.vocab_size = None

        if vocabulary is not None:
            self.vocabulary = vocabulary[:max_tokens] if max_tokens is not None else vocabulary
            if self.oov_token not in self.vocabulary:
                self.vocabulary.insert(0, oov_token)
            self.vocab_size = len(vocabulary)

    def adapt(self, data):
        # print(f'adapt: {self.vocabulary}')
        if self.vocabulary is None:
            if isinstance(data, tf.data.Dataset):
                data = list(data.as_numpy_iterator())[0]
                if data.ndim > 1:
                    data = data.ravel()

            if data is not None:
                # Create a set of unique tokens
                unique_tokens = set(data)
            
                if self.max_tokens:
                    unique_tokens = list(unique_tokens)[:self.max_tokens]
        
                unique_tokens = list(unique_tokens)
                unique_tokens.insert(0, self.oov_token)    
                self.vocabulary = list(unique_tokens)
                self.vocab_size = len(self.vocabulary)

    def build(self, input_shape):
        if self.vocabulary is not None:
            initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.constant(self.vocabulary, dtype=tf.int64), 
                                                              values=tf.range(self.vocab_size, dtype=tf.int64))
    
            default_value = tf.constant(self.vocabulary.index(self.oov_token), dtype=tf.int64)
            self.integer_table = tf.lookup.StaticHashTable(initializer=initializer,
                                                           default_value=default_value)
            super(_IntegerLookup, self).build(input_shape)

        else:
            print(f'Table is not initialized. Run adapt to initialize the table.')

    def get_vocabulary(self):
        return self.vocabulary
    
    def vocabulary_size(self):
        return self.vocab_size

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, dtype=tf.int64)
        return self.integer_table.lookup(inputs)

    def get_config(self):
        config = super(_IntegerLookup, self).get_config()
        config.update({
            "max_tokens": self.max_tokens,
            "oov_token": self.oov_token,
            "vocabulary": [int(item) for item in self.vocabulary],
            "layer_name": self.layer_name
        })

        return config

    @classmethod
    def from_config(cls, config):
        layer = cls(
            vocabulary=config["vocabulary"],
            max_tokens=config["max_tokens"],
            oov_token=config["oov_token"],
            name=config["layer_name"]
        )
        # instance.vocabulary = [int(item) for item in config['vocabulary']]
        return layer
        

@register_keras_serializable(name="_categoricalEncodingLayer")
class _CategoricalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 encode_category,
                 embedding_output_dim,
                 category_output_mode, 
                 feature_ds, 
                 feature_dtype, 
                 output_dim=None, 
                 max_tokens=None, 
                 name="_categoricalEncodingLayer", **kwargs):
        
        super(_CategoricalEncodingLayer, self).__init__()
        self.layer_name = name
        self.encode_category = encode_category
        self.embedding_output_dim = embedding_output_dim
        self.category_output_mode = category_output_mode
        self.feature_ds = feature_ds
        self.feature_dtype = feature_dtype
        self.output_dim = output_dim
        self.max_tokens = max_tokens

    def build(self, input_shape):
        if self.feature_ds is not None:
            if self.feature_dtype == tf.string:
                self.index = _StringLookup(max_tokens=self.max_tokens, name="_stringLookup_"+self.layer_name)
                self.index.adapt(self.feature_ds)
                
            elif self.feature_dtype in [tf.int8, tf.int16, tf.int32, tf.int64]:
                self.index = _IntegerLookup(max_tokens=self.max_tokens, name="_integerLookup_"+self.layer_name)
                self.index.adapt(self.feature_ds)

            vocab_size = self.index.vocabulary_size()
            if vocab_size is None:
                raise ValueError("Vocabulary size is None. Check if `adapt` was called with a valid dataset.")

            if self.encode_category is 'embedding':
                if self.embedding_output_dim == 'auto':
                    # output_dim = min(10,max(2,sqrt(num_unique_cats))) 
                    # where input_dim is the number of unique categories (e.g., size: small, medium and large) and limit to 10 dimensions (maximum)
                    # for example, for 10 categories, output_dim is 3
                    # we use self.index.vocabulary_size() for num. of unique tokens            
                    if self.output_dim is None:
                        self.output_dim = int(min(10, max(2, math.sqrt(vocab_size))))
                else:
                    if self.output_dim is None:
                        self.output_dim = self.embedding_output_dim

                self.encoder = tf.keras.layers.Embedding(input_dim=self.index.vocabulary_size(), output_dim=self.output_dim)

            else:
                self.encoder = tf.keras.layers.CategoryEncoding(num_tokens=self.index.vocabulary_size(), output_mode=self.category_output_mode)
            
            # print(f'In _CategoricalEncodingLayer: {self.index.vocabulary_size()}')

    def call(self, inputs, training=False):
        x = self.index(inputs, training=training)  # index the input feature
        x = self.encoder(x, training=training)  # embedding lookup
        outputs = tf.squeeze(x, axis=1) if len(tf.shape(x)) == 3 else x
        return outputs

    def get_config(self):
        # Return the configuration parameters as a dictionary
        config = super(_CategoricalEncodingLayer, self).get_config()
        # Serialize the layer configurations for the layers created in build_model
        index_config = self.index.get_config()
        encoder_config = self.encoder.get_config()
        
        config.update({
            "encode_category": self.encode_category,
            "embedding_output_dim": self.embedding_output_dim,
            "category_output_mode": self.category_output_mode,
            "feature_ds": None,
            "feature_dtype": self.feature_dtype,
            "output_dim": self.output_dim,
            "max_tokens": self.max_tokens,
            "layer_name": self.layer_name,
            "index_config": index_config,
            "encoder_config": encoder_config
            })
        return config
     
    @classmethod
    def from_config(cls, config):
        # Reconstruct the layer
        #print(f'_CategoricalEncodingLayer: {config["layer_name"]}')
        layer = cls(
            encode_category=config["encode_category"],
            embedding_output_dim=config["embedding_output_dim"],
            category_output_mode=config["category_output_mode"],
            feature_ds=config["feature_ds"],
            feature_dtype=config["feature_dtype"],
            max_tokens=config["max_tokens"],
            output_dim=config["output_dim"],
            name=config["layer_name"]
        )
        
        # Assign the reconstructed lookup and encoder layer
        if config["feature_dtype"] == tf.string:
            layer.index = _StringLookup.from_config(config["index_config"])
            
        elif config["feature_dtype"] in [tf.int8, tf.int16, tf.int32, tf.int64]:
            layer.index = _IntegerLookup.from_config(config["index_config"])

        if config["encode_category"] == 'embedding':
            layer.encoder = tf.keras.layers.Embedding.from_config(config["encoder_config"])
            
        elif config["encode_category"] == 'category':
            layer.encoder = tf.keras.layers.CategoryEncoding.from_config(config["encoder_config"])

        return layer

@register_keras_serializable(name="_numericalEncodingLayer")
class _NumericalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, is_normalization, feature_ds, name="_numericalEncodingLayer", **kwargs):
        super(_NumericalEncodingLayer, self).__init__()
        self.layer_name = name
        self.is_normalization = is_normalization
        self.feature_ds = feature_ds
        self.encoder = None
        
        # Handle normalization or simple casting
        if self.is_normalization:
            self.encoder = tf.keras.layers.Normalization(axis=None)
            self.encoder.adapt(feature_ds)
    
    def call(self, inputs, training=False):
        if self.is_normalization:
            return self.encoder(inputs, training=training)  # Apply normalization if required
        else:
            return tf.cast(inputs, dtype=tf.float32)  # Or just cast to float32

    def get_config(self):
        # Return the configuration parameters as a dictionary
        base_config = super(_NumericalEncodingLayer, self).get_config()
        config = {
            **base_config,
            "is_normalization": self.is_normalization,
            "layer_name": self.layer_name
        }
        # Serialize the layer configurations for the layers created in build_model
        if self.is_normalization:
            encoder_config = self.encoder.get_config()
            config.update({
                "encoder_config": encoder_config,
                })
        return config

    @classmethod
    def from_config(cls, config):
        # Reconstruct the layer
        #print(f'_NumericalEncodingLayer: {config["layer_name"]}')
        layer = cls(
            is_normalization=config["is_normalization"],
            feature_ds=None,
            name=config["layer_name"]
        )
        # Assign the reconstructed lookup and encoder layer
        if config["is_normalization"]:
            encoder.index = tf.keras.layers.Normalization.from_config(config["encoder_config"])

        return layer