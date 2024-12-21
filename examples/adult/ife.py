"""
Created on Mon Nov 10 14:10:00 2024
@author: Fathe, Abdulrahman and Mohd Halim
"""

import tensorflow as tf
import numpy as np

class Attention(tf.keras.layers.Layer):
    def __init__(self, units, attn_norm_fn, num_att, r=2, initializer="glorot_uniform"):
        super(Attention, self).__init__()
        self.units = units # number of classes/responses
        self.num_att = num_att
        self.r = r
        self.initializer = initializer
        if attn_norm_fn == 'sigmoid':
            self.norm_function = tf.keras.layers.Activation(activation='sigmoid')
        else:
            self.norm_function = tf.keras.layers.Softmax()

    def build(self, input_shape): # input_shape = (batch, n_features)
        self.kernel = self.add_weight(shape=(self.num_att, input_shape[-1], self.units),
                                      initializer=self.initializer,
                                      trainable=True,
                                      name='kernel') # shape = (num_att, n_features, n_outputs)

    def call(self, inputs): # input_shape = (batch, n_features)
        z = tf.matmul(inputs, self.kernel) # (batch, n_features) dot (num_att, n_features, n_outputs) = (num_att, batch, n_outputs)
        # z = tf.nn.softmax(z, axis=-1) # (num_att, batch, n_outputs)
        z = self.norm_function(z) # (num_att, batch, n_outputs)
        
        w = tf.math.exp(self.kernel * self.r) # amplify weights
        outputs = tf.matmul(z, tf.transpose(w, perm=(0,2,1)))  # (num_att, batch, n_outputs) dot (num_att, n_outputs, n_features) = (num_att, batch, n_features)
        # outputs = tf.reduce_mean(a, axis=[1])  # shape = (batch, n_features)
        return outputs # (num_att, batch, n_features)

class IterativeFeatureExclusion(tf.keras.layers.Layer):
    def __init__(self, n_features, n_outputs, attn_norm_fn, num_att=8, r=2):
        super(IterativeFeatureExclusion, self).__init__()
        self.attentions = [Attention(n_outputs, attn_norm_fn, num_att, r=r) for i in range(n_features)]
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
        #input_scores = tf.reduce_mean(input_scores, axis=[-1, 0]) # shape = (batch, n_features)
        #input_scores = tf.nn.softmax(input_scores, axis=-1) # shape = (batch, n_features)
        input_scores = tf.reduce_mean(input_scores, axis=[-1]) # shape = (num_att, batch, n_features)
        input_scores = tf.nn.softmax(input_scores, axis=-1) # shape = (num_att, batch, n_features)
        return input_scores

class IFEModule(tf.keras.Model):
    def __init__(self, attn_norm_fn, **kwargs):
        super(IFEModule, self).__init__()

        self._attn_norm_fn = attn_norm_fn
        self._n_features = kwargs['n_features']
        self._n_outputs = kwargs['n_outputs']
        self._num_att = kwargs['num_att']
        self._r = kwargs['r']
        self._ife_num_layers = kwargs['ife_num_layers'] # not use, ignore (set to 1)

        self.input_scores = None
        self._preprocess = tf.keras.layers.BatchNormalization(input_shape=(self._n_features,), name='preprocess_batch_norm')
        self._ife_attn = IterativeFeatureExclusion(self._n_features, self._n_outputs, self._attn_norm_fn, self._num_att, self._r)

    def call(self):
        pass
    
class IFENetRegressor(IFEModule):
    def __init__(self, target_activation='linear', **kwargs):
        super(IFENetRegressor, self).__init__(attn_norm_fn='sigmoid', **kwargs)

        self._clf_num_layers = kwargs['clf_num_layers']
        self._clf_hidden_units = kwargs['clf_hidden_units']
        self._reduction = kwargs['reduction_layer']

        clf_hidden_layers = []
        for l in range(0, self._clf_num_layers):
            clf_hidden_layers.append(tf.keras.layers.Dense(units=self._clf_hidden_units[l], activation='relu'))
            clf_hidden_layers.append(tf.keras.layers.BatchNormalization())

        if self._reduction == 'flatten':
            self._reduction_layer = tf.keras.layers.Flatten()
        elif self._reduction == 'average':
            self._reduction_layer = tf.keras.layers.GlobalAveragePooling1D()
        
        self.clf_hidden_layers = tf.keras.Sequential(clf_hidden_layers, name='fc_hidden_layers')
        self.fc_out = tf.keras.layers.Dense(units=self._n_outputs, activation=target_activation, name='fc_out')

    def call(self, inputs): # (batch, n_features)
        batch_size = tf.shape(inputs)[0]
        x = self._preprocess(inputs) # (batch, n_features)
        norm_inputs = x
        norm_inputs = tf.broadcast_to(norm_inputs, [self._num_att, batch_size, self._n_features]) # expand and broadcast it to the shape of input_scores
        
        self.input_scores = self._ife_attn(x)
        x = norm_inputs * self.input_scores # (head, batch, n_features)

        x = tf.transpose(x, perm=(1,0,2)) # (batch, head, n_features)
        x = self._reduction_layer(x)

        x = self.clf_hidden_layers(x)
        outputs = self.fc_out(x)
        return outputs

class IFENetClassifier(IFEModule):
    def __init__(self, target_activation='softmax', **kwargs):
        super(IFENetClassifier, self).__init__(attn_norm_fn='softmax', **kwargs)

        self._clf_num_layers = kwargs['clf_num_layers']
        self._clf_hidden_units = kwargs['clf_hidden_units']
        self._reduction = kwargs['reduction_layer']

        clf_hidden_layers = []
        for l in range(0, self._clf_num_layers):
            clf_hidden_layers.append(tf.keras.layers.Dense(units=self._clf_hidden_units[l], activation='relu'))
            clf_hidden_layers.append(tf.keras.layers.BatchNormalization())

        if self._reduction == 'flatten':
            self._reduction_layer = tf.keras.layers.Flatten()
        elif self._reduction == 'average':
            self._reduction_layer = tf.keras.layers.GlobalAveragePooling1D()
        
        self.clf_hidden_layers = tf.keras.Sequential(clf_hidden_layers, name='fc_hidden_layers')
        self.fc_out = tf.keras.layers.Dense(units=self._n_outputs, activation=target_activation, name='fc_out')

    def call(self, inputs): # (batch, n_features)
        batch_size = tf.shape(inputs)[0]
        x = self._preprocess(inputs) # (batch, n_features)
        norm_inputs = x
        norm_inputs = tf.broadcast_to(norm_inputs, [self._num_att, batch_size, self._n_features]) # expand and broadcast it to the shape of input_scores
        
        self.input_scores = self._ife_attn(x)
        x = norm_inputs * self.input_scores
        
        x = tf.transpose(x, perm=(1,0,2))
        x = self._reduction_layer(x)
        
        x = self.clf_hidden_layers(x)
        outputs = self.fc_out(x)
        return outputs