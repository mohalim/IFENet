"""
Created on Mon Nov 10 14:10:00 2024
@author: Fathe, Abdulrahman and Mohd Halim
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder

class ColumnEncoder():
    def __init__(self, **kwargs):
        self.columns_cat = kwargs['columns_cat']
        self.columns_num = kwargs['columns_num']
        self.encoders = {}
        self.col_indices = {}

    # fit the encoder
    def fit(self, dataframe):
        for col in dataframe.columns:
            enc_col = dataframe[[col]].to_numpy()
            if col in self.columns_cat: # if categorical, encode
                enc = OneHotEncoder()
                enc_col = enc.fit(enc_col)
                self.encoders[col] = enc

    
    def transform(self, dataframe):
        try:
            encoded_columns = np.empty((len(dataframe),0))
            st = 0
            ed = 0
            for col in dataframe.columns:
                enc_col = dataframe[[col]].to_numpy()
                if col in self.columns_cat: # if categorical, encode
                    enc_col = self.encoders[col].transform(enc_col).toarray()
                    
                ed = st + enc_col.shape[-1]
                index = [st, ed]
                st = ed
                self.col_indices[col] = index
                encoded_columns = np.append(encoded_columns, enc_col, axis=-1)
    
            return encoded_columns
        except:
            print(f'Run fit() to fit the encoder.') 

    # Return a list of features with their scores
    # The feature scores must be passed to this function
    def get_feature_scores(self, input_scores, display=True):         
        feature_scores = {}
        for feat in self.col_indices.keys():
            index = self.col_indices[feat]
            feature_scores[feat] = input_scores[slice(*index)]

        for k in feature_scores.keys():
            print(f'{k}: {feature_scores[k]}') 

        return feature_scores
        
    '''
    # Encode the columns
    def encode(self):
        st = 0
        ed = 0
        for col in self.dataframe.columns:
            enc_col = self.dataframe[[col]].to_numpy()
            if col in self.columns_cat: # if categorical, encode
                enc = OneHotEncoder()
                enc_col = enc.fit_transform(enc_col).toarray()
                self.encoders[col] = enc

            ed = st + enc_col.shape[-1]
            index = [st, ed]
            st = ed
            self.col_indices[col] = index
            self.encoded_columns = np.append(self.encoded_columns, enc_col, axis=-1)

        return self.encoded_columns

    
    # Return encoded columns in numpy array
    def get_encoded_columns(self):
        if self.encoded_columns.size != 0:
            return self.encoded_columns
        else:
            print(f'Run encode() to encode the features.')

    # Return a column specified by feat_name
    def get_column(self, feat_name):
        if self.encoded_columns.size != 0:
            index = self.col_indices[feat_name]
            return self.encoded_columns[:, slice(*index)] # *index unpacks the list
        else:
            print(f'Run encode() to encode the features.')
    '''
    
