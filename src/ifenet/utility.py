"""
Created on Mon Nov 10 09:00:00 2024
@author: Mohd Halim Mohd Noor
"""

import tensorflow as tf
import pandas as pd

def dataframe_to_dataset(dataframe, target_columns, shuffle=True, batch_size=128):
    """
    Converts a Pandas DataFrame to a TensorFlow Dataset.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe.
        target_columns (list): List of column names to use as targets.
        shuffle (bool): Whether to shuffle the dataset.
        batch_size (int): Batch size for the dataset.
    
    Returns:
        tf.data.Dataset: A TensorFlow Dataset object.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input must be a Pandas DataFrame.")
    if not isinstance(target_columns, list):
        raise ValueError("Target columns must be provided as a list.")
    
    try:
        df_copy = dataframe.copy()

        for column in df_copy.select_dtypes(include=['object']).columns:
            df_copy[column] = df_copy[column].str.strip()
        
        for target in target_columns:
            if target not in df_copy.columns:
                raise KeyError(f"Target column '{target}' not found in the DataFrame.")
            if df_copy[target].dtypes == 'object':
                df_copy[target] = df_copy[target].astype('category').cat.codes
            if df_copy[target].dtypes == 'int8' or df_copy[target].dtypes == 'int32' or df_copy[target].dtypes == 'int64':
                df_copy[target] = df_copy[target].astype('float32')
        
        # targets = df_copy.loc[:,target_columns]
        targets = df_copy[target_columns].copy()
        df_copy.drop(columns=target_columns, inplace=True)
        
        df_copy = {key: value.to_numpy()[:,tf.newaxis] for key, value in df_copy.items()}
        dataset = tf.data.Dataset.from_tensor_slices((dict(df_copy), targets))
        
        if shuffle:
            dataset = dataset.shuffle(batch_size*2).batch(batch_size).prefetch(batch_size)
        else:
            dataset = dataset.batch(batch_size).prefetch(batch_size)
        return dataset
        
    except KeyError:
        raise KeyError(f"Target column '{e.args[0]}' is missing from the DataFrame.")

def decode_list(loaded_list):
    decoded_list = []
    for item in loaded_list:
        if isinstance(item, dict) and 'config' in item and 'value' in item['config']:
            val = bytes(item['config']['value'], 'utf-8')
            decoded_list.append(val)
        else:
            val = bytes(item, 'utf-8')
            decoded_list.append(item)
    return decoded_list
