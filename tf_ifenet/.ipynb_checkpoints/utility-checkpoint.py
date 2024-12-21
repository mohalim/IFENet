"""
Created on Mon Nov 10 14:10:00 2024
@author: Mohd Halim Mohd Noor
"""

import tensorflow as tf
import pandas as pd

def dataframe_to_dataset(dataframe, target_columns, batch_size=128, shuffle=True):
    """
    Converts a Pandas DataFrame to a TensorFlow Dataset.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe.
        target_columns (list): List of column names to use as targets.
        batch_size (int): Batch size for the dataset.
        shuffle (bool): Whether to shuffle the dataset.
    
    Returns:
        tf.data.Dataset: A TensorFlow Dataset object.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input must be a Pandas DataFrame.")
    if not isinstance(target_columns, list):
        raise ValueError("Target columns must be provided as a list.")
    
    try:
        df_copy = dataframe.copy()
        
        for target in target_columns:
            if target not in df_copy.columns:
                raise KeyError(f"Target column '{target}' not found in the DataFrame.")
            if df_copy[target].dtypes == 'object':
                df_copy[target] = df_copy[target].astype('category').cat.codes
        
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
