"""
Created on Mon Nov 10 09:00:00 2024
@author: Mohd Halim Mohd Noor
"""

from typing import List, Union, Optional

class DataConfig():
    """
    Configuration for dataset preprocessing.
    
    Attributes:
        categorical_column_names: List of categorical column names.
        numerical_column_names: List of numerical column names.
        encode_category: Encode categorical features ('embedding', 'category').
        embedding_output_dim: Embedding output dimension. ('auto' or a positive integer).
        category_output_mode: How categorical features are encoded ('one_hot', 'multi_hot').
        is_normalization: Whether to normalize numerical features (True or False).
    """
    def __init__(
        self, 
        categorical_column_names: List[str], 
        numerical_column_names: List[str],  
        encode_category: str = 'embedding',
        embedding_output_dim: Union[str, int] = 'auto',
        category_output_mode: str = 'one_hot', 
        is_normalization: bool = False
    ):
        if not isinstance(categorical_column_names, list) or not all(isinstance(col, str) for col in categorical_column_names):
            raise TypeError("categorical_column_names must be a list of strings.")
        
        if not isinstance(numerical_column_names, list) or not all(isinstance(col, str) for col in numerical_column_names):
            raise TypeError("numerical_column_names must be a list of strings.")

        if encode_category not in {'embedding', 'category'}:
            raise ValueError("encode_category must be 'embedding' or 'category'.")

        if not isinstance(is_normalization, bool):
            raise TypeError("is_normalization must be a boolean value.")

        if encode_category == 'embedding':
            if isinstance(embedding_output_dim, str) and embedding_output_dim != 'auto':
                raise ValueError("embedding_output_dim must be 'auto' or an integer.")
            elif isinstance(embedding_output_dim, int) and embedding_output_dim < 0:
                raise ValueError("embedding_output_dim must be 'auto' or an integer. If embedding_output_dim is an integer, it must be greater than 0.")

        if encode_category == 'category':
            if category_output_mode not in {'one_hot', 'multi_hot'}:
                raise ValueError("category_output_mode must be 'one_hot' or 'multi_hot'.")
        
        self.categorical_column_names = categorical_column_names
        self.numerical_column_names = numerical_column_names
        self.encode_category = encode_category
        self.embedding_output_dim = embedding_output_dim
        self.category_output_mode = category_output_mode
        self.is_normalization = is_normalization
        
    def get_config(self):
        # Return the configuration parameters as a dictionary
        config = {
            "categorical_column_names": self.categorical_column_names,
            "numerical_column_names": self.numerical_column_names,
            "encode_category": self.encode_category,
            "embedding_output_dim": self.embedding_output_dim,
            "category_output_mode": self.category_output_mode,
            "is_normalization": self.is_normalization
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            config["categorical_column_names"],
            config["numerical_column_names"],
            config["encode_category"],
            config["embedding_output_dim"],
            config["category_output_mode"],
            config["is_normalization"]
        )
            
class ModelConfig():
    """
    Configuration for model architecture.
    
    Attributes:
        num_att: Number of attention heads.
        r: A scaling factor for amplifying the attention weights. Must be 1 or greater.
        clf_num_layers: Number of layers in the predictive layers. Must be 1 or greater.
        clf_hidden_units: Hidden units in the classification head. 
                          Must align with clf_num_layers.
        reduction_layer: Method for dimensionality reduction ('flatten', 'average', 'max').
    """
    def __init__(
        self, 
        num_att: int = 8, 
        r: float = 3.0, 
        clf_num_layers: int = 1, 
        clf_hidden_units: List[int] = [32], 
        clf_dropout: float = 0.3,
    ):
        
        if not isinstance(num_att, int) or num_att <= 0:
            raise ValueError("num_att must be a positive integer and greater than 0.")

        if not isinstance(r, (float, int)) or r < 1:
            raise ValueError("r must be a float or integer greater than or equal to 1.")

        if not isinstance(clf_num_layers, int) or clf_num_layers < 1:
            raise ValueError("clf_num_layers must be an integer greater than or equal to 1.")
            
        if not isinstance(clf_hidden_units, list) or not all(isinstance(unit, int) and unit > 0 for unit in clf_hidden_units):
            raise TypeError("clf_hidden_units must be a list of positive integers.")
            
        if len(clf_hidden_units) != clf_num_layers:
            raise ValueError(
                f"clf_hidden_units must have exactly {clf_num_layers} elements. "
                f"Got {len(clf_hidden_units)} elements instead."
            )

        if not isinstance(clf_dropout, float) or (clf_dropout < 0.0 and clf_dropout >= 1.0):
            raise ValueError("clf_dropout must be a positive float and less than 1.0.")
            
        self.num_att = num_att
        self.r = r
        self.clf_num_layers = clf_num_layers
        self.clf_hidden_units = clf_hidden_units
        self.clf_dropout = clf_dropout

    def get_config(self):
        # Return the configuration parameters as a dictionary
        config = {
            "num_att": self.num_att,
            "r": self.r,
            "clf_num_layers": self.clf_num_layers,
            "clf_hidden_units": self.clf_hidden_units,
            "clf_dropout": self.clf_dropout,
        }
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(
            config["num_att"],
            config["r"],
            config["clf_num_layers"],
            config["clf_hidden_units"],
            config["clf_dropout"],
        )