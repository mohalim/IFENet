"""
Created on Mon Nov 10 09:00:00 2024
@author: Mohd Halim Mohd Noor
"""

from typing import List, Optional

class DataConfig():
    """
    Configuration for dataset preprocessing.
    
    Attributes:
        categorical_column_names: List of categorical column names.
        numerical_column_names: List of numerical column names.
        category_output_mode: How categorical features are encoded ('one_hot', 'multi_hot', etc.).
        is_normalization: Whether to normalize numerical features.
    """
    def __init__(
        self, 
        categorical_column_names: List[str], 
        numerical_column_names: List[str],  
        category_output_mode: str = 'one_hot', 
        is_normalization: bool = False
    ):
        if not isinstance(categorical_column_names, list) or not all(isinstance(col, str) for col in categorical_column_names):
            raise TypeError("categorical_column_names must be a list of strings.")
        
        if not isinstance(numerical_column_names, list) or not all(isinstance(col, str) for col in numerical_column_names):
            raise TypeError("numerical_column_names must be a list of strings.")
        
        if category_output_mode not in {'one_hot', 'multi_hot'}:
            raise ValueError("category_output_mode must be 'one_hot' or 'multi_hot'")

        if not isinstance(is_normalization, bool):
            raise TypeError("is_normalization must be a boolean value.")
        
        self.categorical_column_names = categorical_column_names
        self.numerical_column_names = numerical_column_names
        self.category_output_mode = category_output_mode
        self.is_normalization = is_normalization
        
    def get_config(self):
        # Return the configuration parameters as a dictionary
        config = {
            "categorical_column_names": self.categorical_column_names,
            "numerical_column_names": self.numerical_column_names,
            "category_output_mode": self.category_output_mode,
            "is_normalization": self.is_normalization
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            config["categorical_column_names"],
            config["numerical_column_names"],
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
        reduction_layer: Method for dimensionality reduction ('flatten', 'average').
    """
    def __init__(
        self, 
        num_att: int = 16, 
        r: float = 3.0, 
        clf_num_layers: int = 1, 
        clf_hidden_units: List[int] = [64], 
        reduction_layer: str = 'flatten'
    ):
        if not isinstance(num_att, int) or num_att <= 0:
            raise ValueError("num_att must be a positive integer.")

        if not isinstance(r, (float, int)) or r < 1:
            raise ValueError("r must be a float or integer greater than or equal to 1.")

        if not isinstance(clf_num_layers, int) or clf_num_layers < 1:
            raise ValueError("clf_num_layers must be an integer greater than or equal to 1.")
            
        if reduction_layer not in {'flatten', 'average'}:
            raise ValueError("reduction_layer must be 'flatten' or 'average'")

        if not isinstance(clf_hidden_units, list) or not all(isinstance(unit, int) and unit > 0 for unit in clf_hidden_units):
            raise TypeError("clf_hidden_units must be a list of positive integers.")
            
        if len(clf_hidden_units) != clf_num_layers:
            raise ValueError(
                f"clf_hidden_units must have exactly {clf_num_layers} elements. "
                f"Got {len(clf_hidden_units)} elements instead."
            )
            
        self.num_att = num_att
        self.r = r
        #self.ife_num_layers = 1
        self.clf_num_layers = clf_num_layers
        self.clf_hidden_units = clf_hidden_units
        self.reduction_layer = reduction_layer

    def get_config(self):
        # Return the configuration parameters as a dictionary
        config = {
            "num_att": self.num_att,
            "r": self.r,
            #"ife_num_layers": self.ife_num_layers,
            "clf_num_layers": self.clf_num_layers,
            "clf_hidden_units": self.clf_hidden_units,
            "reduction_layer": self.reduction_layer
        }
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(
            config["num_att"],
            config["r"],
            config["clf_num_layers"],
            config["clf_hidden_units"],
            config["reduction_layer"]
        )