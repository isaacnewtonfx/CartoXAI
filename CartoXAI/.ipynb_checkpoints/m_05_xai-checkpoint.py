# Core python
import time
import pickle

# Data
import pandas as pd
import numpy as np

# Explainable AI
import shap
import fasttreeshap # A faster implementation of shap
shap.initjs()


def create_shap_explainer(model, use_fasttreeshap= True, out_explainer_file=None):
    """
    A function to create shap explainer on a given trained tree-based model.

    Parameters
    ----------

    model: sklearn model pipeline
        The trained machine learning model.
    use_fasttreeshap: bool
        Indicates whether to use the faster version of shap. NB. The standard shap is very slow when calculating interpretability on large data (default: True)
    data_features: pandas dataframe
        The entire features dataset. This is all features except the target feature.
    out_explainer_file: str
        The file path to save the explainer. (default: None)
        
    Returns
    -------

    An explainer object 
    
    """
    
    if use_fasttreeshap:
        explainer = fasttreeshap.TreeExplainer(model[1], algorithm="auto", n_jobs= -1)
    else:
        explainer = shap.Explainer(model[1])
        
    if out_explainer_file is not None:
        with open(out_explainer_file, 'wb') as f:
            pickle.dump(explainer, f)
    
    return explainer



def calc_local_interpretability(explainer, X_dataset):
    """
    A function to calculate shap values and the local predominant predictor on a given dataset.

    Parameters
    ----------

    explainer: shap or fasttreeshap explainer object.
        The explainer used to calculate shap values on the supplied X_dataset.
    X_dataset: pandas dataframe.
        The test or any unseen dataset containing all features in the model. The target variable (y) should not be part of this dataset.
        
    Returns
    -------

    A list containing raw shap values, transformed shap values and local predominant predictor in line with the rows
    
    """
    
    ###----- Calculate SHAP values
    
    if type(explainer) is fasttreeshap.explainers._tree.Tree:
        
        # Faster algorithm
        shap_values_npy = explainer(X_dataset).values
        
    else:
        
        # Slower algorithm
        shap_values_npy = explainer.shap_values(X_dataset)
        
    # Creating a dataframe for shap values
    column_names_shap = [f + "_shap" for f in X_dataset.columns]
    shap_values_df    = pd.DataFrame(shap_values_npy, columns = column_names_shap)
    
        
    
    ###----- Transform SHAP values into proportional shares (percentages)
    
    # calc absolute values
    shap_values_abs = np.abs(shap_values_npy)

    # define a mask to account for negative values
    mask = np.ones(shap_values_npy.shape, dtype=int)
    mask[shap_values_npy < 0] = -1

    # calc percentages
    shap_row_sums           = shap_values_abs.sum(axis=1)[:,None]
    shap_values_perc        = np.round( ((shap_values_abs/shap_row_sums) * 100), 1)
    shap_values_perc_signed = shap_values_perc * mask # restore the sign

    # Creating a dataframe for shap percentage values
    column_names_shap_perc = [f + "_shap_perc" for f in X_dataset.columns]
    shap_values_perc_df    = pd.DataFrame(shap_values_perc_signed, columns = column_names_shap_perc)
    

    
    ###----- Calculate Local Predominant Predictor
    
    # Get the index of the maximum value for each row. Returns a numpy index_array
    idx_local_vimp = np.argmax(np.abs(shap_values_df), axis=1)
    
    # Use the index to get the column names associated with the index. Returns a list
    local_vimp = [X_dataset.columns[idx] for idx in idx_local_vimp]
        
    return [shap_values_df, shap_values_perc_df, local_vimp]
    