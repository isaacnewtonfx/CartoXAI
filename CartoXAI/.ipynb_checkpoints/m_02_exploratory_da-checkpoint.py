# Data Handling
import numpy as np
import pandas as pd
import geopandas as gpd
from pyogrio import read_dataframe

# Core python
import os,time
import pickle as pkl

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(in_file_obs,in_file_grid, obs_layer_name=None, grid_layer_name=None):
    """
    Function to load the observation and grid data both in vector formats.

    Parameters
    ---------

    in_file_obs: str
        The full or relative file path to the input observations data that has been merged with covariates, in geopackage or the python pickle file format
    in_file_grid: str
        The full or relative file path to the input grid data that has been merged with covariates, in geopackage or the python pickle file format
    obs_layer_name: str
        The name of the layer to load if the observation file is a geopackage file (default: None)
    grid_layer_name: str
        The name of the layer to load if the grid file is a geopackage file (default: None)

    Returns
    -------
    A tuple containing the loaded observation and grid data as geopandas data frames (obs, grid)
    
    """

    # Observation data - Total Nitrogen points merged with covariates
    obs = read_dataframe(in_file_obs, layer= obs_layer_name)

    # Prediction Grid data - 1km spaced points merged with covariates
    grid1km = read_dataframe(in_file_grid, layer= grid_layer_name)

    print("Loaded data from raw files")


    return (obs, grid1km)



def eda(obs, features, target="nitkjd_value_avg", key_column="pk", remove_na_rows = True, find_highly_corr_features = True, corr_threshold = 0.9, out_plot_hist_target_file=None, out_boxplot_target_file= None, out_plot_heatmap_file=None):
    """
    Function to perform exploratory data analysis on the observed data including reprojection of coordinate system, removing NAs and identifying highly correlated features based on a correlation threshold.

    Parameters
    ----------

    obs: geopandas geodataframe
        The observed dataset.
    features: List
        List of all feature names.
    target: str
        The column name of the target variable in the observed dataset. (default: "nitkjd_value_avg")
    key_column: str
        The name of the primary key column on the observed dataset. (default: "pk")
    remove_na_rows: bool
        Indicates whether to remove NA rows from the observed or not. (default: True)
    find_highly_corr_features: bool
        Indicates whether to find highly correlated features or not. (default: True)
    corr_threshold: float
        The threshold beyond which a feature would be detected as highly correlated. (default: 0.9)
    out_plot_hist_target_file: str
        The file path to save the histogram of the target variable. (default: None)
    out_boxplot_target_file: str
        The file path to save the boxplot of the target variable. (default: None)
    out_plot_heatmap_file: str
        The file path to save the correlation heatmap plot. (default: None)
    
    Returns
    -------

    A tuple containing the cleaned observed data as a geopandas geodataframe, list of highly correlated feature names, and list of relevant feature names.

    """
    
    # Check number of rows and columns
    print(f"The dataset contains {obs.shape[0]} samples and "
        f"{obs.shape[1]} columns")

    
    # Make pk an integer
    if obs[key_column].dtype != int:
        print(f"Converting the key column {key_column} to integer data type")
        obs[key_column] = obs[key_column].astype("int")

    # Remove NA rows from observation data
    if remove_na_rows:
        count_na_rows = sum(obs.isna().any(axis=1))
        obs = obs.dropna(axis=0)
        print(f"{count_na_rows} rows containing NA have been removed, remaining {obs.shape[0]} clean rows")
        
    ### Assess outliers 
    # Visualize histogram of target variable
    plt.figure(figsize=(15, 7))
    sns.histplot(data=obs, x=target).set_title('Histogram of target variable')
    
    if out_plot_hist_target_file is not None:
        plt.savefig(out_plot_hist_target_file, dpi=300, bbox_inches= "tight")
        
        
    # Visualize outliers in target variable using boxplot
    plt.figure(figsize=(15, 7))
    sns.boxplot(data=obs, x=target).set_title('Boxplot of the target variable')
    
    if out_boxplot_target_file is not None:
        plt.savefig(out_boxplot_target_file, dpi=300, bbox_inches= "tight")
        

    ### Assess Multicollinearity
    # Remove highly correlated features if any
    if find_highly_corr_features:

        # Create a correlation matrix
        correlation_matrix = obs[features].corr()
        highly_correlated_features = []

        # Get highly correlated features
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > corr_threshold:
                    colname = correlation_matrix.columns[i]
                    highly_correlated_features.append(colname)

        # Subset data with only relevant features
        highly_correlated_features = list(set(highly_correlated_features))
        relevant_features = [f for f in features if f not in highly_correlated_features]

        print(f"Found {len(highly_correlated_features)} highly correlated features")

        # Visualize correlation between features
        sns.set(style="darkgrid") 
        plt.figure(figsize=(40, 30))

        mask = np.triu(np.ones_like(obs[relevant_features].corr(), dtype=bool))
        heatmap = sns.heatmap(obs[relevant_features].corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
        heatmap.set_title('Correlation Heatmap ({} relevant features)'.format(len(relevant_features)), fontdict={'fontsize':35}, pad=16)
        plt.legend(fontsize="20")

        if out_plot_heatmap_file is not None:
            plt.savefig(out_plot_heatmap_file, dpi=300, bbox_inches= "tight")


    return (obs, highly_correlated_features, relevant_features)