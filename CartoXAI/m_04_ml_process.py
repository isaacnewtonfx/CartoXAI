import time,math

# Data
import numpy as np
import pandas as pd
from geocube.api.core import make_geocube

# Machine Learning
from sklearn.model_selection import cross_validate
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def train_machine_learning_model(model, X_train, y_train, X_test, y_test):
    """
    A function to train the machine learning model on the train set and then test the performance on the test set.

    Parameters
    ----------

    model: sklearn model pipeline
        The model pipeline that which is an output from the define_model_pipeline() function in the file m_02_preprocess.py
    X_train: pandas dataframe
        The train features dataset which is an output from the train_test_splitting() function in the file m_02_preprocess.py
    y_train: pandas Series
        The target data corresponding to the train feature dataset which is an output from the train_test_splitting() function in the file m_02_preprocess.py
    X_test: pandas dataframe
        The test features dataset which is an output from the train_test_splitting() function in the file m_02_preprocess.py
    y_test: pandas Series
        The target data corresponding to the test feature dataset which is an output from the train_test_splitting() function in the file m_02_preprocess.py
    
    Returns
    -------

    A trained sklearn model pipeline which would perform scaling as preprocessing before prediction.   
    
    """


    # Train the model on train data
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = round(time.time() - start, 2)

    # Model accuracy using R-Squared -> ((y_true - y_pred)**2).sum()
    train_score = round(model.score(X_train, y_train), 2)
    test_score  = round(model.score(X_test, y_test), 2)

    print("Train Score:", train_score, " Test Score:", test_score, " Training Elapsed Time:", elapsed_time, "sec")

    return model


def perform_cross_validation(model, data_features, data_target, n_splits=10, test_size= 0.2):
    """
    A function to perform cross validation on a trained sklearn model to assess the model generalization performance.

    Parameters
    ----------

    model: sklearn model pipeline
        The trained machine learning model.
    data_features: pandas dataframe
        The entire features dataset. This is all features except the target feature.
    data_target: pandas Series
        The entire target dataset to be used in cross-validation.
    n_splits: int
        The number of splits to consider for cross-validation. (default: 10)
    test_size: float
        The size of the test set to use for cross-validation. (default: 0.2)

    Returns
    -------

    The cross-validation results as a pandas dataframe
    
    """

    cv = ShuffleSplit(n_splits, test_size= test_size, random_state=0)
    cv_results = cross_validate(model, data_features, data_target, cv=cv, scoring="r2")
    cv_results = pd.DataFrame(cv_results)

    return cv_results


def assess_max_depth_cv_curve(model, max_depth, data_features, data_target, param_name="randomforestregressor__max_depth", cv_n_splits=10, cv_test_size=0.3, cv_random_state=0, out_plot_file=None):
    """
    A function to assess overfitting and underfitting by plotting the cross-validation curve of the max_depth as a hyperparameter.

    Parameters
    ----------

    model: sklearn model pipeline
        The trained machine learning model.
    max_depth: List
        A list of integers for the max_depth hyperparameter to consider for training the machine learning model.
    data_features: pandas dataframe
        The entire features dataset. This is all features except the target feature.
    data_target: pandas Series
        The entire target dataset to be used in cross-validation.
    param_name: str
        The model prefixed name of the parameter being assessed as used internally in the scikit-learn model.   
    cv_n_splits: int
        The number of splits to consider for cross-validation. (default: 10)
    cv_test_size: float
        The size of the test set to use for cross-validation. (default: 0.3)
    cv_random_state: float
        This is a seed value for keeping the random state constant. For reproducibility.
    out_plot_file: str
        The file path to save the cross-validation curve plot. (default: None)
        
    Returns
    -------

    None 
    
    """

    cv = ShuffleSplit(n_splits= cv_n_splits, test_size= cv_test_size, random_state= cv_random_state)  
    train_scores, test_scores = validation_curve(
        model, data_features, data_target, param_name=param_name, param_range=max_depth,
        cv=cv, scoring="r2", n_jobs=2)

    sns.set_context("paper", rc={"font.size":12,"axes.titlesize":10,"axes.labelsize":8})  

    plt.figure() # prevent plotting on previous figure
    plt.plot(max_depth, train_scores.mean(axis=1), label="Training score")
    plt.plot(max_depth, test_scores.mean(axis=1), label="Testing score")
    plt.legend()

    plt.xlabel("Maximum depth of decision tree")
    plt.ylabel("R-Squared")
    _ = plt.title("Max Depth Validation curve for Random Forest")

    if out_plot_file is not None:
        plt.savefig(out_plot_file, dpi=300, bbox_inches= "tight")


def assess_feature_importance(model, X_train, out_plot_file):
    """
    A function to assess the importance of the features used to train the machine learning model.

    Parameters
    ----------

    model: sklearn model pipeline
        The trained machine learning model pipeline.
    X_train: pandas dataframe
        The train features dataset which is an output from the train_test_splitting() function in the file m_02_preprocess.py
    out_plot_file: str
        The file path to save the feature importance plot. (default: None)

    Returns
    -------

    None

    """

    importances = model.steps[1][1].feature_importances_
    indices = np.argsort(importances)

    plt.figure() # prevent plotting on previous figure
    fig, ax = plt.subplots()
    fig.set_size_inches(10,20)
    ax.barh(range(len(importances)), importances[indices])
    ax.set_yticks(range(len(importances)))
    _ = ax.set_yticklabels(np.array(X_train.columns)[indices])
    ax.set_title("Feature importances by Mean Decrease in Impurity (MDI)")
    ax.set_xlabel("Mean decrease in impurity")

    if out_plot_file is not None:
        plt.savefig(out_plot_file, dpi=300, bbox_inches= "tight")


def assess_accuracy_metrics(model, obs, relevant_features, target, out_plot_file=None):
    """
    A function to assess the accuracy metrics of the model

    Parameters
    ----------

    model: sklearn model pipeline
        The trained machine learning model pipeline.
    obs: geopandas geodataframe
        The cleaned observation dataset.
    relevant_features: List
        The names of the relevant features.
    target: str
        The name of the target feature.
    out_plot_file: str
        The file path to save the correlation plot. (default: None)
    
    Returns
    -------

    A list containing the correlation coefficient, rmse, mae, r2
    
    """

    obs['pred'] = model.predict(obs[relevant_features])

    text_ycoord = math.floor(max(obs['pred'])*10)/10

    corr = round(obs[target].corr(obs['pred']), 2)
    print("Correlation between observed and predicted is: ", corr)

    rmse = round( math.sqrt(mean_squared_error(obs[target], obs['pred'])), 2)
    print("RMSE between observed and predicted is: ", rmse)

    mae = round(mean_absolute_error(obs[target], obs['pred']), 2)
    print("MAE between observed and predicted is: ", mae)

    r2 = round(r2_score(obs[target], obs['pred']), 2)
    print("Explained Variance between observed and predicted is: ", r2)

    plt.figure() # prevent plotting on previous figure
    sns.regplot(x=target, y="pred", data=obs).set(title='Observed Total Nitrogen vs Predicted')
    plt.text(0, text_ycoord, "corr.: " + str(corr), horizontalalignment='left', size='medium', color='black', weight='semibold')

    if out_plot_file is not None:
        plt.savefig(out_plot_file, dpi=300, bbox_inches= "tight")

    return [corr, rmse, mae, r2]


def predict_trend_on_grid(model, model_qrf, relevant_features, grid, quantiles=[0.05, 0.5, 0.95], out_pred_gpkg_file= None, out_pred_trend_rast_file= None, out_pred_median_rast_file= None, out_pred_uncertainty_rast_file= None):
    """
    A function to predict the target variable on a grid including the uncertainty.

    Parmeters
    ---------

    model: sklearn model pipeline
        The trained machine learning model pipeline.
    model_qrf: sklearn-based quantile random forest model pipeline
        The trained quantile random forest machine learning model pipeline.
    relevant_features: List
        The names of the relevant features.
    grid: geopandas geodataframe
        The prediction grid dataset that has all the relevant features as columns..
    quantiles: List
        The additional list of quantiles to predict.
    out_pred_gpkg_file: str
        The file path to save the output grid as geopackage. (default: None)
    out_pred_trend_rast_file: str
        The file path to save a raster version of the mean predictions. (default: None)
    out_pred_median_rast_file: str
        The file path to save a raster version of the median predictions. (default: None)
    out_pred_uncertainty_rast_file: str
        The file path to save a raster version of the prediction uncertainty. (default: None)
    
    
    Returns
    -------

    The grid with a columns pred_trend,pred_q5,pred_q50,pred_q95,uncertainty indicating the model predictions.

    """


    # Remove NA rows from grid and store in a new variable
    grid_pred = grid.dropna(axis=0).copy()
    grid_pred['pred_trend'] = model.predict(grid_pred[relevant_features])
    
    quantiles_pred = model_qrf.predict(grid_pred[relevant_features], quantiles= quantiles)
    grid_pred['pred_q5']  = quantiles_pred[:,0]
    grid_pred['pred_q50'] = quantiles_pred[:,1]
    grid_pred['pred_q95'] = quantiles_pred[:,2]
    
    # Uncertainty - IQR / median. Using median because it is robust to outliers
    # The smaller the value, the smaller the uncertainty and the higher the value, the higher the uncertainty.
    grid_pred['uncertainty'] = (quantiles_pred[:,2] - quantiles_pred[:,0]) / quantiles_pred[:,1] # (q95 - q5)/median
    
    # Merge predicted results with the original grid using the primary key
    grid = pd.merge(grid, grid_pred[["pk","pred_trend","pred_q5","pred_q50","pred_q95","uncertainty"]], how= "left", on= "pk")
       
    grid['pred_trend'].describe() # mean
    grid['pred_q50'].describe() # median
    grid['uncertainty'].describe() # uncertainty
    

    # Save prediction grid - vector
    if out_pred_gpkg_file is not None:
        # grid[["pk","pred_trend", "geometry"]].to_file(out_pred_grid_gpkg_file, driver="GPKG")
        grid.to_file(out_pred_gpkg_file, driver="GPKG")

    # Save prediction grid - raster
    if out_pred_trend_rast_file is not None:
        out_pred_trend = make_geocube(vector_data=grid, measurements=["pred_trend"], resolution=(-1000, 1000)) #for most crs negative comes first in resolution
        out_pred_trend["pred_trend"].rio.to_raster(out_pred_trend_rast_file)
     
    if out_pred_median_rast_file is not None:    
        out_pred_median = make_geocube(vector_data=grid, measurements=["pred_q50"], resolution=(-1000, 1000)) #for most crs negative comes first in resolution
        out_pred_median["pred_q50"].rio.to_raster(out_pred_median_rast_file)
    
    if out_pred_uncertainty_rast_file is not None:
        out_pred_uncertainty = make_geocube(vector_data=grid, measurements=["uncertainty"], resolution=(-1000, 1000)) #for most crs negative comes first in resolution
        out_pred_uncertainty["uncertainty"].rio.to_raster(out_pred_uncertainty_rast_file)

    return grid