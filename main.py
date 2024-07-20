#------------------------- INTRODUCTION
"""
This file is an implementation of the workflow designed in the thesis, addressing research question 3.1 of objective 3. 
Refer to the thesis for more information on other objectives and research questions.


Title: 
- Improving the understanding of Machine Learning predictions through maps

The Problem: 
- A non-interpretable ML prediction leads to 
    - poor trust and transparency (Gartner, 2023; Robinson et al., 2023)
    - Misleading information and ethical issues (Kang et al., 2023).
    - Pitfalls in making sensitive decisions (Adadi & Berrada, 2018).
- There is literature gap on how to communicate ML model local interpretability using maps.

Aim:
- To explore the potential of cartography to enhance the understanding of a ML model

Research Objectives and Questions:
1. To review existing Explainable Machine Learning models for understanding ML predictions.
RQ1.1: What Local Interpretability models exist?
RQ1.2: Which Local Interpretability models are effective?

2. To propose a Local Interpretability workflow integrated with maps
RQ2.1: How can cartography support the workflow for understanding the Local Interpretability of ML predictions?
RQ2.2: How can the local predominant predictor be visualized?

3. To implement the proposed workflow using a case study.
RQ3.1: What are the technical steps and challenges in implementing the proposed workflow?
RQ3.2: How do target users perceive the prediction and local interpretability?

Potential users:
- Soil Scientists, Fertilizer Researchers, Agronomists, Policy Makers

Stuy area:
- Burkina Faso

Inputs folder:
- Covariates extracted from Google Earth Engine for observation and 1km grid dataset: data/GEE Covariates
- ISRIC WoSIS Soil Total Nitrogen observation spatial data: data/isric_total_N.gpkg

Outputs folder:
- data:  saved/data
- model: saved/model
- plots: saved/plots

NB: Opened figures during execution must be closed for the execution to proceed.

Author:  By Isaac Newton Kissiedu, International MSc. Cartography Student, TU Dresden, Germany
Contact: isaacnewtonfx@gmail.com, isaac_newton.kissiedu@mailbox.tu-dresden.de, iski467f@tu-dresden.de
Personal Website: https://isaacnewtonfx.github.io

"""

#------------------------- IMPORT PYTHON PACKAGES

# Core python
import os
import pickle
import math

# Data
import rioxarray as rxr
from pyogrio import read_dataframe
import pandas as pd
import numpy as np

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.base import clone
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Explainable AI
import shap
import fasttreeshap
shap.initjs()

# Program
from CartoXAI import m_01_data_prep as data_prep
from CartoXAI import m_02_exploratory_da as explore
from CartoXAI import m_03_preprocess as preprocess
from CartoXAI import m_04_ml_process as ml_process
from CartoXAI import m_05_xai as xai


#------------------------- Parameters

# Data paths

in_file_obs = 'saved/data/obs_merged_covs_utmz30n.gpkg'
in_file_grid = 'saved/data/grid1km_merged_covs_utmz30n.gpkg'

# Select primary key column
key_column = "pk"

# Select target feature
target = "nitkjd_value_avg"

# Whether to make new predictions or use previously saved predictions
make_new_predictions = True



#------------------------- 1 Data Preparation

print("\n*** Progress 1/43: Data Preparation ***", end="\n\n")

# Total Nitrogen Observations
obs = data_prep.prepare_obs_data(in_gee_csv_dir= "data/GEE Covariates/obs",
                                in_file_obs= "data/isric_total_N.gpkg",
                                target_col= "nitkjd_value_avg",
                                pk_col= "pk",
                                out_file_gpkg= "saved/data/obs_merged_covs_utmz30n.gpkg",
                                out_gpkg_lyr_name= "total_N", target_epsg=32630)

# 1km Grid
grid1km = data_prep.prepare_grid_data(in_gee_csv_dir= "data/GEE Covariates/grid",
                                  in_file_grid= "data/GEE Covariates/grid/shp_to_gee/grid.shp",
                                  out_file_gpkg= "saved/data/grid1km_merged_covs_utmz30n.gpkg",
                                  out_gpkg_lyr_name= "grid", target_epsg=32630)



#------------------------- 2 Preview data

print("\n*** Progress 2/43: Preview data ***", end="\n\n")

print("Showing observation data with extracted covariates", end="\n\n")
print(obs.head())

print("\nShowing grid data with extracted covariates", end="\n\n")
print(grid1km.head())

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title('Obs')
ax2.set_title('Grid')
# obs.plot(ax=ax1, markersize=0.1)
# grid1km.plot(ax=ax2, markersize=0.1)



#------------------------- 3 Exploratory Data Analysis - Load Data

print("\n*** Progress 3/43: Exploratory Data Analysis - Load Data ***", end="\n\n")

# Load datasets
obs, grid1km = explore.load_data(in_file_obs, in_file_grid, obs_layer_name= "total_N",grid_layer_name= "grid")

# Review columns and select candidate features
print(obs.columns)

# Select ML candidate features
features = [f for f in obs.columns if f not in ["pk","fid","geometry","nitkjd_value_avg","nitkjd_date"]]



#------------------------- 4 Exploratory Data Analysis - Run EDA

print("\n*** Progress 4/43: Exploratory Data Analysis - Run EDA ***", end="\n\n")

obs, highly_correlated_features, relevant_features = explore.eda(obs, features, target, key_column, remove_na_rows= True, 
                                                                 find_highly_corr_features= True, corr_threshold= 0.9, 
                                                                 out_plot_hist_target_file = "saved/plots/target_hist_plot.png",
                                                                 out_boxplot_target_file = "saved/plots/target_box_plot.png",
                                                                 out_plot_heatmap_file= "saved/plots/features_corr_plot.png")

with open("saved/data/relevant_features.pickle", 'wb') as f:
    pickle.dump(relevant_features, f)



#------------------------- 5 Remove outliers from data

print("\n*** Progress 5/43: Remove outliers from data ***", end="\n\n")

Q1 = obs[target].quantile(0.25)
Q3 = obs[target].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 
filter = (obs[target] >= Q1 - 1.5 * IQR) & (obs[target] <= Q3 + 1.5 *IQR)
count_normal_data = sum(filter)
count_outlier_data = sum(filter==False)
perc_of_outliers = round( (count_outlier_data / len(filter))*100, 2)

obs = obs[filter]
# obs.shape[0]

print("Number of normal data:", count_normal_data)
print("Number of outlier data:", count_outlier_data)
print("Percentage of outliers (%):", perc_of_outliers)



#------------------------- 6 Review the distribution of the target data after removing outliers

print("\n*** Progress 6/43: Review the distribution of the target data after removing outliers ***", end="\n\n")

# Visualize histogram of target variable
plt.figure(figsize=(15, 7))
sns.histplot(data=obs, x=target).set_title('Histogram of target variable - large outliers removed')
plt.savefig("saved/plots/target_hist_plot2.png", dpi=300, bbox_inches= "tight")

# Visualize outliers in target variable using boxplot
plt.figure(figsize=(15, 7))
sns.boxplot(data=obs, x=target).set_title('Boxplot of the target variable - large outliers removed')
plt.savefig("saved/plots/target_box_plot2.png", dpi=300, bbox_inches= "tight")



#------------------------- 7 Preprocessing - Subset dataset using relevant features

print("\n*** Progress 7/43: Preprocessing - Subset dataset using relevant features ***", end="\n\n")

all_features  = relevant_features + [target]
data_features, data_target = preprocess.subset_observation_data(obs, relevant_features, target)

print("Showing head records of the dataset", end="\n\n")
print(data_features.head())



#------------------------- 8 Preprocessing - Train-Test Splitting

print("\n*** Progress 8/43: Preprocessing - Train-Test Splitting ***", end="\n\n")

X_train, X_test, y_train, y_test = preprocess.train_test_splitting(data_features, data_target, test_size= 0.2)

X_train.shape[0] / data_features.shape[0] * 100

with open("saved/data/X_test.pickle", 'wb') as f:
    pickle.dump(X_test, f)
    
with open("saved/data/y_test.pickle", 'wb') as f:
    pickle.dump(y_test, f)



#------------------------- 9 Preprocessing - Define a Machine Learning model pipeline with data standardization

print("\n*** Progress 9/43: Preprocessing - Define a Machine Learning model pipeline with data standardization ***", end="\n\n")

model = preprocess.define_model_pipeline(preprocessor= StandardScaler(), processor= RandomForestRegressor())

# Show the model steps
print("Showing model pipeline steps", end="\n\n")
print(model.named_steps)

# Show the model hyperparameters
print("\nShowing parameters in model pipeline", end="\n\n")
print(model.get_params())

print("\nShowing the model pipeline", end="\n\n")
print(model)



#------------------------- 10 Processing - Train Random Forest (just a single train/test split)

print("\n*** Progress 10/43: Processing - Train Random Forest (just a single train/test split) ***", end="\n\n")

model = ml_process.train_machine_learning_model(model, X_train, y_train, X_test, y_test)


#------------------------- 11 Hyperparameter Tuning

print("\n*** Progress 11/43: Hyperparameter Tuning ***", end="\n\n")

# Because of the pipeline, we have to indicate the name of the object where the param is to be applied as a prefix__
# rf_grid = {"randomforestregressor__n_estimators": np.arange(10, 100, 10),
#            "randomforestregressor__max_depth": [1, 3, 5, 10, 15],
#            "randomforestregressor__min_samples_split": np.arange(2, 20, 2),
#            "randomforestregressor__min_samples_leaf": np.arange(1, 20, 2),
#            "randomforestregressor__max_samples": [1000]}

rf_grid = {"randomforestregressor__n_estimators": randint(10, 100),
           "randomforestregressor__max_depth": randint(1, 20),
           "randomforestregressor__min_samples_split": randint(2, 20),
           "randomforestregressor__min_samples_leaf": randint(1, 20),
           "randomforestregressor__max_samples": [1000]}

# Instantiate RandomizedSearchCV model
random_search = RandomizedSearchCV(model,
                              param_distributions=rf_grid,
                              scoring="r2",
                              n_iter=500,
                              cv=5,
                              verbose=True)
# fit
random_search.fit(X_train, y_train)



#------------------------- 12 Rename Hyperparameter CV Results columns

print("\n*** Progress 12/43: Rename Hyperparameter CV Results columns ***", end="\n\n")

results = random_search.cv_results_
df = pd.DataFrame(results)
df['iteration'] = range(1, len(df) + 1)

df = df.rename(columns={"param_randomforestregressor__max_depth": "param_max_depth",
                        "param_randomforestregressor__n_estimators": "param_n_estimators",
                        "param_randomforestregressor__min_samples_split": "param_min_samples_split",
                        "param_randomforestregressor__min_samples_leaf": "param_min_samples_leaf",
                        "param_randomforestregressor__max_samples": "param_max_samples"})

print("Showing the head records of the hyperparameter search results", end="\n\n")
print(df.head())

print("best estimator: ", random_search.best_estimator_)
print("best estimator cv score:" , round(random_search.best_score_,2))



#------------------------- 13 Plot Hyperparameter Results

print("\n*** Progress 13/43: Plot Hyperparameter Results ***", end="\n\n")

plt.figure(figsize=(10, 6))
plt.plot(df['iteration'], df['mean_test_score'], marker='o')
plt.xlabel('Iteration')
plt.ylabel('Mean Test Score')
plt.title('Iteration vs. Mean Test Score')
plt.grid(True)
plt.show()



#------------------------- 14 Parallel Coordinate Plot of all parameters

print("\n*** Progress 14/43: Parallel Coordinate Plot of all parameters ***", end="\n\n")

# Convert hyperparameters to string for better visualization
df['param_max_depth'] = df['param_max_depth'].astype(int)
df['param_n_estimators'] = df['param_n_estimators'].astype(int)
df['param_min_samples_split'] = df['param_min_samples_split'].astype(int)
df['param_min_samples_leaf'] = df['param_min_samples_leaf'].astype(int)
df['param_max_samples'] = df['param_max_samples'].astype(int)


fig = px.parallel_coordinates(
    df,
    dimensions=['param_n_estimators','param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf','param_max_samples', 'mean_test_score'],
    color='mean_test_score',
    color_continuous_scale=px.colors.sequential.Viridis
)
# fig.update_layout(title='Parallel Coordinates Plot of Hyperparameters and Mean Test Score')
fig.show()

fig.write_image("saved/plots/hyperparams_vs_mean_test_score.png", width=1500, scale=1)




#------------------------- 15 Best Model Selection

print("\n*** Progress 15/43: Best Model Selection ***", end="\n\n")

model = random_search.best_estimator_



#------------------------- 16 Processing - Validation curve of max_depth to assess Overfitting and Underfitting
# Checking if there is the need to tune the max_depth hyperparameter which is set to None by default.

print("\n*** Progress 16/43: Processing - Validation curve of max_depth to assess Overfitting and Underfitting ***", end="\n\n")

max_depth = [1, 5, 10, 15, 20, 25]
ml_process.assess_max_depth_cv_curve(model, max_depth, data_features, data_target, "randomforestregressor__max_depth",                                     
                                  cv_n_splits= 10, cv_test_size= 0.2, cv_random_state= 0, 
                                  out_plot_file="saved/plots/RF_validation_curve.png")



#------------------------- 17 Processing - Retrain the selected model on entire training set so it knows well about the entire training data

print("\n*** Progress 17/43: Processing - Retrain the selected model on entire training set so it knows well about the entire training data ***", end="\n\n")

# Fit on train data
model = ml_process.train_machine_learning_model(model, X_train, y_train, X_test, y_test)



#------------------------- 18 Processing - Final Model Evaluation on Unseen Test Data

print("\n*** Progress 18/43: Processing - Final Model Evaluation on Unseen Test Data ***", end="\n\n")

print("Showing model score on Test Data", end="\n\n")
print(model.score(X_test, y_test))



#------------------------- 19 Train a Quantile Random Forest Model based on best hyperparameters to be used for uncertainty estimation

print("\n*** Progress 19/43: Train a Quantile Random Forest Model based on best hyperparameters to be used for uncertainty estimation ***", end="\n\n")

print("Showing the best hyperparameters", end="\n\n")
random_search.best_params_

model_qrf = preprocess.define_model_pipeline(preprocessor= StandardScaler(), 
                                             processor= RandomForestQuantileRegressor(max_depth=15,max_samples=1000,min_samples_leaf=3,min_samples_split=10,n_estimators=60))

# Fit on train data
model_qrf = ml_process.train_machine_learning_model(model_qrf, X_train, y_train, X_test, y_test)


#------------------------- 20 Save Models

print("\n*** Progress 20/43: Save Models ***", end="\n\n")

with open("saved/model/ml_model.pickle", 'wb') as f:
    pickle.dump(model, f)
    
with open("saved/model/ml_model_qrf.pickle", 'wb') as f:
    pickle.dump(model_qrf, f)  



#------------------------- 21 Processing - Random Forest Feature Importance
# Gini Importance or Mean Decrease in Impurity (MDI) calculates each feature importance as the sum over the 
# number of splits (across all tress) that include the feature, proportionally to the number of samples it splits

print("\n*** Progress 21/43: Processing - Random Forest Feature Importance ***", end="\n\n")

ml_process.assess_feature_importance(model,X_train, out_plot_file="saved/plots/RF_feature_importance.png")



#------------------------- 22 Processing - Assess Model Accuracy Metrics

print("\n*** Progress 22/43: Processing - Assess Model Accuracy Metrics ***", end="\n\n")

[corr, rmse, mae, r2] = ml_process.assess_accuracy_metrics(model, obs, relevant_features, target, out_plot_file= "saved/plots/obs_vs_pred.png")



#------------------------- 23 Processing - Random Forest Prediction on Grid1km

print("\n*** Progress 23/43: Processing - Random Forest Prediction on Grid1km. It takes at least 2h 30min on Corei7-11800H, 2.3GHz, 16Gb Ram ***", end="\n\n")

if os.path.exists("saved/data/grid1km_pred.gpkg") and make_new_predictions == False:
     grid1km = read_dataframe("saved/data/grid1km_pred.gpkg", layer= "grid1km_pred")
else:
     grid1km = ml_process.predict_trend_on_grid(model,model_qrf, relevant_features, grid1km, quantiles= [0.05, 0.5, 0.95],
                                out_pred_gpkg_file= "saved/data/grid1km_pred.gpkg", 
                                out_pred_trend_rast_file= "saved/data/pred1km_n_g_kg.tif",
                                out_pred_median_rast_file= "saved/data/pred1km_median_n_g_kg.tif",
                                out_pred_uncertainty_rast_file= "saved/data/pred1km_uncertainty_n_g_kg.tif")

print("Showing the head records of the Grid1km after predictions", end="\n\n")
print(grid1km.head())

# Time taken for server computer
# 2h 45min 15s

# Time taken for Lenovo laptop
# 2h 30min




#------------------------- 24 Plot ML Prediction mean,median,uncertainty

print("\n*** Progress 24/43: Plot ML Prediction mean ***", end="\n\n")

cmap = sns.color_palette("rocket_r", as_cmap=True)
ax = grid1km.plot(column= "pred_trend", legend= True, legend_kwds={"label": "Total Nitrogen (g/kg)"}, figsize= (10,7), cmap=cmap, markersize= 0.1)
ax.axis("off")
ax.set_title("Total Nitrogen (g/kg) ML Mean Prediction in Burkina Faso")
plt.savefig("saved/plots/TotalN_pred_mean.png", dpi=300, bbox_inches= "tight")



#------------------------- 25 Plot Quantiles q5,q50,q95

print("\n*** Progress 25/43: Plot Quantiles q5,q50,q95 ***", end="\n\n")

cmap = sns.color_palette("rocket_r", as_cmap=True)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 7))
for ax in axes.flat:
    ax.axis('off')

# vmin and vmax for using to same scale for plotting
vmin = min(grid1km['pred_q5'])
vmax = max(grid1km['pred_q95'])

grid1km.plot(column= "pred_q5",  legend= True, cmap= cmap, markersize= 0.1, ax=axes[0], vmin= vmin, vmax= vmax)
axes[0].set_title("Total Nitrogen (g/kg) ML Q5% Prediction")

grid1km.plot(column= "pred_q50", legend= True, cmap= cmap, markersize= 0.1, ax=axes[1], vmin= vmin, vmax= vmax)
axes[1].set_title("Total Nitrogen (g/kg) ML Q50% Prediction")

grid1km.plot(column= "pred_q95", legend= True, cmap= cmap, markersize= 0.1, ax=axes[2], vmin= vmin, vmax= vmax)
axes[2].set_title("Total Nitrogen (g/kg) ML Q95% Prediction")

plt.savefig("saved/plots/TotalN_pred_q5_q50_q95.png", dpi=300, bbox_inches= "tight")



#------------------------- 26 Uncertainty = (q95 - q5)/q50 OR IQR/Median

print("\n*** Progress 26/43: Uncertainty = (q95 - q5)/q50 OR IQR/Median ***", end="\n\n")

cmap = sns.color_palette("rocket_r", as_cmap=True)
ax = grid1km.plot(column= "uncertainty", legend= True, legend_kwds={"label": "Uncertainty"}, figsize= (10,7), cmap=cmap, markersize= 0.1)
ax.axis('off')
ax.set_title("Total Nitrogen ML Uncertainty Estimation in Burkina Faso")
plt.savefig("saved/plots/TotalN_pred_uncertainty.png", dpi=300, bbox_inches= "tight")

print("Showing descriptive statistics of the uncertainty", end="\n\n")
print(grid1km['uncertainty'].describe())



#------------------------- Explainable AI zone

print("\n*** BEGINNNING EXPLAINABLE AI ***", end="\n\n")


#------------------------- 27 Create an explainer on the machine learning model

print("\n*** Progress 27/43: Create an explainer on the machine learning model ***", end="\n\n")

explainer = xai.create_shap_explainer(model, use_fasttreeshap= True, out_explainer_file= None)



#------------------------- 28 Calculate Local Interpretability

print("\n*** Progress 28/43: Calculate Local Interpretability ***", end="\n\n")

# Scaling the grid1km features because the model was trained on scaled training features
scaler = model[0]

grid1km_subset = grid1km[grid1km['pred_trend'].isna() == False].copy()
grid1km_subset = grid1km_subset.reset_index(drop=True) # necessary to allow easy merging with the shap results

X_grid_scaled    = scaler.fit_transform(grid1km_subset[relevant_features])
X_grid_scaled_df = pd.DataFrame(X_grid_scaled, columns = relevant_features)

[shap_values_df, shap_values_perc_df, local_vimp] = xai.calc_local_interpretability(explainer, X_grid_scaled_df)



#------------------------- 29 Validate SHAP values

print("\n*** Progress 29/43: Validate SHAP values ***", end="\n\n")

print("y_grid_pred", np.array(grid1km_subset['pred_trend'][:5]))

print("shap_calculated", np.array(explainer.expected_value[0] + shap_values_df.sum(axis=1))[:5])

a = grid1km_subset['pred_trend']
b = explainer.expected_value[0] + shap_values_df.sum(axis=1)
c = round(np.corrcoef(a,b)[0,1], 2)
rmse = math.sqrt(mean_squared_error(a,b))

print ("corr: ", c, "rmse: ", rmse)




#------------------------- 30 Merge Local Interpretability results with the original dataset. Save this dataset !!!
# merging the subset

print("\n*** Progress 30/43: Merge Local Interpretability results with the original dataset ***", end="\n\n")

# First merge the results with the subset that was used as input data to restore the pk index
grid1km_subset_shap = pd.concat([grid1km_subset.loc[:, ["pk"]], shap_values_df, shap_values_perc_df], axis=1)

# Add the local predominant predictor as a column
grid1km_subset_shap['local_vimp'] = local_vimp 

# Merge with the original grid data
grid1km           = read_dataframe("saved/data/grid1km_pred.gpkg", layer= "grid1km_pred")
grid1km_pred_shap = pd.merge(grid1km, grid1km_subset_shap, "left", "pk")

# Set No Data values
grid1km_pred_shap['local_vimp'][grid1km_pred_shap['local_vimp'].isna()] = "No Data"

with open("saved/data/grid1km_pred_shap.pickle", 'wb') as f:
     pickle.dump(grid1km_pred_shap, f)




#------------------------- 31 Review the unique local predominant predictors

print("\n*** Progress 31/43: Review the unique local predominant predictors ***", end="\n\n")

print("Showing the unique local predominant predictors", end="\n\n")
print(grid1km_pred_shap['local_vimp'].unique())



#------------------------- 32 Force Plot on the first prediction on the grid

print("\n*** Progress 32/43: Force Plot on the first prediction on the grid ***", end="\n\n")

print("Prediction at first row: ", grid1km_subset['pred_trend'][0], "Mean prediction from X_test:", model.predict(X_test).mean())
shap.plots.force(explainer.expected_value[0], shap_values_df.to_numpy()[0,:], grid1km_subset.iloc[0][relevant_features], matplotlib= True, show= False)
plt.savefig("saved/plots/shap_force_plot.png", dpi= 300, bbox_inches='tight')
plt.show()


#------------------------- 33 Summary Plot

print("\n*** Progress 33/43: Summary Plot ***", end="\n\n")

shap.summary_plot(shap_values_df.to_numpy(), grid1km_subset[relevant_features], show= False)
plt.savefig("saved/plots/shap_summary_plot.png", dpi=300, bbox_inches='tight')
plt.show()



#------------------------- 34 Local Interpretability map BIO17ALT

print("\n*** Progress 34/43: Local Interpretability map BIO17ALT ***", end="\n\n")

cmap_continuous = sns.color_palette("rocket_r", as_cmap= True)
cmap_diverging = sns.color_palette("bwr", as_cmap= True)

# Getting the max values to center the legend on 0 for diverging color palette
vmax_shap = max( abs(grid1km_pred_shap['BIO17ALT_shap']) )
vmax_shap_perc = max( abs(grid1km_pred_shap['BIO17ALT_shap_perc']))

fig, axes = plt.subplots(nrows= 1, ncols= 3, figsize= (30, 7))
grid1km_pred_shap.plot(column= "BIO17ALT", legend= True, cmap= cmap_continuous, markersize= 0.1, ax= axes[0])
axes[0].axis('off')
axes[0].set_title("BIO17ALT Predictor")

grid1km_pred_shap.plot(column= "BIO17ALT_shap", legend= True, cmap= cmap_diverging, markersize= 0.1, ax= axes[1], vmin= -1*vmax_shap, vmax= vmax_shap)
axes[1].axis('off')
axes[1].set_title("BIO17ALT Local Impacts (SHAP Values)")

grid1km_pred_shap.plot(column= "BIO17ALT_shap_perc", legend= True, cmap=cmap_diverging, markersize= 0.1, ax= axes[2], vmin= -1*vmax_shap_perc, vmax= vmax_shap_perc)
axes[2].axis('off')
axes[2].set_title("BIO17ALT Local Impacts Proportional Share (%)")

plt.savefig("saved/plots/BIO17ALT.png", dpi= 300, bbox_inches= "tight")



#------------------------- 35 Local Interpretability map M11LSTDA01

print("\n*** Progress 35/43: Local Interpretability map M11LSTDA01 ***", end="\n\n")

# Getting the max values to center the legend on 0 for diverging color palette
vmax_shap = max( abs(grid1km_pred_shap['M11LSTDA01_shap']) )
vmax_shap_perc = max( abs(grid1km_pred_shap['M11LSTDA01_shap_perc']))

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 7))
grid1km_pred_shap.plot(column= "M11LSTDA01", legend= True, cmap= cmap_continuous, markersize= 0.1, ax= axes[0])
axes[0].axis('off')
axes[0].set_title("M11LSTDA01 Predictor")

grid1km_pred_shap.plot(column= "M11LSTDA01_shap", legend= True, cmap= cmap_diverging, markersize= 0.1, ax= axes[1], vmin=-1*vmax_shap, vmax=vmax_shap)
axes[1].axis('off')
axes[1].set_title("M11LSTDA01 Local Impacts (SHAP Values)")

grid1km_pred_shap.plot(column= "M11LSTDA01_shap_perc", legend= True, cmap= cmap_diverging, markersize= 0.1, ax= axes[2], vmin=-1*vmax_shap_perc, vmax=vmax_shap_perc)
axes[2].axis('off')
axes[2].set_title("M11LSTDA01 Local Impacts Proportional Share (%)")

plt.savefig("saved/plots/M11LSTDA01.png", dpi=300, bbox_inches= "tight")


#------------------------- 36 Local Interpretability map M11LSTDA05

print("\n*** Progress 36/43: Local Interpretability map M11LSTDA05 ***", end="\n\n")

# Getting the max values to center the legend on 0 for diverging color palette
vmax_shap = max( abs(grid1km_pred_shap['M11LSTDA05_shap']) )
vmax_shap_perc = max( abs(grid1km_pred_shap['M11LSTDA05_shap_perc']))

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 7))
grid1km_pred_shap.plot(column= "M11LSTDA05", legend= True, cmap= cmap_continuous, markersize= 0.1, ax= axes[0])
axes[0].axis('off')
axes[0].set_title("M11LSTDA05 Predictor")

grid1km_pred_shap.plot(column= "M11LSTDA05_shap", legend= True, cmap= cmap_diverging, markersize= 0.1, ax= axes[1], vmin=-1*vmax_shap, vmax=vmax_shap)
axes[1].axis('off')
axes[1].set_title("M11LSTDA05 Local Impacts (SHAP Values)")

grid1km_pred_shap.plot(column= "M11LSTDA05_shap_perc", legend= True, cmap= cmap_diverging, markersize= 0.1, ax= axes[2], vmin=-1*vmax_shap_perc, vmax=vmax_shap_perc)
axes[2].axis('off')
axes[2].set_title("M11LSTDA05 Local Impacts Proportional Share (%)")

plt.savefig("saved/plots/M11LSTDA05.png", dpi=300, bbox_inches= "tight")



#------------------------- 37 Local Interpretability map M13RB3A08

print("\n*** Progress 37/43: Local Interpretability map M13RB3A08 ***", end="\n\n")

# Getting the max values to center the legend on 0 for diverging color palette
vmax_shap = max( abs(grid1km_pred_shap['M13RB3A08_shap']) )
vmax_shap_perc = max( abs(grid1km_pred_shap['M13RB3A08_shap_perc']))

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 7))
grid1km_pred_shap.plot(column= "M13RB3A08", legend= True, cmap= cmap_continuous, markersize= 0.1, ax= axes[0])
axes[0].axis('off')
axes[0].set_title("M13RB3A08 Predictor")

grid1km_pred_shap.plot(column= "M13RB3A08_shap", legend= True, cmap=cmap_diverging, markersize= 0.1, ax= axes[1], vmin=-1*vmax_shap, vmax=vmax_shap)
axes[1].axis('off')
axes[1].set_title("M13RB3A08 Local Impacts (SHAP Values)")

grid1km_pred_shap.plot(column= "M13RB3A08_shap_perc", legend= True, cmap= cmap_diverging, markersize= 0.1, ax= axes[2], vmin=-1*vmax_shap_perc, vmax=vmax_shap_perc)
axes[2].axis('off')
axes[2].set_title("M13RB3A08 Local Impacts Proportional Share (%)")

plt.savefig("saved/plots/M13RB3A08.png", dpi=300, bbox_inches= "tight")



#------------------------- 38 Local Interpretability map BIO19ALT

print("\n*** Progress 38/43: Local Interpretability map BIO19ALT ***", end="\n\n")

# Getting the max values to center the legend on 0 for diverging color palette
vmax_shap = max( abs(grid1km_pred_shap['BIO19ALT_shap']) )
vmax_shap_perc = max( abs(grid1km_pred_shap['BIO19ALT_shap_perc']))

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 7))
grid1km_pred_shap.plot(column= "BIO19ALT", legend= True, cmap= cmap_continuous, markersize= 0.1, ax= axes[0])
axes[0].axis('off')
axes[0].set_title("BIO19ALT Predictor")

grid1km_pred_shap.plot(column= "BIO19ALT_shap", legend= True, cmap= cmap_diverging, markersize= 0.1, ax= axes[1], vmin=-1*vmax_shap, vmax=vmax_shap)
axes[1].axis('off')
axes[1].set_title("BIO19ALT Local Impacts (SHAP Values)")

grid1km_pred_shap.plot(column= "BIO19ALT_shap_perc", legend= True, cmap= cmap_diverging, markersize= 0.1, ax= axes[2], vmin=-1*vmax_shap_perc, vmax=vmax_shap_perc)
axes[2].axis('off')
axes[2].set_title("BIO19ALT Local Impacts Proportional Share (%)")

plt.savefig("saved/plots/BIO19ALT.png", dpi=300, bbox_inches= "tight")



#------------------------- 39 Local Interpretability map M09B4A01

print("\n*** Progress 39/43: Local Interpretability map M09B4A01 ***", end="\n\n")

# Getting the max values to center the legend on 0 for diverging color palette
vmax_shap = max( abs(grid1km_pred_shap['M09B4A01_shap']) )
vmax_shap_perc = max( abs(grid1km_pred_shap['M09B4A01_shap_perc']))

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 7))
grid1km_pred_shap.plot(column= "M09B4A01", legend= True, cmap= cmap_continuous, markersize= 0.1, ax= axes[0])
axes[0].axis('off')
axes[0].set_title("M09B4A01 Predictor")

grid1km_pred_shap.plot(column= "M09B4A01_shap", legend= True, cmap= cmap_diverging, markersize= 0.1, ax= axes[1], vmin=-1*vmax_shap, vmax=vmax_shap)
axes[1].axis('off')
axes[1].set_title("M09B4A01 Local Impacts (SHAP Values)")

grid1km_pred_shap.plot(column= "M09B4A01_shap_perc", legend= True, cmap= cmap_diverging, markersize= 0.1, ax= axes[2], vmin=-1*vmax_shap_perc, vmax=vmax_shap_perc)
axes[2].axis('off')
axes[2].set_title("M09B4A01 Local Impacts Proportional Share (%)")

plt.savefig("saved/plots/M09B4A01.png", dpi=300, bbox_inches= "tight")



#------------------------- 40 Automating the writing of all maps

print("\n*** Progress 40/43: Automating the writing of all maps ***", end="\n\n")

# This max value is to ensure we use the same scale for all plots being compared
vmax_shap = max( abs( 
                pd.concat([
                grid1km_pred_shap['BIO17ALT_shap'],
                grid1km_pred_shap['M11LSTDA01_shap'],
                grid1km_pred_shap['M11LSTDA05_shap'],
                grid1km_pred_shap['M13RB3A08_shap'],
                grid1km_pred_shap['BIO19ALT_shap'],
                grid1km_pred_shap['M11LSTDA03_shap'],
                grid1km_pred_shap['BIO6ALT_shap'],
                grid1km_pred_shap['M13RB3A07_shap'],
                grid1km_pred_shap['BIO14ALT_shap'],
                grid1km_pred_shap['M13RB2A09_shap']])
        ) 
)

# This max value is to ensure we use the same scale for all plots being compared
vmax_shap_perc = max( abs( 
                pd.concat([
                grid1km_pred_shap['BIO17ALT_shap_perc'],
                grid1km_pred_shap['M11LSTDA01_shap_perc'],
                grid1km_pred_shap['M11LSTDA05_shap_perc'],
                grid1km_pred_shap['M13RB3A08_shap_perc'],
                grid1km_pred_shap['BIO19ALT_shap_perc'],
                grid1km_pred_shap['M11LSTDA03_shap_perc'],
                grid1km_pred_shap['BIO6ALT_shap_perc'],
                grid1km_pred_shap['M13RB3A07_shap_perc'],
                grid1km_pred_shap['BIO14ALT_shap_perc'],
                grid1km_pred_shap['M13RB2A09_shap_perc']])
        ) 
)


# A huge dictionary for defining the specifications for automating the plotting
map_specs = {


				#--- Map 1

				# standalone visualization
				1:  { 'col_name': 'pred_trend',
								'title':'Total Nitrogen (g/kg) Mean',
								'subtitle':'',
								'legend_label': 'Total Nitrogen (g/kg)', 
								'vmin':None, 
								'vmax':None,
								'cmap_type':'continuous',
								'save_fn':'map1_TotalN_pred_mean.png'},
								
				#--- Map 2	

				# comparison visualization	
				2:  {'col_name': 'pred_q5',
							'title':'Total Nitrogen (g/kg) Q5%',
							'subtitle':'',
							'legend_label': 'Total Nitrogen (g/kg)', 
							'vmin':min(grid1km_pred_shap['pred_q5']), 
							'vmax':max(grid1km_pred_shap['pred_q95']),
							'cmap_type':'continuous',
							'save_fn':'map2_TotalN_pred_q5.png'},
				
				# comparison visualization			
				3: {'col_name': 'pred_q50',
					'title':'Total Nitrogen (g/kg) Q50%',
							'subtitle':'',
							'legend_label': 'Total Nitrogen (g/kg)', 
							'vmin':min(grid1km_pred_shap['pred_q5']), 
							'vmax':max(grid1km_pred_shap['pred_q95']),
							'cmap_type':'continuous',
							'save_fn':'map2_TotalN_pred_q50.png'},
				
				# comparison visualization				
				4: {'col_name': 'pred_q95',
					'title':'Total Nitrogen (g/kg) Q95%',
							'subtitle':'',
							'legend_label': 'Total Nitrogen (g/kg)', 
							'vmin':min(grid1km_pred_shap['pred_q5']), 
							'vmax':max(grid1km_pred_shap['pred_q95']),
							'cmap_type':'continuous',
							'save_fn':'map2_TotalN_pred_q95.png'},
             
			 	#--- Map 3

				# standalone visualization	
			    5: {'col_name': 'uncertainty',
					'title':'Total Nitrogen Prediction Uncertainty',
								'subtitle':'',
								'legend_label': 'Uncertainty (IQR/Q50)', 
								'vmin':None, 
								'vmax':None,
								'cmap_type':'continuous',
								'save_fn':'map3_TotalN_pred_uncertainty.png'},
							

				#--- Map 4

				# standalone visualization	
				6: {'col_name': 'BIO17ALT',
					'title':'BIO17ALT Predictor',
							'subtitle':'Precipitation of driest quarter',
							'legend_label': 'Precipitation (mm)', 
							'vmin':None, 
							'vmax':None,
							'cmap_type':'continuous',
							'save_fn':'map4_BIO17ALT_predictor.png'},

				# standalone visualization	
				7: {'col_name': 'BIO17ALT_shap',
					'title':'BIO17ALT Local Impacts (SHAP Values)',
								'subtitle':'Precipitation of driest quarter',
								'legend_label': None, 
								'vmin':-1* max(abs(grid1km_pred_shap['BIO17ALT_shap'])), 
								'vmax': max(abs(grid1km_pred_shap['BIO17ALT_shap'])),
								'cmap_type':'diverging',
								'save_fn':'map4_BIO17ALT_shap.png'},

				# standalone visualization	
				8: {'col_name': 'BIO17ALT_shap_perc',
					'title':'BIO17ALT Local Impacts (%)',
										'subtitle':'Precipitation of driest quarter',
										'legend_label': 'Percentage (%)', 
										'vmin':-1* max(abs(grid1km_pred_shap['BIO17ALT_shap_perc'])), 
										'vmax': max(abs(grid1km_pred_shap['BIO17ALT_shap_perc'])),
										'cmap_type':'diverging',
										'save_fn':'map4_BIO17ALT_shap_perc.png'},

				# comparison visualization	
				9: {'col_name': 'BIO17ALT_shap',
					'title':'BIO17ALT Local Impacts (SHAP Values)',
								'subtitle':'Precipitation of driest quarter',
								'legend_label': None, 
								'vmin': -1*vmax_shap, 
								'vmax': vmax_shap,
								'cmap_type':'diverging',
								'save_fn':'map4_BIO17ALT_shap_compare.png'},					

				# comparison visualization	
				10: {'col_name': 'BIO17ALT_shap_perc',
					'title':'BIO17ALT Local Impacts (%)',
									'subtitle':'Precipitation of driest quarter',
									'legend_label': 'Percentage (%)', 
									'vmin': -1*vmax_shap_perc, 
									'vmax': vmax_shap_perc,
									'cmap_type':'diverging',
									'save_fn':'map4_BIO17ALT_shap_perc_compare.png'},
									
										
				#--- Map 5		

				# standalone visualization	
				11: {'col_name': 'M11LSTDA01',
					'title':'M11LSTDA01 Predictor',
								'subtitle':'Land Surface Temperature Day for January',
								'legend_label': 'LST (Kelvin)', 
								'vmin':None, 
								'vmax':None,
								'cmap_type':'continuous',
								'save_fn':'map5_M11LSTDA01_predictor.png'},
							
				# standalone visualization	
				12: {'col_name': 'M11LSTDA01_shap',
					'title':'M11LSTDA01 Local Impacts (SHAP Values)',
									'subtitle':'Land Surface Temperature Day for January',
									'legend_label': None, 
									'vmin':-1* max(abs(grid1km_pred_shap['M11LSTDA01_shap'])), 
									'vmax': max(abs(grid1km_pred_shap['M11LSTDA01_shap'])),
									'cmap_type':'diverging',
									'save_fn':'map5_M11LSTDA01_shap.png'},
				
				# standalone visualization	
				13: {'col_name': 'M11LSTDA01_shap_perc',
					'title':'M11LSTDA01 Local Impacts (%)',
										'subtitle':'Land Surface Temperature Day for January',
										'legend_label': 'Percentage (%)', 
										'vmin':-1* max(abs(grid1km_pred_shap['M11LSTDA01_shap_perc'])), 
										'vmax': max(abs(grid1km_pred_shap['M11LSTDA01_shap_perc'])),
										'cmap_type':'diverging',
										'save_fn':'map5_M11LSTDA01_shap_perc.png'},
				
				# comparison visualization							
				14: {'col_name': 'M11LSTDA01_shap',
					'title':'M11LSTDA01 Local Impacts (SHAP Values)',
									'subtitle':'Land Surface Temperature Day for January',
									'legend_label': None, 
									'vmin':-1*vmax_shap, 
									'vmax':vmax_shap,
									'cmap_type':'diverging',
									'save_fn':'map5_M11LSTDA01_shap_compare.png'},
				
				# comparison visualization				
				15: {'col_name': 'M11LSTDA01_shap_perc',
					'title':'M11LSTDA01 Local Impacts (%)',
										'subtitle':'Land Surface Temperature Day for January',
										'legend_label': 'Percentage (%)', 
										'vmin':-1*vmax_shap_perc, 
										'vmax':vmax_shap_perc,
										'cmap_type':'diverging',
										'save_fn':'map5_M11LSTDA01_shap_perc_compare.png'},


				#--- Map 6	

				# standalone visualization	
				16: {'col_name': 'M11LSTDA05',
					'title':'M11LSTDA05 Predictor',
							'subtitle':'Land Surface Temperature Day for May',
							'legend_label': 'LST (Kelvin)', 
							'vmin':None, 
							'vmax':None,
							'cmap_type':'continuous',
							'save_fn':'map6_M11LSTDA05_predictor.png'},
							
				# standalone visualization	
				17: {'col_name': 'M11LSTDA05_shap',
					'title':'M11LSTDA05 Local Impacts (SHAP Values)',
									'subtitle':'Land Surface Temperature Day for May',
									'legend_label': None, 
									'vmin':-1* max(abs(grid1km_pred_shap['M11LSTDA05_shap'])), 
									'vmax': max(abs(grid1km_pred_shap['M11LSTDA05_shap'])),
									'cmap_type':'diverging',
									'save_fn':'map6_M11LSTDA05_shap.png'},
							
				# standalone visualization	
				18: {'col_name': 'M11LSTDA05_shap_perc',
					'title':'M11LSTDA05 Local Impacts (%)',
										'subtitle':'Land Surface Temperature Day for May',
										'legend_label': 'Percentage (%)', 
										'vmin':-1* max(abs(grid1km_pred_shap['M11LSTDA05_shap_perc'])), 
										'vmax': max(abs(grid1km_pred_shap['M11LSTDA05_shap_perc'])),
										'cmap_type':'diverging',
										'save_fn':'map6_M11LSTDA05_shap_perc.png'},
										
				# comparison visualization				
				19: {'col_name': 'M11LSTDA05_shap',
					'title':'M11LSTDA05 Local Impacts (SHAP Values)',
									'subtitle':'Land Surface Temperature Day for May',
									'legend_label': None, 
									'vmin':-1*vmax_shap, 
									'vmax':vmax_shap,
									'cmap_type':'diverging',
									'save_fn':'map6_M11LSTDA05_shap_compare.png'},
				
				# comparison visualization	
				20: {'col_name': 'M11LSTDA05_shap_perc',
					'title':'M11LSTDA05 Local Impacts (%)',
										'subtitle':'Land Surface Temperature Day for May',
										'legend_label': 'Percentage (%)', 
										'vmin':-1*vmax_shap_perc, 
										'vmax':vmax_shap_perc,
										'cmap_type':'diverging',
										'save_fn':'map6_M11LSTDA05_shap_perc_compare.png'},


				#--- Map 7

				# standalone visualization	
				21: {'col_name': 'M13RB3A08',
					'title':'M13RB3A08 Predictor',
							'subtitle':'MODIS Reflectance Band 3 (Blue) for August',
							'legend_label': None, 
							'vmin':None, 
							'vmax':None,
							'cmap_type':'continuous',
							'save_fn':'map7_M13RB3A08_predictor.png'},

				# standalone visualization		
				22: {'col_name': 'M13RB3A08_shap',
					'title':'M13RB3A08 Local Impacts (SHAP Values)',
									'subtitle':'MODIS Reflectance Band 3 (Blue) for August',
									'legend_label': None, 
									'vmin':-1* max(abs(grid1km_pred_shap['M13RB3A08_shap'])), 
									'vmax': max(abs(grid1km_pred_shap['M13RB3A08_shap'])),
									'cmap_type':'diverging',
									'save_fn':'map7_M13RB3A08_shap.png'},
				
				# standalone visualization	
				23: {'col_name': 'M13RB3A08_shap_perc',
					'title':'M13RB3A08 Local Impacts (%)',
										'subtitle':'MODIS Reflectance Band 3 (Blue) for August',
										'legend_label': 'Percentage (%)', 
										'vmin':-1* max(abs(grid1km_pred_shap['M13RB3A08_shap_perc'])), 
										'vmax': max(abs(grid1km_pred_shap['M13RB3A08_shap_perc'])),
										'cmap_type':'diverging',
										'save_fn':'map7_M13RB3A08_shap_perc.png'},
										

				# comparison visualization	
				24: {'col_name': 'M13RB3A08_shap',
					'title':'M13RB3A08 Local Impacts (SHAP Values)',
									'subtitle':'MODIS Reflectance Band 3 (Blue) for August',
									'legend_label': None, 
									'vmin':-1*vmax_shap, 
									'vmax':vmax_shap,
									'cmap_type':'diverging',
									'save_fn':'map7_M13RB3A08_shap_compare.png'},

				# comparison visualization				
				25: {'col_name': 'M13RB3A08_shap_perc',
					'title':'M13RB3A08 Local Impacts (%)',
										'subtitle':'MODIS Reflectance Band 3 (Blue) for August',
										'legend_label': 'Percentage (%)', 
										'vmin':-1*vmax_shap_perc, 
										'vmax':vmax_shap_perc,
										'cmap_type':'diverging',
										'save_fn':'map7_M13RB3A08_shap_perc_compare.png'},						

				#--- Map 8

				# standalone visualization	
				26: {'col_name': 'BIO19ALT',
					'title':'BIO19ALT Predictor',
							'subtitle':'Precipitation of coldest quarter',
							'legend_label': 'Precipitation (mm)', 
							'vmin':None, 
							'vmax':None,
							'cmap_type':'continuous',
							'save_fn':'map8_BIO19ALT_predictor.png'},
				
				# standalone visualization	
				27: {'col_name': 'BIO19ALT_shap',
					'title':'BIO19ALT Local Impacts (SHAP Values)',
								'subtitle':'Precipitation of coldest quarter',
								'legend_label': None, 
								'vmin':-1* max(abs(grid1km_pred_shap['BIO19ALT_shap'])), 
								'vmax': max(abs(grid1km_pred_shap['BIO19ALT_shap'])),
								'cmap_type':'diverging',
								'save_fn':'map8_BIO19ALT_shap.png'},
							
				# standalone visualization	
				28: {'col_name': 'BIO19ALT_shap_perc',
					'title':'BIO19ALT Local Impacts (%)',
										'subtitle':'Precipitation of coldest quarter',
										'legend_label': 'Percentage (%)', 
										'vmin':-1* max(abs(grid1km_pred_shap['BIO19ALT_shap_perc'])), 
										'vmax': max(abs(grid1km_pred_shap['BIO19ALT_shap_perc'])),
										'cmap_type':'diverging',
										'save_fn':'map8_BIO19ALT_shap_perc.png'},


				# comparison visualization	
				29: {'col_name': 'BIO19ALT_shap',
					'title':'BIO19ALT Local Impacts (SHAP Values)',
								'subtitle':'Precipitation of coldest quarter',
								'legend_label': None, 
								'vmin':-1*vmax_shap, 
								'vmax':vmax_shap,
								'cmap_type':'diverging',
								'save_fn':'map8_BIO19ALT_shap_compare.png'},

				# comparison visualization			
				30: {'col_name': 'BIO19ALT_shap_perc',
					'title':'BIO19ALT Local Impacts (%)',
										'subtitle':'Precipitation of coldest quarter',
										'legend_label': 'Percentage (%)', 
										'vmin':-1*vmax_shap_perc, 
										'vmax':vmax_shap_perc,
										'cmap_type':'diverging',
										'save_fn':'map8_BIO19ALT_shap_perc_compare.png'},



				#--- Map 9

				# standalone visualization	
				31: {'col_name': 'M09B4A01',
					'title':'M09B4A01 Predictor',
							'subtitle':'MODIS Green Surface Reflectance',
							'legend_label': '', 
							'vmin':None, 
							'vmax':None,
							'cmap_type':'continuous',
							'save_fn':'map9_M09B4A01_predictor.png'},
						
				# standalone visualization	
				32: {'col_name': 'M09B4A01_shap',
					'title':'BIO19ALT Local Impacts (SHAP Values)',
								'subtitle':'MODIS Green Surface Reflectance',
								'legend_label': None, 
								'vmin':-1* max(abs(grid1km_pred_shap['M09B4A01_shap'])), 
								'vmax': max(abs(grid1km_pred_shap['M09B4A01_shap'])),
								'cmap_type':'diverging',
								'save_fn':'map9_M09B4A01_shap.png'},

				# standalone visualization		
				33: {'col_name': 'M09B4A01_shap_perc',
					'title':'M09B4A01 Local Impacts (%)',
										'subtitle':'MODIS Green Surface Reflectance',
										'legend_label': 'Percentage (%)', 
										'vmin':-1* max(abs(grid1km_pred_shap['M09B4A01_shap_perc'])), 
										'vmax': max(abs(grid1km_pred_shap['M09B4A01_shap_perc'])),
										'cmap_type':'diverging',
										'save_fn':'map9_M09B4A01_shap_perc.png'},


				# comparison visualization	
				34: {'col_name': 'M09B4A01_shap',
					'title':'M09B4A01 Local Impacts (SHAP Values)',
								'subtitle':'MODIS Green Surface Reflectance',
								'legend_label': None, 
								'vmin':-1*vmax_shap, 
								'vmax':vmax_shap,
								'cmap_type':'diverging',
								'save_fn':'map9_M09B4A01_shap_compare.png'},

				# comparison visualization			
				35: {'col_name': 'M09B4A01_shap_perc',
					'title':'M09B4A01 Local Impacts (%)',
										'subtitle':'MODIS Green Surface Reflectance',
										'legend_label': 'Percentage (%)', 
										'vmin':-1*vmax_shap_perc, 
										'vmax':vmax_shap_perc,
										'cmap_type':'diverging',
										'save_fn':'map9_M09B4A01_shap_perc_compare.png'},

}



cmap_continuous = sns.color_palette("rocket_r", as_cmap=True)
cmap_diverging = sns.color_palette("bwr", as_cmap=True)

for key in map_specs:
	map_spec = map_specs[key]
	col_name = map_spec['col_name']
     
	print("Map {}/{}-{}".format(key, len(map_spec), map_spec['title']))
	
	fig, ax = plt.subplots(figsize=(10, 7))

	gdf = grid1km_pred_shap[[col_name,'geometry']]	
	map_spec_cmap = cmap_continuous if map_spec['cmap_type'] == 'continuous' else cmap_diverging
	gdf.plot(column= col_name, legend= True, legend_kwds= {"label": map_spec['legend_label']}, ax= ax, cmap= map_spec_cmap, markersize= 0.1, vmin= map_spec['vmin'], vmax= map_spec['vmax'])
	ax.axis("off")
	ax.set_title(map_spec['title'], size=13, fontweight="bold")

	plt.suptitle(map_spec['subtitle'], x=0.45, y= 0.15, fontsize= 13, horizontalalignment="center")
	# plt.title("sub title", fontsize=10)

	plt.savefig(f"saved/plots/{map_spec['save_fn']}", dpi=300, bbox_inches= "tight")

	plt.show()



#------------------------- Local Predominant Predictor

#------------------------- 41 Review Local Predominant Variable statistics

print("\n*** Progress 41/43: Review Local Predominant Variable statistics ***", end="\n\n")

local_vimp_stats = grid1km_pred_shap['local_vimp'].value_counts().to_frame()
local_vimp_stats['percent'] = round( (local_vimp_stats['count'] / local_vimp_stats['count'].sum()) * 100, 2)
local_vimp_stats = local_vimp_stats.reset_index()

print("Showing the Local Predominant Predictor statistics", end="\n\n")
print(local_vimp_stats)

local_vimp_stats.to_csv('saved/data/local_vimp_raw.csv', index=False)

# Plot
local_vimp_stats.plot.barh(x='local_vimp', y='percent')
plt.gca().invert_yaxis()

plt.savefig("saved/plots/local_vimp_raw.png", dpi=300, bbox_inches= "tight")
plt.show()



#------------------------- 42 Reclassify Local Predominant predictors
# Set Variables whose count are less than 300 to "Others" because they are insignificant and thus cannot be seen on the static map. As recommendation, go interactive

print("\n*** Progress 42/43: Reclassify Local Predominant Predictors ***", end="\n\n")

# Merge the stats columns into the main table
grid1km_pred_shap = grid1km_pred_shap.merge(local_vimp_stats, 'left', on='local_vimp')

# Query for count less than 300 and set local_vimp to 'Others'
print("Assigning a class of 'Others' to predictors whose pixel count is less than 300 representing less than 0.1 percent of the total pixels\n\n")
qry = grid1km_pred_shap['count'] < 300 
grid1km_pred_shap.loc[qry, 'local_vimp'] = 'Others'

### Re-calculate the stats and review plot
local_vimp_stats = grid1km_pred_shap['local_vimp'].value_counts().to_frame()
local_vimp_stats['percent'] = round( (local_vimp_stats['count'] / local_vimp_stats['count'].sum()) * 100, 2)
local_vimp_stats = local_vimp_stats.reset_index()

print("Showing the Local Predominant Predictor statistics after reclassification", end="\n\n")
print(local_vimp_stats)

local_vimp_stats.to_csv('saved/data/local_vimp_reclassified.csv', index=False)

local_vimp_stats.plot.barh(x='local_vimp', y='percent')
plt.gca().invert_yaxis()

plt.savefig("saved/plots/local_vimp_reclassified.png", dpi=300, bbox_inches= "tight")
plt.show()



#------------------------- 43 Define a color scheme for visualization

print("\n*** Progress 43/43: Define a color scheme for visualization ***", end="\n\n")

fig, ax = plt.subplots(figsize=(10, 7))
# fig.set_facecolor("#4b5563")

# Define custom colors
color_dict = {'BIO17ALT':  '#a3a3a3',
              'M11LSTDA01': '#dc2626',
              'M11LSTDA05': 'gold',
              'M13RB3A08': '#EC08FF', 
              'BIO19ALT': 'green',                      
              'Others' :   'blue',
              'No Data' :  'white'}

# Set color column
grid1km_pred_shap['local_vimp_color'] = grid1km_pred_shap['local_vimp'].map(color_dict)

grid1km_pred_shap.plot(legend= True, figsize= (12,10), markersize= 0.17, color= grid1km_pred_shap['local_vimp_color'], ax=ax)

# Add manual legend
# https://github.com/geopandas/geopandas/issues/2279
from matplotlib.lines import Line2D
custom_points = [Line2D([0], [0], marker="o", linestyle="none", markersize=10, color=color) for color in color_dict.values()]
leg_points = ax.legend(custom_points, color_dict.keys())
ax.add_artist(leg_points)
ax.axis('off')

plt.savefig("saved/plots/local predominant predictor map.png", dpi=300, bbox_inches= "tight")

print("\n*** All done ***", end= "\n\n")


"""
References:

Adadi, A., & Berrada, M. (2018). Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI). IEEE Access, 6, 52138-52160. https://doi.org/10.1109/ACCESS.2018.2870052

Gartner, G. (2023). Towards a Research Agenda for Increasing Trust in Maps and Their Trustworthiness. Kartografija i Geoinformacije, 21, 48-58. https://doi.org/10.32909/kg.21.si.4

Kang, Y., Zhang, Q., & Roth, R. (2023). The Ethics of AI-Generated Maps: A Study of DALLE 2 and Implications for Cartography (arXiv:2304.10743). arXiv. http://arxiv.org/abs/2304.10743

Robinson, A. C., Çöltekin, A., Griffin, A. L., & Ledermann, F. (2023). Cartography in GeoAI: Emerging Themes and Research Challenges. Proceedings of the 6th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery, 1-2. https://doi.org/10.1145/3615886.3627734
"""