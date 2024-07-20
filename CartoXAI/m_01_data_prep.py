import geopandas as gpd
import pandas as pd
import pyogrio as pyogr
import os


def prepare_obs_data(in_gee_csv_dir, in_file_obs, target_col, pk_col= "pk", out_file_gpkg= None, out_gpkg_lyr_name= None, target_epsg=32630):
    """
    Function to merge all covariate csv files extracted from Google Earth Engine, with the observation data.

    Parameters
    ---------

    in_gee_csv_dir: str
        The full or relative path to the folder that contains the extracted csv files from Google Earth Engine
    in_file_obs: str
        The full or relative file path to the input observations data in geopackage format
    target_col: str
        The column name of the target variable on the observations dataset
    pk_col: str
        The column name of the primary key on the observations dataset (default: pk)
    out_file_gpkg: str
        The full or relative file path to save the prepared observation data. Passing None means no data saved to disk (default: None)
    out_gpkg_lyr_name: str
        The name of the layer to use for storing the prepared observation data in the output geopackage. Ignored if out_file_gpkg is None  (default: None)
    """

    # Load data
    obs = gpd.read_file(in_file_obs, engine= "pyogrio")
    obs = obs[[pk_col, target_col, "geometry"]]

    # Reproject to the target projected crs
    if obs.crs.to_epsg() != target_epsg:
        print(f"Reprojecting the observation points to EPSG:{target_epsg}")
        obs = obs.to_crs(f"EPSG:{target_epsg}")

    # Convert pk column to integer
    obs = obs.astype({pk_col: int})

    # Merge all csv files extracted from GEE
    all_csv_files = [in_gee_csv_dir + "/" + file for file in os.listdir(in_gee_csv_dir) if '.csv' in file]

    for f in all_csv_files:
        data = pd.read_csv(f, usecols= lambda col: col not in ["system:index", ".geo"])
        data = data.astype({"fid": int})

        # drop the fid column after merging to avoid duplicate column errors on subsequent merging
        obs = obs.merge(data, "left", left_on= pk_col, right_on= "fid").drop(columns= "fid")


    # Save
    if out_file_gpkg is not None:
        pyogr.write_dataframe(obs, out_file_gpkg, out_gpkg_lyr_name, driver= "GPKG")

    return obs


def prepare_grid_data(in_gee_csv_dir, in_file_grid, out_file_gpkg= None, out_gpkg_lyr_name= None, target_epsg=32630):
    """
    Function to merge all covariate csv files extracted from Google Earth Engine, with the grid data.

    Parameters
    ---------

    in_gee_csv_dir: str
        The full or relative path to the folder that contains the extracted csv files from Google Earth Engine
    in_file_grid: str
        The full or relative file path to the input grid data in shapefile format
    out_file_gpkg: str
        The full or relative file path to save the prepared grid data. Passing None means no data saved to disk (default: None)
    out_gpkg_lyr_name: str
        The name of the layer to use for storing the prepared grid data in the output geopackage. Ignored if out_file_gpkg is None (default: None)
    """
    
    # Load data
    grid = gpd.read_file(in_file_grid, engine="pyogrio")

    # Reproject to UTMZ30N to facilitate the geostatistics downstream
    if grid.crs.to_epsg() != target_epsg:
        print(f"Reprojecting the grid points to EPSG:{target_epsg}")
        grid = grid.to_crs(f"EPSG:{target_epsg}")

    # Convert fid to integer
    grid = grid.astype({'fid': int})
    grid.head()

    # Merge all files with the grid shapefile
    all_csv_files = [in_gee_csv_dir + "/" + file for file in os.listdir(in_gee_csv_dir) if '.csv' in file]

    for f in all_csv_files:
        data = pd.read_csv(f, usecols= lambda col: col not in ["system:index",".geo","pk"])
        grid  = grid.merge(data, "left", "fid")

    # Save
    if out_file_gpkg is not None:
        pyogr.write_dataframe(grid, out_file_gpkg, out_gpkg_lyr_name, driver= "GPKG")

    return grid