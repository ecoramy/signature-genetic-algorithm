# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:52:29 2024

@author: ecoramy
"""

import pandas as pd
import numpy as np
from scipy import stats
import math
from pathlib import Path
import os
import re
from sklearn.ensemble import RandomForestRegressor
import pygad
from scipy import stats
from timeit import default_timer as timer
from datetime import timedelta
import get_sig_functions
import time
import iisignature as isig
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
np.random.seed(42)  # Set NumPy seed
pygad.random.seed(42)  # Set pyGAD's internal random seed


def append_keys_to_stock_names(df, keys):
    """
    Append keys to the 'stock_name' column in the dataframe based on the given keys list.
    
    Args:
    - df (pd.DataFrame): The dataframe with a 'stock_name' column.
    - keys (list): List of keys to append.
    
    Returns:
    - pd.DataFrame: The updated dataframe.
    """
    rows_per_key = len(df) // len(keys)
    remainder = len(df) % len(keys)

    # Assign keys to rows
    current_key_index = 0
    key_counts = [rows_per_key + (1 if i < remainder else 0) for i in range(len(keys))]
    
    for i, count in enumerate(key_counts):
        start_idx = sum(key_counts[:i])
        end_idx = start_idx + count
        df.loc[start_idx:end_idx-1, "stock_name"] += f"_{keys[current_key_index]}"
        current_key_index += 1

    return df


def main(portfolio_date, options):
    print(Path(__file__).stem,portfolio_date)
   
    date_regex = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
    match = date_regex.match(portfolio_date)
   
    assert match, "invalid portfolio_date"
   
    base_directory = Path(__file__).parent.resolve() / "portfolios" / portfolio_date         
    
    results_sig_polynomial = os.path.join( base_directory, R"4_results_sigga_polynomial" )
    Path(results_sig_polynomial).mkdir(parents=True, exist_ok=True)
                       
    base_folder = results_sig_polynomial  # Replace with the path to the parent folder containing the 10 directories

    # List to store the dataframes
    dataframes = []

    # Loop through the directories to concatenate the sigga results of PLS regression
    for folder_name in os.listdir(base_folder):
        if folder_name.startswith("wexpand"):  # Check if the folder name starts with "wdyadic"
            file_path = os.path.join(base_folder, folder_name, "sigga_best_aggregate.csv")
            
            # Check if the file exists
            if os.path.exists(file_path):
                # Read the CSV file and append it to the list
                df = pd.read_csv(file_path)
                dataframes.append(df)
            else:
                print(f"File not found: {file_path}")

    # Concatenate all dataframes
    if dataframes:
        concatenated_df = pd.concat(dataframes, ignore_index=True)
        print("All files concatenated successfully.")
        
        # Save the concatenated dataframe to a new CSV file
        output_file = os.path.join(base_folder, "concatenated_sigga_best_aggregate.csv")
        concatenated_df.to_csv(output_file, index=False)
        print(f"Concatenated file saved to: {output_file}")
    else:
        print("No files were concatenated.")
    
    
    
    
    
    # Loop through the directories to concatenate the sigga results of Polynomial regression
    keys = ["bpt_sigD1", "bpt_sigD2", "bpt_sigD3", "LL1t_sigD1", "LL1t_sigD2", "LL1t_sigD3", "LL2t_sigD1", "LL2t_sigD2", "LL2t_sigD3"]
    dataframes_poly = []
    for folder_name in os.listdir(base_folder):
        if folder_name.startswith("wexpand"):  # Check if the folder name starts with "wdyadic"
            file_path = os.path.join(base_folder, folder_name, "polynomial_r2.csv")
            
            # Check if the file exists
            if os.path.exists(file_path):
                # Read the CSV file and append it to the list
                df = pd.read_csv(file_path)
                dataframes_poly.append(df)
            else:
                print(f"File not found: {file_path}")

    # Concatenate all dataframes
    if dataframes_poly:
        concatenated_df = pd.concat(dataframes_poly, ignore_index=True)
        print("All files concatenated successfully.")
        
        # Save the concatenated dataframe to a new CSV file
        # output_file = os.path.join(base_folder, "concatenated_sigga_polynomial_r2.csv")
        # concatenated_df.to_csv(output_file, index=False)
        concatenated_df_tagged = append_keys_to_stock_names(concatenated_df, keys)
        file_path_tag  =os.path.join(base_folder, "concatenated_sigga_polynomial_r2_tagged.csv")
        concatenated_df_tagged.to_csv(file_path_tag, index=False)
        print(f"Concatenated file saved to: {file_path_tag}")
    else:
        print("No files were concatenated.")
         


# print(updated_df.head())  
      
                                               

if __name__ == "__main__":  
    options=dict()  
    main("2021-03-31", options)
   





                     
                                              