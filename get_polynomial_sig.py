# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:52:29 2024

@author: ecoramy
"""

import pandas as pd
import numpy as np
from scipy import stats
import math
import random
import common
from pathlib import Path
import scipy.stats as statsf
import os
import re
from sklearn import preprocessing
from iisignature import *
from sklearn import linear_model
import numbers
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

# Solve Non-Deterministic Problems
# https://pygad.readthedocs.io/en/latest/pygad_more.html#solve-non-deterministic-problems
# https://blog.derlin.ch/genetic-algorithms-with-pygad
# Hyperparameter Tuning the Random Forest in Python
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

def get_base_scale(c):
    return 100+(c/ pow(10, math.floor(math.log10(max(abs(c)))))/1)

def get_returns(c):
    return np.diff(np.log(c)) 


def main(portfolio_date, options):
    print(Path(__file__).stem,portfolio_date)
   
    date_regex = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
    match = date_regex.match(portfolio_date)
   
    assert match, "invalid portfolio_date"
   
    base_directory = Path(__file__).parent.resolve() / "portfolios" / portfolio_date
   
    # market = common.FUND_TICKER
    market = "IVV"
    level = 'Stock'
    as_of_date = "AsOf"
    
    
    results_sig_polynomial = os.path.join( base_directory, R"4_results_sig_polynomial" )
    Path(results_sig_polynomial).mkdir(parents=True, exist_ok=True)
    
    ret_mrkt_risk = pd.read_csv(os.path.join( base_directory, R"2_dataProcessed\Factors\ETF_FACTORS_qoq_returns.csv"), header=0, index_col=0)    
    ret_mrkt_risk =ret_mrkt_risk.drop(ret_mrkt_risk.index[-1])
    
    ret_equity_file = base_directory / "2_dataProcessed" / "Assets" / "Asset_q_returns.csv"
    ret_equity =pd.read_csv(ret_equity_file, index_col=0)
    ret_equity =ret_equity.drop(ret_equity.index[-1])
    # asset_ret_check = asset_ret[common.stock_tickers]   
    # common_factors = False
    factor_ret = None
           
    F_folder = base_directory / "2_dataProcessed" / "Factors"      
    
    sigga_best_aggregate = pd.DataFrame()  # Initialize the aggregate DataFrame
               
    flag = False
    # for stock_ticker in os.listdir(F_folder):
    for stock_ticker in common.stock_tickers:
        print(f"Processing ticker: {stock_ticker}")        
       
        # if stock_ticker != "BMY":
        #     continue
        
        F_folder_stock = F_folder / stock_ticker
        # checking if it is a file
        if not F_folder_stock.is_dir():
            continue

        if flag:
            print(",", end="")
        flag = True
        print(Path(os.path.basename(stock_ticker)).with_suffix(''), end="")
        
                                                                               
        ret_stock = ret_equity.loc[:, stock_ticker]
        ret_factor = pd.read_csv( F_folder_stock / 'FDM_ECOFIN_q_returns.csv', index_col=0 )
        ret_factor =ret_factor.drop(ret_factor.index[-1])
       
        stats_folder = base_directory /"3_results_sigga_wexpand" /"3_wexpand_leadl2time_sigga_L3" / "AsOf" / common.FUND_TICKER / "Stock" / stock_ticker
        generations_file = stats_folder / 'solution_generations.txt'
        
        if not generations_file.exists():
            print("(skipping)", end="")
            continue
                
        sigga_selections = pd.read_csv(generations_file, header=None, index_col=0)
        sigga_best = pd.read_csv(stats_folder / 'selection.csv', header=None, index_col=None, nrows=1)                                 
        
        sigga_best["stock_ticker"] = stock_ticker
        sigga_best_aggregate = pd.concat([sigga_best_aggregate, sigga_best], ignore_index=True)            
        
        temp = pd.concat([ret_stock, ret_factor, ret_mrkt_risk ], axis=1, join='inner')
        temp.index.name = 'Time'
        temp.index = pd.to_datetime( temp.index )
        names = temp.columns
               
        num_columns = sigga_best.shape[1]        
        # selected_chromosomes =list( sig_ga_best.iloc[0,1:num_columns] )
        selected_chromosomes =list( sigga_best.iloc[0,1:(num_columns-1)] )         
        variable_x =(temp.loc[:, selected_chromosomes]).iloc[:,0]
        variable_multi_x =temp.loc[:, selected_chromosomes]
        variable_multi_x_ave = variable_multi_x.mean(axis=1)
                        
        degree =2    
        # # Define a custom parameter to control the number of features
        # num_features = 30  # You decide this parameter                
        
        # Create polynomial features
        poly =PolynomialFeatures(degree)
        X_poly_full  =poly.fit_transform(variable_multi_x)        
        
        # # # Select the first 'num_features' columns (excluding the constant term if needed)
        # if num_features < X_poly_full.shape[1]:  # Check if truncation is needed
        #     X_poly = X_poly_full[:, :num_features]
        # else:
        #     X_poly = X_poly_full  # Use all features if num_features is too high
        
        # if degree == 2:
        #     num_features = 20
        #     X_poly = X_poly_full[:, :num_features]
        #     # X_poly = X_poly_full
        # elif degree == 3:
        #     num_features = 25
        #     X_poly = X_poly_full[:, :num_features]
        # elif degree == 4:
        #     num_features = 30
        #     X_poly = X_poly_full[:, :num_features]
        # else:
        #     raise ValueError("Unsupported degree. Only degrees 2, 3, or 4 are supported.")

        if degree == 2:
            num_features = 5
            X_poly = X_poly_full[:, :num_features]
            # X_poly = X_poly_full
        elif degree == 3:
            num_features = 12
            X_poly = X_poly_full[:, :num_features]
        elif degree == 4:
            num_features = 12
            X_poly = X_poly_full[:, :num_features]
        else:
            raise ValueError("Unsupported degree. Only degrees 2, 3, or 4 are supported.")                                    
        
        # Fit the polynomial regression model
        model =LinearRegression()
        model.fit(X_poly, ret_stock)
        
        # Predictions
        y_pred =model.predict(X_poly)
        
        # Evaluate R-squared
        r2 =r2_score(ret_stock, y_pred)
        sse_df =len(ret_stock) -num_features -1
        sst_df =len(ret_stock) -1
        r2_adj =1 - (1 - r2) * sst_df / sse_df
        r2_adjusted =1 - ((1 - r2)/ sse_df) * sst_df
        
                
        # Perform a two-sample t-test
        t_stat, p_value = stats.ttest_ind(y_pred, ret_stock, equal_var=False, alternative='two-sided')
        print(f"t-statistic: {t_stat}, p-value: {p_value}")
        
        # # Perform a two-sample t-test
        # ret_stock_mean = ret_stock.mean()
        # t_stat, p_value = stats.ttest_1samp(y_pred, ret_stock_mean)
        # print(f"t-statistic: {t_stat}, p-value: {p_value}")
        
        
        # # Perform a two-sample t-test
        # t_stat, p_value = stats.ttest_rel(ret_stock, y_pred)
        # print(f"t-statistic: {t_stat}, p-value: {p_value}")
        
        
        # Output the results
        # print(f"T-statistic: {t_stat}")
        # print(f"P-value: {p_value}")
        
        # Interpretation
        alpha = 0.05  # Significance level
        if p_value < alpha:
            print("Reject the null hypothesis: Statistically significant result. The groups are significantly different.")
        else:
            print("Fail to reject the null hypothesis: Not statistically significant. No significant difference between the groups.") 
        
        
        print(f"R-squared:{r2}")
        print(f"adj-R-squared:{r2_adj}")
        print(f"T-statistic: {t_stat}")
        print(f"p-value:{p_value}")
        
        # Coefficients
        print(f"Model coefficients: {model.coef_}")
        print(f"Model predictions: {y_pred}")
                   
               
        final_y_pred_file = os.path.join(base_directory, R"4_results_sig_polynomial\polynomial_ypred.csv")

        
        # If the file doesn't exist, create it; otherwise, append the new data
        if not os.path.exists(final_y_pred_file):            
            polynomial_ypred = pd.DataFrame({
                stock_ticker: y_pred, 
                "ret Actual": ret_stock
            }, dtype='float64') 
            polynomial_ypred.to_csv(final_y_pred_file, index=True)
        else:
            # Read the existing file
            existing_data = pd.read_csv(final_y_pred_file, index_col=0)
            
            # Append the new predictions as a new column            
            new_data = pd.DataFrame({
                stock_ticker: y_pred, 
                "ret Actual": ret_stock
            }, dtype='float64')
            combined_data = pd.concat([existing_data, new_data], axis=1)
            
            # # Save the updated file
            combined_data.to_csv(final_y_pred_file)
        
        
        
        final_r_squared_file = os.path.join(base_directory, R"4_results_sig_polynomial\polynomial_r2.csv")
        
        # If the file doesn't exist, create it; otherwise, append the new data
        if not os.path.exists(final_r_squared_file):
            polynomial_r2 = pd.DataFrame({ "stock_name": [stock_ticker], "R_squared": [r2],"adjR_squared": [r2_adj] })            
            polynomial_r2.to_csv(final_r_squared_file, index=False)
        else:
            # Read the existing file
            existing_data = pd.read_csv(final_r_squared_file)
            
            # Append the new predictions as a new row
            new_data = pd.DataFrame({ "stock_name": [stock_ticker], "R_squared": [r2],"adjR_squared": [r2_adj] })            
            combined_data = pd.concat([existing_data, new_data], axis=0)
            
            # Sort the combined data by "adjR_squared" in descending order
            combined_data = combined_data.sort_values(by="adjR_squared", ascending=True)
            
            # # Save the updated file
            combined_data.to_csv(final_r_squared_file, index=False)                      
                
            # Output the aggregate DataFrame to a file if needed
            sigga_best_aggregate.to_csv(base_directory / R"4_results_sig_polynomial\sigga_best_aggregate.csv", index=False)    
                                               

if __name__ == "__main__":  
    options=dict()  
    main("2021-03-31", options)
   
    # from scipy import stats
    # # Example data: Sample data for two groups
    # group1 = [5, 6, 7, 8, 9]  # Data for group 1
    # group2 = [7, 8, 9, 10, 11]  # Data for group 2
    
    # # Perform a two-sample t-test
    # t_stat, p_value = stats.ttest_ind(group1, group2)
    
    # # Output the results
    # print(f"T-statistic: {t_stat}")
    # print(f"P-value: {p_value}")
    
    # # Interpretation
    # alpha = 0.05  # Significance level
    # if p_value < alpha:
    #     print("Reject the null hypothesis: Statistically significant result. The groups are significantly different.")
    # else:
    #     print("Fail to reject the null hypothesis: Not statistically significant. No significant difference between the groups.") 
    
    '''
    Null Hypothesis (H0): The sample data occurs purely from chance.
    Alternative Hypothesis (HA): The sample data is influenced by some non-random cause.
    '''
    # import numpy as np
    # import roughpy as rp

    # # Define parameters
    # k = 3  # Signature depth
    # x = [5, 6, 7, 8, 9, 15, -95, 0.5, -4, 15, -25, 0.33, -0.57, 69, -1, 2, 3]
    
    # # Prepare the increments as a 2D array
    # x = np.array(x).reshape(-1, 1)
    
    # # Create the Lie Increment Stream
    # stream = rp.LieIncrementStream.from_increments(x, depth=k)
    
    # # Define a valid interval
    # interval = rp.RealInterval(0, len(x) - 1)

    # # Compute the signature
    # sig = stream.signature(interval, depth=k)
    
    # print("Signature:", sig)                                                                     