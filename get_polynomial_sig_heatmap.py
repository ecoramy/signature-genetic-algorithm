# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:52:29 2024

@author: ecoramy
"""
import get_polynomial_sig_graphing
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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

ecofin_list = [
    "USAADP", "USABCONF", "USABOT", "USABP", "USABR", "USACA", "USACARS", 
    "USACBBS", "USACCONF", "USACCPI", "USACF", "USACFNAI", "USACHJC", 
    "USACJC", "USACNCN", "USACOR", "USACOSC", "USACP", "USACPIC", 
    "USACPMI", "USACPPI", "USACSP", "USACU", "USADINV", "USADPINC", 
    "USADUR", "USAEHS", "USAEMPST", "USAEXPX", "USAEXVOL", "USAFACT", 
    "USAFBI", "USAFDI", "USAFER", "USAFINF", "USAFOET", "USAGAGR", 
    "USAGASSC", "USAGBVL", "USAGCP", "USAGD", "USAGFCF", "USAGGR", 
    "USAGPAY", "USAGREV", "USAGSP", "USAGYLD", "USAHSTT", "USAIMPX", 
    "USAIMVOL", "USAISMNYI", "USAJCLM", "USAJOBOFF", "USAJVAC", "USALC", 
    "USALPS", "USALUNR", "USAM0", "USAM1", "USAM2", "USAMANWG", "USAMKT", 
    "USAMORTG", "USAMP", "USAMPAY", "USANAHB", "USANATGSC", "USANFIB", 
    "USANHS", "USANLTTF", "USANMPMI", "USANO", "USAOIL", "USAOPT", 
    "USAPCEPI", "USAPFED", "USAPHS", "USAPPIC", "USAPROD", "USAPSAV", 
    "USARSM", "USARSY", "USATOT", "USATOUR", "USATVS", "USAUNR", 
    "USAUNRY", "USAWAGE"
]

# ETF list
etf_list = [
    "IVV", "IJK", "OEF", "IWO", "IUSG", "IWV", "IWB", "ITOT", "IJJ", "IVW", "IWN", "IOO",
    "IUSV", "IYY", "IWR", "AGG", "IJH", "IVE", "IJT", "IWP", "IWM", "IJR", "DVY", "IJS",
    "IWS", "EPP", "EEM", "ILF", "EWA", "EWO", "EWK", "EWC", "EWQ", "EWG", "EWH", "EWI",
    "EWJ", "JPXN", "EWN", "EWS", "EWP", "EWD", "EWL", "EWU", "FXI", "EWZ", "EWM", "EWW",
    "EZA", "EWY", "EWT", "SHY", "IEF", "TLT", "TIP", "LQD", "IYK", "IYC", "IXC", "IYE",
    "IXG", "IYG", "IYF", "IXJ", "IBB", "IYH", "IYT", "IYJ", "IYM", "IYR", "ICF", "IGM",
    "IGV", "SOXX", "IYW", "IXP", "IYZ", "IDU"
]

# Balance sheet, income statement, and cash flow data, fundamenal 
fdm_list = [
    "cashnequsd", "divyield", "debtusd", "equityusd", "taxassets", "ppnenet", "inventory",
    "intangibles", "investments", "receivables", "accoci", "retearn", "taxliabilities",
    "deferredrev", "deposits", "payables", "assetsnc", "assets", "liabilities",
    "assetsc", "debtc", "investmentsc", "liabilitiesc", "debtnc", "investmentsnc",
    "liabilitiesnc", "ebitdausd", "ps1", "pe1", "ebt", "ev", "fcf", "fcfps",
    "grossmargin", "invcap", "bvps", "evebitda", "evebit", "de", "pb", "tbvps",
    "ps", "ebitdamargin", "netmargin", "marketcap", "sps", "payoutratio", "currentratio",
    "tangibles", "workingcapital", "sharesbas", "ncff", "ncfi", "ncfo", "ncfdiv",
    "ncfcommon", "ncfdebt", "ncfbus", "ncfinv", "capex", "sbcomp", "ncfx", "depamor",
    "ncf", "ebitusd", "epsusd", "netinccmnusd", "revenueusd", "rnd", "sgna", "gp",
    "taxexp", "intexp", "opex", "opinc", "cor", "consolinc", "shareswa", "shareswadil"
]


def divide_dataframes(dataframes):
    # Dictionary to hold divided dataframes
    divided_dfs = {}
    keys = ["bpt_sigD1", "bpt_sigD2", "bpt_sigD3", "LL1t_sigD1", "LL1t_sigD2", "LL1t_sigD3", "LL2t_sigD1", "LL2t_sigD2", "LL2t_sigD3"]
    # Define the 9 keys for each dataframe
    # keys = [
    #     "bp_time_sig_D1",
    #     "bp_time_sig_D2",
    #     "bp_time_sig_D3",
    #     "LL1_time_sig_D1",
    #     "LL1_time_sig_D2",
    #     "LL1_time_sig_D3",
    #     "LL2_time_sig_D1",
    #     "LL2_time_sig_D2",
    #     "LL2_time_sig_D3"
    # ]

    # Length of each sub-dataframe
    chunk_length = 101

    for name, df in dataframes.items():
        # Check if the dataframe has enough rows
        if len(df) < chunk_length * len(keys):
            raise ValueError(f"{name} doesn't have enough rows to split into 9 parts.")

        # Divide the dataframe into chunks
        for i, key in enumerate(keys):
            start_idx = i * chunk_length
            end_idx = start_idx + chunk_length

            # Naming convention for the divided dataframe
            new_key = f"{name}_{key.replace(' ', '_')}"  # Replace spaces with underscores for keys
            divided_dfs[new_key] = df.iloc[start_idx:end_idx]

    return divided_dfs


def heatmap_window(df_expand, name ="expand", sel_param =20):       
    
    df_sorted = df_expand.sort_values(by='adj-r2', ascending=False)   
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted = df_sorted.iloc[:-5]
    # df_result = pd.concat([df_sorted.head(sel_param), df_sorted.tail(sel_param)])
    df_result = df_sorted.head(sel_param)  
    df_result = df_result.reset_index(drop=True)
        
    
    df = pd.DataFrame(df_result)    

    # Define categories and tagging
    categories = {
        "ecofin": ["USAADP", "USABCONF", "USABOT", "USABP", "USABR", "USACA", "USACARS", 
        "USACBBS", "USACCONF", "USACCPI", "USACF", "USACFNAI", "USACHJC", 
        "USACJC", "USACNCN", "USACOR", "USACOSC", "USACP", "USACPIC", 
        "USACPMI", "USACPPI", "USACSP", "USACU", "USADINV", "USADPINC", 
        "USADUR", "USAEHS", "USAEMPST", "USAEXPX", "USAEXVOL", "USAFACT", 
        "USAFBI", "USAFDI", "USAFER", "USAFINF", "USAFOET", "USAGAGR", 
        "USAGASSC", "USAGBVL", "USAGCP", "USAGD", "USAGFCF", "USAGGR", 
        "USAGPAY", "USAGREV", "USAGSP", "USAGYLD", "USAHSTT", "USAIMPX", 
        "USAIMVOL", "USAISMNYI", "USAJCLM", "USAJOBOFF", "USAJVAC", "USALC", 
        "USALPS", "USALUNR", "USAM0", "USAM1", "USAM2", "USAMANWG", "USAMKT", 
        "USAMORTG", "USAMP", "USAMPAY", "USANAHB", "USANATGSC", "USANFIB", 
        "USANHS", "USANLTTF", "USANMPMI", "USANO", "USAOIL", "USAOPT", 
        "USAPCEPI", "USAPFED", "USAPHS", "USAPPIC", "USAPROD", "USAPSAV", 
        "USARSM", "USARSY", "USATOT", "USATOUR", "USATVS", "USAUNR", 
        "USAUNRY", "USAWAGE"],
        
        "etf": ["IVV", "IJK", "OEF", "IWO", "IUSG", "IWV", "IWB", "ITOT", "IJJ", "IVW", "IWN", "IOO",
        "IUSV", "IYY", "IWR", "AGG", "IJH", "IVE", "IJT", "IWP", "IWM", "IJR", "DVY", "IJS",
        "IWS", "EPP", "EEM", "ILF", "EWA", "EWO", "EWK", "EWC", "EWQ", "EWG", "EWH", "EWI",
        "EWJ", "JPXN", "EWN", "EWS", "EWP", "EWD", "EWL", "EWU", "FXI", "EWZ", "EWM", "EWW",
        "EZA", "EWY", "EWT", "SHY", "IEF", "TLT", "TIP", "LQD", "IYK", "IYC", "IXC", "IYE",
        "IXG", "IYG", "IYF", "IXJ", "IBB", "IYH", "IYT", "IYJ", "IYM", "IYR", "ICF", "IGM",
        "IGV", "SOXX", "IYW", "IXP", "IYZ", "IDU"],
        
        "fdm": ["cashnequsd","netincdis" , "pe",  "divyield", "debtusd", "equityusd", "taxassets", "ppnenet", "inventory",
        "intangibles", "investments", "receivables", "accoci", "retearn", "taxliabilities",
        "deferredrev", "deposits", "payables", "assetsnc", "assets", "liabilities",
        "assetsc", "debtc", "investmentsc", "liabilitiesc", "debtnc", "investmentsnc",
        "liabilitiesnc", "ebitdausd", "ps1", "pe1", "ebt", "ev", "fcf", "fcfps",
        "grossmargin", "invcap", "bvps", "evebitda", "evebit", "de", "pb", "tbvps",
        "ps", "ebitdamargin", "netmargin", "marketcap", "sps", "payoutratio", "currentratio",
        "tangibles", "workingcapital", "sharesbas", "ncff", "ncfi", "ncfo", "ncfdiv",
        "ncfcommon", "ncfdebt", "ncfbus", "ncfinv", "capex", "sbcomp", "ncfx", "depamor",
        "ncf", "ebitusd", "epsusd", "netinccmnusd", "revenueusd", "rnd", "sgna", "gp",
        "taxexp", "intexp", "opex", "opinc", "cor", "consolinc", "shareswa", "shareswadil"]
    }

    # Function to tag variables
    def tag_variable(value):
        for category, values in categories.items():
            if value in values:
                return f"{category}_{value}"
        return value

    # Apply tagging to gene1 through gene5
    for col in ["gene1", "gene2", "gene3", "gene4", "gene5"]:
        df[col] = df[col].apply(tag_variable)

    # Reshape data for heatmap
    df_melted = df.melt(id_vars=["adj-r2"], value_vars=["gene1", "gene2", "gene3", "gene4", "gene5"],
                        var_name="gene", value_name="variable")

    # Aggregate adj-r2 by variable
    heatmap_data = df_melted.groupby("variable")["adj-r2"].mean().reset_index()

    # Prepare heatmap data
    heatmap_data = heatmap_data.pivot_table(index="variable", values="adj-r2")
     
    # Plot heatmap
    plt.figure(figsize=(8, 12))
    sns.heatmap(heatmap_data, cmap="RdBu", annot=False, fmt=".2f", linewidths=.5, annot_kws={'size': 5})
    # sns.heatmap(heatmap_data, cmap="coolwarm", annot=False, fmt=".2f", linewidths=.5, annot_kws={'size': 5})
    # sns.heatmap(heatmap_data, cmap='viridis', annot=False, fmt=".2f", linewidths=.5, annot_kws={'size': 5})
    
    
    # sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f", linewidths=.5, 
    #          annot_kws={'size': 2}, cbar_kws={'label': 'adj-r2'})
    
    
    # plt.title("Heatmap of Expanding Window's adj-r2 by Factors")
    
    if name == 'expand':
        plt.title("Heatmap of Expanding Window's adj-r2 by Factors")
    elif name == 'dyadic':
        plt.title("Heatmap of Dyadic Window's adj-r2 by Factors")
    elif name == 'slide':
        plt.title("Heatmap of Sliding Window's adj-r2 by Factors")
    else:
        plt.title(f"Heatmap of {name.capitalize()} Window's adj-r2 by Factors")
    
    plt.xlabel("adj-r2")
    plt.ylabel("Tagged Variables")
    plt.show()

    return

def heatmap_window_stock(df_expand, name ="expand", sel_param=20):       
    
    df_sorted = df_expand.sort_values(by='adj-r2', ascending=False)   
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted = df_sorted.iloc[:-5]
    # df_result = pd.concat([df_sorted.head(sel_param), df_sorted.tail(sel_param)])
    df_result = df_sorted.head(sel_param)  
    df_result = df_result.reset_index(drop=True)
        
    
    df = pd.DataFrame(df_result)    

    # Define categories and tagging
    categories = {
        "ecofin": ["USAADP", "USABCONF", "USABOT", "USABP", "USABR", "USACA", "USACARS", 
        "USACBBS", "USACCONF", "USACCPI", "USACF", "USACFNAI", "USACHJC", 
        "USACJC", "USACNCN", "USACOR", "USACOSC", "USACP", "USACPIC", 
        "USACPMI", "USACPPI", "USACSP", "USACU", "USADINV", "USADPINC", 
        "USADUR", "USAEHS", "USAEMPST", "USAEXPX", "USAEXVOL", "USAFACT", 
        "USAFBI", "USAFDI", "USAFER", "USAFINF", "USAFOET", "USAGAGR", 
        "USAGASSC", "USAGBVL", "USAGCP", "USAGD", "USAGFCF", "USAGGR", 
        "USAGPAY", "USAGREV", "USAGSP", "USAGYLD", "USAHSTT", "USAIMPX", 
        "USAIMVOL", "USAISMNYI", "USAJCLM", "USAJOBOFF", "USAJVAC", "USALC", 
        "USALPS", "USALUNR", "USAM0", "USAM1", "USAM2", "USAMANWG", "USAMKT", 
        "USAMORTG", "USAMP", "USAMPAY", "USANAHB", "USANATGSC", "USANFIB", 
        "USANHS", "USANLTTF", "USANMPMI", "USANO", "USAOIL", "USAOPT", 
        "USAPCEPI", "USAPFED", "USAPHS", "USAPPIC", "USAPROD", "USAPSAV", 
        "USARSM", "USARSY", "USATOT", "USATOUR", "USATVS", "USAUNR", 
        "USAUNRY", "USAWAGE"],
        
        "etf": ["IVV", "IJK", "OEF", "IWO", "IUSG", "IWV", "IWB", "ITOT", "IJJ", "IVW", "IWN", "IOO",
        "IUSV", "IYY", "IWR", "AGG", "IJH", "IVE", "IJT", "IWP", "IWM", "IJR", "DVY", "IJS",
        "IWS", "EPP", "EEM", "ILF", "EWA", "EWO", "EWK", "EWC", "EWQ", "EWG", "EWH", "EWI",
        "EWJ", "JPXN", "EWN", "EWS", "EWP", "EWD", "EWL", "EWU", "FXI", "EWZ", "EWM", "EWW",
        "EZA", "EWY", "EWT", "SHY", "IEF", "TLT", "TIP", "LQD", "IYK", "IYC", "IXC", "IYE",
        "IXG", "IYG", "IYF", "IXJ", "IBB", "IYH", "IYT", "IYJ", "IYM", "IYR", "ICF", "IGM",
        "IGV", "SOXX", "IYW", "IXP", "IYZ", "IDU"],
        
        "fdm": ["cashnequsd","netincdis", "pe", "divyield", "debtusd", "equityusd", "taxassets", "ppnenet", "inventory",
        "intangibles", "investments", "receivables", "accoci", "retearn", "taxliabilities",
        "deferredrev", "deposits", "payables", "assetsnc", "assets", "liabilities",
        "assetsc", "debtc", "investmentsc", "liabilitiesc", "debtnc", "investmentsnc",
        "liabilitiesnc", "ebitdausd", "ps1", "pe1", "ebt", "ev", "fcf", "fcfps",
        "grossmargin", "invcap", "bvps", "evebitda", "evebit", "de", "pb", "tbvps",
        "ps", "ebitdamargin", "netmargin", "marketcap", "sps", "payoutratio", "currentratio",
        "tangibles", "workingcapital", "sharesbas", "ncff", "ncfi", "ncfo", "ncfdiv",
        "ncfcommon", "ncfdebt", "ncfbus", "ncfinv", "capex", "sbcomp", "ncfx", "depamor",
        "ncf", "ebitusd", "epsusd", "netinccmnusd", "revenueusd", "rnd", "sgna", "gp",
        "taxexp", "intexp", "opex", "opinc", "cor", "consolinc", "shareswa", "shareswadil"]
    }

    # Function to tag variables
    def tag_variable(value):
        for category, values in categories.items():
            if value in values:
                return f"{category}_{value}"
        return value

    # Apply tagging to gene1 through gene5
    for col in ["gene1", "gene2", "gene3", "gene4", "gene5"]:
        df[col] = df[col].apply(tag_variable)

    # Reshape data for heatmap
    df_melted = df.melt(id_vars=["adj-r2", "stock_ticker"], value_vars=["gene1", "gene2", "gene3", "gene4", "gene5"],
                        var_name="gene", value_name="variable")

    # Pivot table to prepare for heatmap
    heatmap_data = df_melted.pivot_table(index="variable", columns="stock_ticker", values="adj-r2", aggfunc="mean")
     
    # Plot heatmap
    plt.figure(figsize=(8, 12))
    sns.heatmap(heatmap_data, cmap="RdBu", annot=False, fmt=".2f", linewidths=.5, annot_kws={'size': 5})
    # sns.heatmap(heatmap_data, cmap="coolwarm", annot=False, fmt=".2f", linewidths=.5, annot_kws={'size': 5})
    # sns.heatmap(heatmap_data, cmap='viridis', annot=False, fmt=".2f", linewidths=.5, annot_kws={'size': 5})
    
    
    # sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f", linewidths=.5, 
    #          annot_kws={'size': 2}, cbar_kws={'label': 'adj-r2'})
    
    
    # plt.title("Heatmap of Expanding Window's adj-r2 by Factors")
    
    if name == 'expand':
        plt.title("Heatmap of Expanding Window's adj-r2 by Factors")
    elif name == 'dyadic':
        plt.title("Heatmap of Dyadic Window's adj-r2 by Factors")
    elif name == 'slide':
        plt.title("Heatmap of Sliding Window's adj-r2 by Factors")
    else:
        plt.title(f"Heatmap of {name.capitalize()} Window's adj-r2 by Factors")
    
    plt.xlabel("adj-r2")
    plt.ylabel("Tagged Variables")
    plt.show()

    return

def onehotenconding_window_features(data, n_obs=100, sel_above=3):        
    df_sorted = data.sort_values(by='adj-r2', ascending=False)   
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted = df_sorted.iloc[:-5]
    
    # 1) select head AND tail
    # df_result = pd.concat([df_sorted.head(50), df_sorted.tail(50)])
    
    # 2) select head OR tail  # df_result = df_sorted.tail(n_obs)
    # df_result = df_sorted.head(n_obs)
    df_result = df_sorted.tail(n_obs)
    
    # # 3) select CENTER 
    # center_index = len(df_sorted) // 2  # Find the center index
    
    # # Calculate start and end indices to slice from the center
    # start_index = max(center_index - n_obs // 2, 0)
    # end_index = start_index + n_obs

    # # Extract rows from the center
    # df_result = df_sorted.iloc[start_index:end_index]
        
    
    df_result = df_result.reset_index(drop=True)
        
    
    df = pd.DataFrame(df_result)    

    # Define categories and tagging
    categories = {
        "ecofin": ["USAADP", "USABCONF", "USABOT", "USABP", "USABR", "USACA", "USACARS", 
        "USACBBS", "USACCONF", "USACCPI", "USACF", "USACFNAI", "USACHJC", 
        "USACJC", "USACNCN", "USACOR", "USACOSC", "USACP", "USACPIC", 
        "USACPMI", "USACPPI", "USACSP", "USACU", "USADINV", "USADPINC", 
        "USADUR", "USAEHS", "USAEMPST", "USAEXPX", "USAEXVOL", "USAFACT", 
        "USAFBI", "USAFDI", "USAFER", "USAFINF", "USAFOET", "USAGAGR", 
        "USAGASSC", "USAGBVL", "USAGCP", "USAGD", "USAGFCF", "USAGGR", 
        "USAGPAY", "USAGREV", "USAGSP", "USAGYLD", "USAHSTT", "USAIMPX", 
        "USAIMVOL", "USAISMNYI", "USAJCLM", "USAJOBOFF", "USAJVAC", "USALC", 
        "USALPS", "USALUNR", "USAM0", "USAM1", "USAM2", "USAMANWG", "USAMKT", 
        "USAMORTG", "USAMP", "USAMPAY", "USANAHB", "USANATGSC", "USANFIB", 
        "USANHS", "USANLTTF", "USANMPMI", "USANO", "USAOIL", "USAOPT", 
        "USAPCEPI", "USAPFED", "USAPHS", "USAPPIC", "USAPROD", "USAPSAV", 
        "USARSM", "USARSY", "USATOT", "USATOUR", "USATVS", "USAUNR", 
        "USAUNRY", "USAWAGE"],
        
        "etf": ["IVV", "IJK", "OEF", "IWO", "IUSG", "IWV", "IWB", "ITOT", "IJJ", "IVW", "IWN", "IOO",
        "IUSV", "IYY", "IWR", "AGG", "IJH", "IVE", "IJT", "IWP", "IWM", "IJR", "DVY", "IJS",
        "IWS", "EPP", "EEM", "ILF", "EWA", "EWO", "EWK", "EWC", "EWQ", "EWG", "EWH", "EWI",
        "EWJ", "JPXN", "EWN", "EWS", "EWP", "EWD", "EWL", "EWU", "FXI", "EWZ", "EWM", "EWW",
        "EZA", "EWY", "EWT", "SHY", "IEF", "TLT", "TIP", "LQD", "IYK", "IYC", "IXC", "IYE",
        "IXG", "IYG", "IYF", "IXJ", "IBB", "IYH", "IYT", "IYJ", "IYM", "IYR", "ICF", "IGM",
        "IGV", "SOXX", "IYW", "IXP", "IYZ", "IDU"],
        
        "fdm": ["cashnequsd", "netincdis" , "pe", "divyield", "debtusd", "equityusd", "taxassets", "ppnenet", "inventory",
        "intangibles", "investments", "receivables", "accoci", "retearn", "taxliabilities",
        "deferredrev", "deposits", "payables", "assetsnc", "assets", "liabilities",
        "assetsc", "debtc", "investmentsc", "liabilitiesc", "debtnc", "investmentsnc",
        "liabilitiesnc", "ebitdausd", "ps1", "pe1", "ebt", "ev", "fcf", "fcfps",
        "grossmargin", "invcap", "bvps", "evebitda", "evebit", "de", "pb", "tbvps",
        "ps", "ebitdamargin", "netmargin", "marketcap", "sps", "payoutratio", "currentratio",
        "tangibles", "workingcapital", "sharesbas", "ncff", "ncfi", "ncfo", "ncfdiv",
        "ncfcommon", "ncfdebt", "ncfbus", "ncfinv", "capex", "sbcomp", "ncfx", "depamor",
        "ncf", "ebitusd", "epsusd", "netinccmnusd", "revenueusd", "rnd", "sgna", "gp",
        "taxexp", "intexp", "opex", "opinc", "cor", "consolinc", "shareswa", "shareswadil"]
    }

    # Function to tag variables
    def tag_variable(value):
        for category, values in categories.items():
            if value in values:
                return f"{category}_{value}"
        return value

    # Apply tagging to gene1 through gene5
    for col in ["gene1", "gene2", "gene3", "gene4", "gene5"]:
        df[col] = df[col].apply(tag_variable)
            
    
    # Combine all gene columns
    all_genes = df[["gene1", "gene2", "gene3", "gene4", "gene5"]].values.ravel()
    
    # Create a OneHotEncoder instance
    encoder = OneHotEncoder(sparse_output=False)
    
    # Fit and transform the data
    one_hot_encoded = encoder.fit_transform(all_genes.reshape(-1, 1))
    
    # Get feature names
    feature_names = encoder.get_feature_names_out()
    
    # Create a new DataFrame with one-hot encoded features
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=feature_names)
    
    # Sum the occurrences of each feature
    feature_counts = one_hot_df.sum().sort_values(ascending=False)
    
    # Display the top 20 most frequent features
    # print("Top 20 most frequent features:")
    # print(feature_counts.head(30))
    
    # Select features with counts greater than 3
    selected_features = feature_counts[feature_counts > sel_above]
    
    # Display the selected features
    # print("Features with counts greater than 3:")
    # print(selected_features)        

    return selected_features

def multi_wind_selectedfeatures(df_slide, df_expand, df_dyadic, n_obs=100, sel_above=3):
    # Initialize an empty list to store the results
    results = []
    
    # Iterate through each dataframe and apply the function
    for df in [df_slide, df_expand, df_dyadic]:
        # Apply the function and store the result
        features = onehotenconding_window_features(df, n_obs, sel_above)
        # Convert the result to a pandas Series (if not already one) and append to results
        results.append(pd.Series(features))
    
    # Concatenate all the results into one Series
    combined = pd.concat(results)
    combined_sorted =combined.sort_index().index
    
    # Sort the values and drop duplicates
    x_final_output = combined_sorted.drop_duplicates()
    final_output =pd.Series(x_final_output)
    
    return final_output

def single_wind_selectedfeatures(df, n_obs=100, sel_above=3):
    # Initialize an empty list to store the results
    results = []
    
    # Iterate through each dataframe and apply the function
    
     # Apply the function and store the result
    features = onehotenconding_window_features(df, n_obs, sel_above)
        # Convert the result to a pandas Series (if not already one) and append to results    
    
    # Concatenate all the results into one Series
    features_sorted =features.sort_index().index
    
    # Sort the values and drop duplicates
    x_final_output = features_sorted.drop_duplicates()
    final_output =pd.Series(x_final_output)
    
    return final_output


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
        df.loc[start_idx:end_idx-1, "stock_ticker"] += f"_{keys[current_key_index]}"
        current_key_index += 1

    return df



def max_adjR_squared(group):
    max_idx = group['adjR_squared'].idxmax()
    max_adjR2 =group['adjR_squared'].max()
    # return group.loc[max_idx, 'stock_name'], max_adjR2
    # return pd.Series({'stock_name_tagged': group.loc[max_idx, 'stock_name'], 'max_adjR2': max_adjR2})
    return pd.Series({'stock_ticker': group.loc[max_idx, 'stock_name'], 'max_adjR2': max_adjR2})


def filter_adjR_squared(group):
    """
    Returns all rows where 'adjR_squared' > 0.1.
    
    Args:
        group (pd.DataFrame): A grouped DataFrame.
        
    Returns:
        pd.DataFrame: Filtered rows with 'adjR_squared' > 0.1.
    """
    filtered_group = group[group['adjR_squared'] > 0.1]  # Filter rows where adjR_squared > 0.1
    
    if filtered_group.empty:  # Handle case where no rows satisfy the condition
        return pd.DataFrame()  # Return an empty DataFrame
    
    filtered_group = filtered_group.drop(columns=['R_squared'], errors='ignore')  # Remove 'R_squared' column
    filtered_group = filtered_group.rename(columns={'stock_name': 'stock_ticker'})  # Rename column

    return filtered_group


# def selby_adjR_squared(group):
#     filtered_group = group[group['adjR_squared'] > 0.1]  # Filter for adjR_squared > 0.1

#     if filtered_group.empty:  # Handle case where no values satisfy the condition
#         return pd.Series({'stock_ticker': None, 'max_adjR2': None})
    
#     result_df = pd.DataFrame([filtered_group['stock_name'].values], columns=[f'stock_{i+1}' for i in range(len(filtered_group))])

#     return result_df

def select_rows_based_on_ticker(get_df_tagged, get_df_polyreg_best):
    """
    Select rows from 'get_df_tagged' where the 'stock_ticker' column matches
    the 'stock_ticker' values from 'get_df_polyreg_best'.
    
    Args:
    - get_df_tagged (pd.DataFrame): The dataframe to filter.
    - get_df_polyreg_best (pd.DataFrame): The dataframe containing stock tickers to match.
    
    Returns:
    - pd.DataFrame: Filtered rows from 'get_df_tagged'.
    """
    # Extract the list of stock tickers from the best dataframe
    stock_tickers = get_df_polyreg_best["stock_ticker"].unique()
    
    # Filter rows from get_df_tagged based on the stock tickers
    filtered_df = get_df_tagged[get_df_tagged["stock_ticker"].isin(stock_tickers)]
    
    return filtered_df


def join_dataframes_by_ticker(get_filtered_df, get_df_polyreg_best):
    """
    Joins two DataFrames on the 'stock_ticker' column.

    Args:
    - get_filtered_df (pd.DataFrame): First DataFrame to join.
    - get_df_polyreg_best (pd.DataFrame): Second DataFrame to join.

    Returns:
    - pd.DataFrame: Joined DataFrame.
    """
    # Perform an inner join on 'stock_ticker'
    joined_df = pd.merge(get_filtered_df, get_df_polyreg_best, on="stock_ticker", how="inner")
    return joined_df

def reorder_column(df, column_to_move, target_column):
    """
    Reorders a DataFrame column by moving it to appear right before a target column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_to_move (str): The column to be moved.
        target_column (str): The column before which `column_to_move` should be placed.

    Returns:
        pd.DataFrame: The DataFrame with reordered columns.
    """
    columns = list(df.columns)
    
    # Remove the column to move from its current position
    columns.pop(columns.index(column_to_move)).strip()
    
    # Insert the column to move before the target column
    columns.insert(columns.index(target_column), column_to_move)
    
    # Reorder the DataFrame columns
    return df[columns]


    
def main(portfolio_date, options):
    print(Path(__file__).stem, portfolio_date)
    
    # Ensure the portfolio_date matches the expected format
    date_regex = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
    match = date_regex.match(portfolio_date)
    assert match, "Invalid portfolio_date"

    # Define the base directory for the given portfolio date
    base_directory = Path(__file__).parent.resolve() / "portfolios" / portfolio_date
    
    joined_reordered_df_directory =base_directory
    joined_reordered_df_directory.mkdir(parents=True, exist_ok=True)
    
    results_sig_polynomial = base_directory / "4_results_sigga_polynomial"
    results_sig_polynomial.mkdir(parents=True, exist_ok=True)
    # Path(results_sig_polynomial).mkdir(parents=True, exist_ok=True)

    # Update base_directory to point to the folder where the CSV files are located
    base_directory = results_sig_polynomial

    # List of filenames to process
    files = [
        'concatenated_sigga_best_aggregate_wslide.csv',
        'concatenated_sigga_best_aggregate_wexpand.csv',
        'concatenated_sigga_best_aggregate_wdyadic.csv',
        'concatenated_sigga_polynomial_r2_tagged_wexpand.csv'
    ]

    # Define column mapping for renaming
    column_mapping = {
        '0': 'adj-r2',
        '1': 'gene1',
        '2': 'gene2',
        '3': 'gene3',
        '4': 'gene4',
        '5': 'gene5'
    }

    # Dictionary to hold processed dataframes
    dataframes = {}

    # Process each file and store in the dictionary
    for file_name in files:
        file_path = base_directory / file_name  # Construct the full file path as a Path object
        assert file_path.exists(), f"File not found: {file_path}"  # Ensure the file exists
        
        # Load the CSV file
        df = pd.read_csv(file_path, header=0)

        # Add an index column (if not already present)
        df.reset_index(inplace=False)

        # Rename columns 0 through 5
        df.rename(columns=column_mapping, inplace=True)

        # Store the modified dataframe in the dictionary
        key = file_name.split('.')[0]  # Use the file name (without extension) as the key
        dataframes[key] = df

        # Print the first few rows of the processed DataFrame for confirmation
        print(f"Processed DataFrame: {key}")
        print(df.head())

    # Access the dataframes dynamically
    df_slide = dataframes['concatenated_sigga_best_aggregate_wslide']
    df_expand = dataframes['concatenated_sigga_best_aggregate_wexpand']
    df_dyadic = dataframes['concatenated_sigga_best_aggregate_wdyadic']
    df_expand_polyreg = dataframes['concatenated_sigga_polynomial_r2_tagged_wexpand']

    # Example of analysis on one of the dataframes
    print("Summary statistics for df_slide:")
    print(df_slide.describe())
    print(df_expand.describe())
    print(df_dyadic.describe())
    print(df_expand_polyreg.describe())
    
    # total of 3 methods (bptime, LL1time, LL2time), 3 sig degrees (1,2,3) each with 101 stocks
    # Dictionary of all dataframes
    all_dfs = {
        "slide": df_slide,
        "expand": df_expand,
        "dyadic": df_dyadic,
    }
    
    # Divide each dataframe into 9 parts
    divided_dfs = divide_dataframes(all_dfs)

    # Example: Accessing one of the divided dataframes
    print("Example divided dataframe:")
    print(divided_dfs["slide_bpt_sigD1"].head())          
    
    # thoree grpahs are produced as follow: 
    # 1) heatmap_window with 50 on both head + tail heatmap_window
    # 2) heatmap_window with 100 on head heatmap_window
    # 2) heatmap_window_stocks with 100 on head heatmap_window
    
    name = {'expand', 'slide', 'dyadic'}    
    get_heatmap_window =heatmap_window(df_expand, name ="expand", sel_param=100)
    
    get_heatmap_window_stock =heatmap_window_stock(df_expand, name ="expand", sel_param=100)
    
    # get_onehotenconding_window_features =onehotenconding_window_features(df_slide, n_obs=300, sel_above=3) 
    
    # get_multi_wind_selectedfeatures =multi_wind_selectedfeatures(df_slide, df_expand, df_dyadic, n_obs=100, sel_above=4)
    # print(get_multi_wind_selectedfeatures)
    
    get_single_wind_selectedfeatures =single_wind_selectedfeatures(df_expand, n_obs=100, sel_above=3)
    print(get_single_wind_selectedfeatures)
    
    df_polyreg_sorted =df_expand_polyreg.sort_values(by='stock_name', ascending=True) 
    # Group the DataFrame into chunks of 9 rows and apply the function
    # get_df_polyreg_best = df_polyreg_sorted.groupby(np.arange(len(df_polyreg_sorted)) // 9).apply(max_adjR_squared)
    get_df_polyreg_best = df_polyreg_sorted.groupby(np.arange(len(df_polyreg_sorted)) // 9).apply(filter_adjR_squared)
    get_df_polyreg_best = get_df_polyreg_best.reset_index(drop=True)
    print(get_df_polyreg_best)
    
    
    keys = ["bpt_sigD1", "bpt_sigD2", "bpt_sigD3", "LL1t_sigD1", "LL1t_sigD2", "LL1t_sigD3", "LL2t_sigD1", "LL2t_sigD2", "LL2t_sigD3"]
    get_df_tagged = append_keys_to_stock_names(df_expand, keys)
    print(get_df_tagged)
    
    get_filtered_df = select_rows_based_on_ticker(get_df_tagged, get_df_polyreg_best)
    # Print the result
    print(get_filtered_df)
    
    
    get_joined_df =join_dataframes_by_ticker(get_filtered_df, get_df_polyreg_best)         
    # get_joined_reordered_df = reorder_column(get_joined_df, "max_adjR2", "gene1")
    get_joined_reordered_df = reorder_column(get_joined_df, "adjR_squared", "gene1")       
    # Rename the "stock_ticker" column to "stock_ticker_tagged"
    get_joined_reordered_df.rename(columns={"stock_ticker": "stock_ticker_tagged"}, inplace=True)
    # Create a new "stock_ticker" column the same as "stock_ticker_tagged" by removes the first "_" and everything after it
    get_joined_reordered_df["stock_ticker"] = get_joined_reordered_df["stock_ticker_tagged"].str.split("_").str[0]
    print(get_joined_reordered_df)
    
    file_path = joined_reordered_df_directory / "get_joined_reordered_wexpand_df.csv"
    get_joined_reordered_df.to_csv(file_path, index=False)
    print(f"DataFrame saved to: {file_path}")
    
if __name__ == "__main__":  
    options=dict()  
    main("2021-03-31", options)
    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                                                                  