# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 23:37:25 2024

@author: ecoramy
"""

import pandas as pd
import numpy as np
import common
from sklearn.model_selection import KFold  
from pathlib import Path
from sklearn import preprocessing
import os
import re
from get_factordis import factordis
import math



def main(portfolio_date):
    
    print(Path(__file__).stem,portfolio_date)
    
    date_regex = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
    match = date_regex.match(portfolio_date)
    
    assert match, "invalid portfolio_date"     
    

# In[]
    base_directory = Path(__file__).parent.resolve() / "portfolios" / portfolio_date
    
    # market = common.FUND_TICKER
    market = "IVV"
    level = 'Stock'
    as_of_date = "AsOf"
       
    result_sigga_polyreg = os.path.join( base_directory)
    Path(result_sigga_polyreg).mkdir(parents=True, exist_ok=True)
    
    # mrkt_risks_levels = pd.read_csv(os.path.join( base_directory, R"2_dataProcessed\Factors\MRKT_RISK_qoq_prices.csv"), header=0, index_col=0)
    ret_mrkt_risk =pd.read_csv(os.path.join( base_directory, R"2_dataProcessed\Factors\ETF_FACTORS_qoq_returns.csv"), header=0, index_col=0)
    ret_mrkt_risk =ret_mrkt_risk.drop(ret_mrkt_risk.index[-1])
    
    # asset_ret = pd.read_csv(os.path.join( base_directory, R"1_dataFeed\Assets\stock_prices.csv"), header=0, index_col=0)
    ret_equity = pd.read_csv(os.path.join( base_directory, R"2_dataProcessed\Assets\Asset_q_returns.csv"), header=0, index_col=0)
    ret_equity =ret_equity.drop(ret_equity.index[-1])
    # asset_ret_check = asset_ret[common.stock_tickers]   
    common_factors = False
    # factor_ret = None
    ret_factor = None
    
    # factors_folder = Path(base_directory) / "1_dataFeed" / "Factors" 
    # factor_ret_file = factors_folder / "FUN_ECOFIN_q_levels_orig.csv"
    
    F_folder = Path(base_directory) / "2_dataProcessed" / "Factors" 
    F_ret_file = F_folder / "FDM_ECOFIN_q_returns.csv"
                       
            
    if F_ret_file.exists():
        ret_factor = pd.read_csv(F_ret_file, header=0, index_col=0)
               
    if ret_factor != None:
        common_factors = True
        print("Using common factors")    
       
        assert ret_factor != None,  "F_ret_file {0} not found".format(F_ret_file)    
               
    
    # for each stock
    # for i in range(asset_ret.shape[1]): #range(asset_ret.shape[1])  
    # for i in [asset_ret.columns.get_loc('AAPL')]:
        # asset_ticker = asset_ret.columns[i+1]
    for i, asset_ticker in enumerate(common.stock_tickers):
        print(f"Processing ticker: {asset_ticker}")
        asset_ticker = asset_ticker
       
        # if asset_ticker != "GOOGL":
        #     continue
       
        if i: print( "," , end="")
        print( asset_ticker, end="")
    
       
        if not common_factors:      
            asset_folder = Path(base_directory) / "1_dataFeed" / "Factors" / asset_ticker
            equity_folder = Path(base_directory) / "2_dataProcessed" / "Factors" / asset_ticker
           
            # factor_ret_file =  asset_folder / "FUN_ECOFIN_q_levels_orig.csv"                      
            # factor_ret = pd.read_csv(factor_ret_file, header=0, index_col=0)
            
            F_ret_file =  equity_folder / "FDM_ECOFIN_q_returns.csv"                      
            
            if not F_ret_file.exists():                            
                print(f"\n Warning: File not found - {F_ret_file}. Skipping to next ticker.")
                continue  # Skip to the next iteration
            
            
            ret_factor = pd.read_csv(F_ret_file, header=0, index_col=0)

                        
           
        all_FandA = []
        coef_acc = []
               
        # asset_ret_series = asset_ret.iloc[:, i+1].dropna()      
        ret_asset_series = ret_equity.loc[:, asset_ticker].dropna()
        asset_folder = os.path.join( result_sigga_polyreg )
       
        Path(asset_folder).mkdir(parents=True, exist_ok=True)   
        
        
        # best_sigga_polyreg_file = Path(asset_folder) / "get_joined_reordered_wexpand_df.csv"
        # best_sigga_polyreg_df =pd.read_csv(best_sigga_polyreg_file, header=0, index_col=None)
        
        ret_factor =ret_factor.drop(ret_factor.index[-1])
        # temp = pd.concat([asset_ret_series, factor_ret, mrkt_risks_levels ], axis=1, join='inner')
        temp =pd.concat([ret_asset_series, ret_factor, ret_mrkt_risk], axis=1, join='inner')
        temp.index.name = 'Time'
        temp.index = pd.to_datetime( temp.index )
        names = temp.columns
       
        scaler = preprocessing.StandardScaler()           
       
        # scaled_levels =temp.apply(get_base_scale)
       
        # returns =scaled_levels.apply(get_returns).set_index(temp.index[1:])
        # standardized_returns0 = pd.DataFrame(scaler.fit_transform(returns), columns = returns.columns, index=temp.index[1:])
        # standardized_returns =standardized_returns0.drop(standardized_returns0.index[-1])
        standardized_returns = temp
        
        # Filter the dataframe for the selected asset
        # factor_sigga = best_sigga_polyreg_df[best_sigga_polyreg_df['stock_ticker'] == asset_ticker]
        # # Extract the gene column names as a list
        # gene_columns =factor_sigga[['gene1', 'gene2', 'gene3', 'gene4', 'gene5']].values.flatten().tolist()
        # gene_columns_unique = list(set(gene_columns))
        # # Ensure the column names exist in the temp dataframe
        # # gene_columns = [col for col in gene_columns if col in temp.columns]
        # gene_columns_unique = [col for col in gene_columns_unique if col in temp.columns]
        # # factor_ret =temp[gene_columns]
        # factor_ret =temp[gene_columns_unique]
        factor_ret =temp
        
        # Check if factor_ret is empty (contains all NaN)
        if factor_ret.empty:
            print(f"No matching columns found in 'temp' for the genes of {asset_ticker}.")
        else:
            print(factor_ret)
        
        factor = 'FDM_ECOFIN_ETF' 
        output_folder = os.path.join( base_directory, R"2_dataProcessed\Factors\{0}".format(asset_ticker))        
        # In[2] cross validation factor data
        factorP = pd.DataFrame([])
        factorT = pd.DataFrame([])
        n_splits=5
        kf = KFold(n_splits=n_splits)
        
        for i in range(factor_ret.shape[1]):
            x = factor_ret.iloc[:, i]
            df_Q = pd.DataFrame([], index=np.arange(n_splits), columns=[x.name + '_Q' + str(k) for k in [1, 16, 50, 84, 99]]) #[1, 5, 16, 50, 84, 95, 99]
            df_T = pd.DataFrame([], index=np.arange(n_splits), columns=[x.name + '_T' + str(k) for k in [1, 2]]) 
                                       
            for f, (train_index, test_index) in enumerate(kf.split(x)):
                my_factordis = factordis(x.iloc[train_index], 20)
                df_Q.loc[f, :] = my_factordis['VaR']
                df_T.loc[f, :] = my_factordis['ThD']
            
            factorP = pd.concat([factorP, df_Q], axis=1, join='outer', sort=False)
            factorT = pd.concat([factorT, df_T], axis=1, join='outer', sort=False)
            
        factorP.to_csv( os.path.join( output_folder, factor + '_Q_Qtl_P.csv'), index_label='kfold')
        factorT.to_csv( os.path.join( output_folder, factor + '_T_ThD_P.csv'), index_label='kfold')
        

# In[0]
if __name__ == "__main__":
    main("2021-03-31" )
    
    # for seq in range(5):
    #     print(seq)
