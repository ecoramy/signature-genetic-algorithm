"""
Created on Thu Jun 27 15:07:19 2024

@author: ecoramy
"""
import common
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold  
from pathlib import Path
import re
from adj_hermite import adj_hermite_oos
import omega
import compressed_pickle
from sklearn import preprocessing

# In[]
def get_XtX_inv(factor_ret, size, order, pen_lbd, cache, cache_her):

    XtX_inv = {}
    X_her_kfolds_oos = {}
    
    n_splits=5
    kf = KFold(n_splits=n_splits)

    for i in range(factor_ret.shape[1]):  
    
        try:
            # data = factor_ret_file.iloc[:, i]
            factor_name = factor_ret.columns[i]            
            
            cache_entry = None
            cache_entry_her = None
            if factor_name.startswith("USA"):
                cache_entry = cache.get(factor_name)
                cache_entry_her = cache_her.get(factor_name)
                if cache_entry != None:
                    XtX_inv=XtX_inv|cache_entry
                    X_her_kfolds_oos =X_her_kfolds_oos|cache_entry_her
                    # print("cache hit:",factor_name)
                    continue
                else:
                    # print("cache miss:",factor_name)
                    cache_entry = dict()
                    cache_entry_her = dict()

            # remove 1 most recent data point
            X_All = factor_ret.iloc[:, i]
            
          
            saved_XtX_inv = dict()  
            saved_X_her_kfolds_oos =dict()
                    
            for f, (train_index, test_index) in enumerate(kf.split(X_All)):
                
                key = "{factor_name}_{fold}".format(factor_name = factor_name, fold=f) 
                
                x = X_All.iloc[test_index]
                X = adj_hermite_oos(x, order)
                
                XtX_list = list(map(lambda xxx: X.T @ X + xxx * omega.get_omega(order=order) , pen_lbd))
                R_list = list(map(lambda xxx: np.linalg.cholesky(xxx).transpose(), XtX_list))
                R_inv_list = list(map(lambda xxx: np.linalg.inv(xxx), R_list))
                XtX_inv_list = list(map(lambda xxx: xxx @ xxx.T, R_inv_list))
                X_list =X
                              
                                
                saved_XtX_inv[key] = XtX_inv_list
                saved_X_her_kfolds_oos[key] =  X_list
                
                if cache_entry != None:
                   cache_entry[key] =  XtX_inv_list
                   cache_entry_her[key] =  X_list
                   
                   
            if cache_entry != None:
                cache[factor_name] = cache_entry
                cache_her[factor_name] = cache_entry_her
            XtX_inv |= saved_XtX_inv
            X_her_kfolds_oos |= saved_X_her_kfolds_oos
        except Exception as error:          
            print("Skipping factor {0} because of exception: {1}".format(factor_name,error))                    

    return XtX_inv, X_her_kfolds_oos


# In[]
def main(portfolio_date, options):
    
    print(Path(__file__).stem,portfolio_date)
    
    date_regex = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
    match = date_regex.match(portfolio_date)
    
    assert match, "invalid portfolio_date" 
    
    base_directory =   Path(__file__).parent.resolve()  / "portfolios" / portfolio_date

    F_folder = Path( base_directory) / "2_dataProcessed" / "Factors"
    
    common_factor_file = F_folder / "common_factor_Q_Ret.csv"
    
    if common_factor_file.exists():
        print("Using common factors")
        common_factor_ret = pd.read_csv( common_factor_file , header=0, index_col=0)
        M = get_XtX_inv(common_factor_ret, size=36, order=4, pen_lbd=np.arange(0.1, 10.50, 0.5))
        np.save(  F_folder / "Q_sizeKFold_order4_oos", M )
        return
            
    cache = dict()
    cache_her = dict()
    
# In[]    
    for filename in os.listdir(F_folder):
        F_folder_stock = os.path.join(F_folder, filename)
        # checking if it is a file
        if os.path.isfile(F_folder_stock):
            continue
      
        print( Path(os.path.basename(filename)).with_suffix(''), end=",")

# In[]
    # base_directory = Path(__file__).parent.resolve() / "portfolios" / portfolio_date
    
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
        
    
    F_folder = Path(base_directory) / "2_dataProcessed" / "Factors" 
    F_ret_file = F_folder / "FUN_ECOFIN_q_returns.csv"
                      
    
        
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
       
        # if asset_ticker != "AAPL":
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
        # # factor_ret =temp[gene_columns]
        # gene_columns_unique = [col for col in gene_columns_unique if col in temp.columns]
        # factor_ret =temp[gene_columns_unique]
        factor_ret =temp
            
        
        
        F_folder_stock = os.path.join(F_folder, asset_ticker)      
# In[]
             
        # factor_ret = pd.read_csv( os.path.join( F_folder_stock , 'FUN_ECOFIN_q_returns.csv') , header=0, index_col=0)
        
        M = get_XtX_inv(factor_ret, size=36, order=4, pen_lbd=np.arange(0.1, 10.50, 0.5), cache=cache, cache_her=cache_her ) 
                
        if options.get('compressed_pickle', False):
            compressed_pickle.compressed_pickle( os.path.join( F_folder_stock, 'XtX_inv_order4_kfolds_oos'), M )
        else:
            np.save( os.path.join( F_folder_stock, 'XtX_inv_order4_kfolds_oos'), M[0] )
            np.save( os.path.join( F_folder_stock, 'X_adjHer_order4_kfolds_oos'), M[1] )







    print ("")        
if __name__ == "__main__":
    main("2021-03-31")