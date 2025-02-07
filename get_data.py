import pandas as pd
from sklearn import preprocessing
from pathlib import Path
import os
import common
import math
import numpy as np
import re
import nmf


def get_base_scale(c):
    return 100+(c/ pow(10, math.floor(math.log10(max(abs(c)))))/1)

def get_returns(c):
    return np.diff(np.log(c))


def main(portfolio_date, options=dict()):
    print(Path(__file__).stem,portfolio_date)
    options_absolute_factor = options.get('absolute_factor', True)
    options_relative_factor = options.get('relative_factor', False)

    
    date_regex = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
    match = date_regex.match(portfolio_date)
    
    assert match, "invalid portfolio_date" 
    
    # In[1]
    """
    # Set directory to (quandl + public) data downloads from matlab code     
    # create folders and subfolers    
    # create 'security master' file that holds stocks in ivv index: 'names' + 'tickers' + 'weights' + 'industry' + 'sector'    
    # get the index IVV data, prices and levels (scaled ) from AAPL file
     
    """   
    portfolio_directory = R"C:\Users\ecoramy\Documents\MATLAB\Factor Selection v11.0.0\portfolios"
    portfolio_name ="IVV_411"
    base_directory =  Path(__file__).parent.resolve() /  "portfolios" / portfolio_date
    queued_directory = Path(portfolio_directory) / portfolio_name / "data" / portfolio_date / "factor_selection" / "queued"
    security_master_file =  Path(portfolio_directory) / portfolio_name / "data" / portfolio_date / "security_master.csv"
    security_master_file = os.path.join(portfolio_directory,portfolio_name,'data',portfolio_date,'security_master.csv')
    
    
    folders = [ R"1_dataFeed\Assets", R"1_dataFeed\Factors", R"1_DataFeed\Benchmarks\{0}".format(common.FUND_TICKER), R"2_dataProcessed\Assets",
               R"2_dataProcessed\Factors", R"2_dataProcessed\Benchmarks\{0}".format(common.FUND_TICKER)]
    
    for folder in folders:
        Path(os.path.join(base_directory, folder)).mkdir(parents=True, exist_ok=True)
    
    
    scaler = preprocessing.StandardScaler()
    
    security_master = pd.read_csv(security_master_file)
    new_security_master=security_master
    set1 = set(common.OUR_STOCKS) # are we missing any stocks in the security master?
    set2 = set(security_master['bloombergName'])
    missing = list(sorted(set1 - set2))
    
    if len(missing)>0:
        print ("Missing stocks in security_master: "+ ','.join(missing))
    
    
    new_security_master.to_csv( os.path.join(base_directory,'1_dataFeed','{0}_unique_stocks.csv'.format(common.FUND_TICKER.upper())), index=False)
       
    file = Path(queued_directory) / "AAPL.csv"
   
    levels = pd.read_csv(file)
    levels.set_index('Time', inplace=True)
    levels.index = pd.to_datetime( levels.index )
       
    # names = levels.columns
    scaled_levels =levels.apply(get_base_scale)
    returns =scaled_levels.apply(get_returns).set_index(levels.index[1:])
    standardized_returns = pd.DataFrame(scaler.fit_transform(returns), columns = returns.columns, index=levels.index[1:])
    
    index_prices = levels.iloc[: , 0]
    index_returns =standardized_returns.iloc[: , 0]
    index_levels = scaled_levels.iloc[: , 0]
    
    index_prices.to_csv(  os.path.join( base_directory , R"1_dataFeed\Benchmarks\{0}\{0}_q_levels_orig.csv".format(common.FUND_TICKER)) )
    index_returns.to_csv(  os.path.join( base_directory , R"2_dataProcessed\Benchmarks\{0}\{0}_q_returns.csv".format(common.FUND_TICKER)) )
    index_levels.to_csv(  os.path.join( base_directory , R"2_dataProcessed\Benchmarks\{0}\{0}_q_levels.csv".format(common.FUND_TICKER)) )

    # In[2]   
    """
    # get specific stocks files
    # files_grabbed = []   
    # for files in common.OUR_STOCKS:
    #     files_grabbed.extend(Path(queued_directory).glob(files + '.csv'))   
    # stock returns: standardized
    # stock_levels = pd.DataFrame(): scaled levels
    # stock_returns = pd.DataFrame(): standardized
    # get returns and levels both original and scaled for all stocks (assets) and factors
    
    """
    stock_returns_list = []
    stock_scaled_levels_list = []
    stock_prices_list = [index_prices]
    
    
    files_grabbed = Path(queued_directory).glob('*.csv')
    
    for file in files_grabbed:
        filename = Path(os.path.basename(file))
        filename_wo_ext = filename.with_suffix('')
        
        print(filename_wo_ext, end=",")
        
        
        scaler = preprocessing.StandardScaler()
        
        levels =pd.read_csv(file)
        levels.set_index('Time', inplace=True)
        levels.index =pd.to_datetime( levels.index )
        
        levels_clean =pd.DataFrame([])
        for i in range(levels.shape[1]):
            if len(levels.iloc[:,i])  -len(levels.iloc[:,i].drop_duplicates()) +1 <=math.ceil( len(levels.iloc[:,i])  *0.5):
                x = levels.iloc[:,i]                
                levels_clean =pd.concat([levels_clean, x], axis=1, join='outer', sort=False)  
            else:
                continue 
        
        levels_clean.index.set_names('Time', inplace=True)
        levels_clean.index =pd.to_datetime( levels_clean.index )     
        
        
        # if 'nmf_groups' in options:
        #     assert not options_relative_factor, "option relative_factor not compatible with nmf"
        #     nmf_factors = nmf.nmf_helper(options, levels.iloc[:,2:])
        #     levels =pd.concat( [levels.iloc[:,0:2] , levels.loc[:,nmf_factors] ], axis=1 )
            
        
        # names = levels.columns              
        scaled_levels =levels_clean.apply(get_base_scale)
        returns =scaled_levels.apply(get_returns).set_index(levels_clean.index[1:])
        standardized_returns =pd.DataFrame(scaler.fit_transform(returns), columns = returns.columns, index=levels_clean.index[1:])
        
        
        Path( os.path.join(base_directory, R"1_dataFeed\Factors\{0}".format(filename_wo_ext))).mkdir(parents=True, exist_ok=True)
        Path( os.path.join(base_directory ,R"2_dataProcessed\Factors\{0}".format(filename_wo_ext))).mkdir(parents=True, exist_ok=True)
        
                
        filter_col = [col for col in levels_clean.iloc[: , 2:] if not options.get('ignore_usa_factors', False) or not col.startswith('USA')]
        
        factor_levels = scaled_levels.iloc[: , 2:].loc[:, filter_col]             
        factor_levels_orig =levels_clean.iloc[: , 2:].loc[:, filter_col]     
        factor_levels.to_csv(os.path.join(base_directory , R"2_dataProcessed\Factors\{0}\FDM_ECOFIN_q_levels.csv".format(filename_wo_ext)))
        factor_levels_orig.to_csv(os.path.join(base_directory , R"1_DataFeed\Factors\{0}\FDM_ECOFIN_q_levels_orig.csv".format(filename_wo_ext)))
    
        
        factor_returns =  standardized_returns.iloc[: , 2:].loc[:,filter_col] 
                              
        relative_factor_returns = factor_returns.sub(index_returns,axis=0)
        relative_factor_returns.columns = [ 'rel$' + s  for s in relative_factor_returns.columns]
        
              
        if options_absolute_factor and options_relative_factor:
            final_factor_returns = pd.concat([factor_returns, relative_factor_returns], axis=1)             
        elif options_absolute_factor:
            final_factor_returns = factor_returns
        elif options_relative_factor:  
            final_factor_returns = relative_factor_returns            
        else:
            raise RuntimeError("Neither absolute_factor nor relative_factor selected")

            
        final_factor_returns.to_csv(os.path.join(base_directory , R"2_dataProcessed\Factors\{0}\FDM_ECOFIN_q_returns.csv".format(filename_wo_ext)))
        
        if options.get('relative_equity', False):
            # stock - index
            diff = standardized_returns.iloc[:,1]-standardized_returns.iloc[:,0]
            diff.name = standardized_returns.columns[1]
            stock_returns_list.append(diff)
        else:
            stock_returns_list.append(standardized_returns.iloc[:,1])
                
        stock_prices_list.append(levels_clean.iloc[:,1])
        stock_scaled_levels_list.append(scaled_levels.iloc[:,1])
        
        
    stock_returns =pd.concat(stock_returns_list,axis=1) 
    stock_scaled_levels =pd.concat(stock_scaled_levels_list,axis=1) 
    
    stock_prices =pd.concat(stock_prices_list,axis=1) 
    stock_prices.to_csv( os.path.join(base_directory , R'1_dataFeed\Assets\stock_prices.csv'))
    
    stock_scaled_levels.to_csv( os.path.join(base_directory , R'2_dataProcessed\Assets\Asset_q_levels.csv'))
    stock_returns.to_csv(os.path.join(base_directory , R'2_dataProcessed\Assets\Asset_q_returns.csv'))
       
    print("")

        
# In[0]        
if __name__ == "__main__": 
    options = dict()
    options["relative_equity"] = False
    options["absolute_factor"] = True
    options["relative_factor"] = False
    
    options["ignore_usa_factors"] = False
    options["nmf_groups"] = "nmf_groups.csv"
    options["nmf_group_size"] =  "nmf_group_size.csv"
   
    main("2021-03-31",options)