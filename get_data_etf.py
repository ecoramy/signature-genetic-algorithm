"""
Name    : data_getcreateExport
Author  : Ramy Sukarieh
Contact : ecoramy@gmail.com
Time    : 5/2/2023
Desc    : data, get create write files folders
"""

import pandas as pd
from yahoofinancials import YahooFinancials
from datetime import datetime
from pathlib import Path
import os
import math
import numpy as np
import cmondir
import re
from sklearn import preprocessing


def get_base_scale(c):
    return 100+(c/ pow(10, math.floor(math.log10(max(abs(c)))))/1)

def get_returns(c):
    return np.diff(np.log(c))

def get_stock_data(ticker: str, start_date: str, end_date: str, time_interval: str) -> pd.DataFrame:
    '''
    :param ticker: stock ticker
    :param start_date: start_end in %Y-%m-%d format
    :param end_date: end_date in %Y-%m-%d format
    :param time_interval: data frequency in either daily or weekly
    '''
    stock = YahooFinancials(ticker)
    data = stock.get_historical_price_data(start_date=start_date,
                                           end_date=end_date,
                                           time_interval=time_interval)

    dat = pd.DataFrame(data[ticker]['prices']).drop(['date', 'adjclose'], axis=1)
    dat.dropna(inplace=True)  # drop row with na to remove non trading days
    dat['formatted_date'] = pd.to_datetime(dat['formatted_date'], format="%Y-%m-%d")
    dat.set_index('formatted_date', inplace=True)
    dat.index.name = ""
    dat.columns = ['High', "Low", "Open", "Close", "Volume"]
    dat = dat[["Open", 'High', "Low", "Close", "Volume"]]
    return dat


def main(portfolio_date, options=dict()):
    print(Path(__file__).stem,portfolio_date)
    options_absolute_factor = options.get('absolute_factor', True)
    options_relative_factor = options.get('relative_factor', False)

    
    date_regex = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
    match = date_regex.match(portfolio_date)
    
    assert match, "invalid portfolio_date" 

################################################(  CREATE FOLDERS/SUBFOLDERS  )
    # Path(cmondir.baseDIR + "1_etf_data").mkdir(parents=True, exist_ok=True) 
    # Path(cmondir.baseDIR + "2_etf_processed").mkdir(parents=True, exist_ok=True)        
    base_directory = Path(__file__).parent.resolve() / "portfolios" / portfolio_date 


######################################(  GET SPECIFIC STOCKS FILES  )######################################  
    
    # etftickers=[ 'IVV','AOK','IJK','EFV','OEF','IEFA','VLUE','IWO','XT','IAGG',
    # 'IWX','ESML','IUSG','IWV','JKI','IQLT','DIVB','IPAC','IWB','JKJ',
    # 'IDV','ITOT','AOM','IJJ','EUMV','IVW','IDEV','JKD','IWN','IOO','IUSV',
    # 'IYY','IWR','ISZE','EUSA','AGG','JKK','CRBN','USRT','MTUM','SMMD',
    # 'ACWX','SUSA','IMTB','IWL','SMMV','EFAV','IJH','AOR','JKG','IEUS','IVE',
    # 'IEMG','JKE','IJT','SDG','HDV','QUAL','IWP','ACWI','LRGF','IUSB',
    # 'JKL','ESGD','IXUS','SIZE','IWM','HAWX','ESGU','ILTB','IWY','SMLF','SCZ',
    # 'IJR','AOA','JKH','IMTM','DVY','IEUR','JKF','IJS','IPFF','DGRO','USMV',
    # 'IWS','ACWV','DSI','ISTB','AMCA','IWC','EFG','IVLU','TOK','EPP','URTH',
    # 'ACWF','INTF','ISCF','EEM','HEEM','EMXC','DVYE','EMGF','ILF','AAXJ',
    # 'BKF','EEMA','ESGE','EEMV','EEMS','KSA','FM','HEWP','HEWL',
    # 'HAUD','HEWU','EWA','HEWC','EWO','EWK','EWC','EDEN','EFNL','EWQ','EWG',
    # 'EWGS','EWH','EIRL','EIS','EWI','EWJ','HEWJ','JPMV','HJPX','JPXN',
    # 'SCJ','EWN','ENZL','HEWG','ENOR','EWS','HEWI','EWP','EWD','EWL','EWU',
    # 'EWUS','FXI','MCHI','ECNS','CNYA','HEWW','HEWY','INDY','EPU','EWZ','EWZS',
    # 'ECH','ICOL','INDA','SMIN','EIDO','EWM','EWW','EPHE','EPOL','QAT',
    # 'ERUS','EZA','EWY','EWT','THD','TUR','UAE','AGT','EAGG','GBF','GVI',
    # 'NEAR','FIBR','ICSH','BYLD','SHY','IEI','IEF','TLH','TLT','AGZ','GOVT',
    # 'SHV','TFLO','CMF','MUB','NYF','MEAR','SUB','IBMI','IBMJ','IBMK'
    # ,'IBML','IBMM','IBMN','STIP','TIP','CMBS','GNMA','MBB','USIG','LQD',
    # 'SLQD','LQDI','LQDH','SUSC','SUSB','IGEB','FLOT','IGSB',
    # 'IGIB','IGLB','IGBH','QLTA','BGRN','HYG','USHY','SHYG','HYXE','FALN',
    # 'HYDB','EMHY','HYXU','GHYG','HYGH','ISHG','CEMB','LEMB','IGOV','EMB',
    # 'EMBH','IQLT','IGRO','DVYA','ICLN','HEFA',
    # 'DEFA','HSCZ','HEZU','IYLD','PFF','IECS','IEDI','RXI','KXI','IYK',
    # 'IYC','ITB','IXC','FILL','IYE','IEO','IEZ','IEFN','IXG','EUFN','IAI','IYG',
    # 'IYF','IAK','IAT','IEIH','IEHS','IXJ','IBB','IYH','IHI','IHE','EXI','IYT','ITA','IYJ',
    # 'EMIF','IFRA','IGF','MXI','IYM','WOOD','RING','PICK','SLVP','IYR','ICF','IFEU','REET',
    # 'WPS','IFGL','REM','REZ','IETC','IGM','IGN','IGV','SOXX','IYW','IRBO','IXP','IEME',
    # 'IYZ','JXI','IDU','COMT','CMDY','IAUF','VEGI','IAU','SLV','GSG' ]; 
    
    ###########################################################################
    # on dates ("2018-09-30", "2018-06-30" ) removed 'EAGG', 'IBMN', 'QLTA', 'IGBH', 
    # 'BGRN', 'IRBO', 'IAUF', 'ESML', 'IBMM', 'LQDI', 'IECS', 'IEDI', 'IEFN', 'IEIH', 'IEHS', 'IFRA', 'IETC', 'IEME', 'CMDY'
    # 'DIVB', 'USHY', 'SMMD','AMCA','EMXC', 'SUSC', 'SUSB', 'IGEB', 'HYDB',
    
    etftickers=[ 'IVV','AOK','IJK','EFV','OEF','IEFA','VLUE','IWO','XT','IAGG',
    'IWX','IUSG','IWV','JKI','IQLT','IPAC','IWB','JKJ',
    'IDV','ITOT','AOM','IJJ','EUMV','IVW','IDEV','JKD','IWN','IOO','IUSV',
    'IYY','IWR','ISZE','EUSA','AGG','JKK','CRBN','USRT','MTUM',
    'ACWX','SUSA','IMTB','IWL','SMMV','EFAV','IJH','AOR','JKG','IEUS','IVE',
    'IEMG','JKE','IJT','SDG','HDV','QUAL','IWP','ACWI','LRGF','IUSB',
    'JKL','ESGD','IXUS','SIZE','IWM','HAWX','ESGU','ILTB','IWY','SMLF','SCZ',
    'IJR','AOA','JKH','IMTM','DVY','IEUR','JKF','IJS','IPFF','DGRO','USMV',
    'IWS','ACWV','DSI','ISTB','IWC','EFG','IVLU','TOK','EPP','URTH',
    'ACWF','INTF','ISCF','EEM','HEEM','DVYE','EMGF','ILF','AAXJ',
    'BKF','EEMA','ESGE','EEMV','EEMS','KSA','FM','HEWP','HEWL',
    'HAUD','HEWU','EWA','HEWC','EWO','EWK','EWC','EDEN','EFNL','EWQ','EWG',
    'EWGS','EWH','EIRL','EIS','EWI','EWJ','HEWJ','JPMV','HJPX','JPXN',
    'SCJ','EWN','ENZL','HEWG','ENOR','EWS','HEWI','EWP','EWD','EWL','EWU',
    'EWUS','FXI','MCHI','ECNS','CNYA','HEWW','HEWY','INDY','EPU','EWZ','EWZS',
    'ECH','ICOL','INDA','SMIN','EIDO','EWM','EWW','EPHE','EPOL','QAT',
    'ERUS','EZA','EWY','EWT','THD','TUR','UAE','AGT','GBF','GVI',
    'NEAR','FIBR','ICSH','BYLD','SHY','IEI','IEF','TLH','TLT','AGZ','GOVT',
    'SHV','TFLO','CMF','MUB','NYF','MEAR','SUB','IBMI','IBMJ','IBMK'
    ,'IBML','STIP','TIP','CMBS','GNMA','MBB','USIG','LQD',
    'SLQD','LQDH','FLOT','IGSB',
    'IGIB','IGLB','HYG','SHYG','HYXE','FALN',
    'EMHY','HYXU','GHYG','HYGH','ISHG','CEMB','LEMB','IGOV','EMB',
    'EMBH','IGRO','DVYA','ICLN','HEFA',
    'DEFA','HSCZ','HEZU','IYLD','PFF','RXI','KXI','IYK',
    'IYC','ITB','IXC','FILL','IYE','IEO','IEZ','IXG','EUFN','IAI','IYG',
    'IYF','IAK','IAT','IXJ','IBB','IYH','IHI','IHE','EXI','IYT','ITA','IYJ',
    'EMIF','IGF','MXI','IYM','WOOD','RING','PICK','SLVP','IYR','ICF','IFEU','REET',
    'WPS','IFGL','REM','REZ','IGM','IGN','IGV','SOXX','IYW','IXP',
    'IYZ','JXI','IDU','COMT','VEGI','IAU','SLV','GSG' ];
    ###########################################################################
    
    # etftickers=[ 'FLOT','IGSB',
    # 'IGIB','IGLB','HYG','SHYG','HYXE','FALN',
    # 'EMHY','HYXU','GHYG','HYGH','ISHG','CEMB','LEMB','IGOV','EMB',
    # 'EMBH','IGRO','DVYA','ICLN','HEFA',
    # 'DEFA','HSCZ','HEZU','IYLD','PFF','RXI','KXI','IYK',
    # 'IYC','ITB','IXC','FILL','IYE','IEO','IEZ','IXG','EUFN','IAI','IYG',
    # 'IYF','IAK','IAT','IXJ','IBB','IYH','IHI','IHE','EXI','IYT','ITA','IYJ',
    # 'EMIF','IGF','MXI','IYM','WOOD','RING','PICK','SLVP','IYR','ICF','IFEU','REET',
    # 'WPS','IFGL','REM','REZ','IGM','IGN','IGV','SOXX','IYW','IXP',
    # 'IYZ','JXI','IDU','COMT','VEGI','IAU','SLV','GSG' ]; 
   
    # etftickers.sort()
    etfs_market =[ 'IVV','AAPL', 'IWM','EFA', 'RSP', 'GLD','SLV', 'ILTB', 'COMT', 'CMDY', 'ACWI', 'XBI']; 
    etfs_style =[ 'IWF','IWD','MTUM', 'VYM', 'USRT', 'IEMG', 'USMV', 'ACWV'] 
    # assets =[*etfs_market , *etfs_style ]
    assets =[*etftickers ]
    df =  pd.DataFrame()
        
    for ticker in assets:            
        start_date = "2001-02-28"
        # end_date = datetime.today().strftime("%Y-%m-%d")
        end_date = "2017-06-30"
        time_interval = "monthly"
        stock_df = get_stock_data(ticker, start_date=start_date, end_date=end_date, time_interval=time_interval)
        print(stock_df)
        print(ticker)
        stock_col =  stock_df ["Close"]
        df=pd.concat([df,stock_col],axis=1)
        
    df.columns = assets
    df.to_csv(os.path.join(base_directory , R"1_DataFeed\Factors\ETF_FACTORS_mom_prices.csv"))
    
    
    df['Dates'] =pd.to_datetime(df.index)
    df.set_index('Dates', inplace=True)
    dff =df.resample('QE').first()
    dff_cleaned = dff.dropna(axis=1, how='any')          
    dff_cleaned.to_csv(os.path.join(base_directory , R"2_dataProcessed\Factors\ETF_FACTORS_qoq_prices.csv"))
    
    scaler = preprocessing.StandardScaler()
    scaled_levels =dff_cleaned.apply(get_base_scale)
    returns =scaled_levels.apply(get_returns).set_index(dff_cleaned.index[1:])
    standardized_returns = pd.DataFrame(scaler.fit_transform(returns), columns = returns.columns, index=dff_cleaned.index[1:])  
    standardized_returns.to_csv(os.path.join(base_directory , R"2_dataProcessed\Factors\ETF_FACTORS_qoq_returns.csv"))
    
    
if __name__ == "__main__":  
    main("2020-12-31" )
    options=dict()    
    
 
    
    