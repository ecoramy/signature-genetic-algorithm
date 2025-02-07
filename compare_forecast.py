# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 22:20:25 2023

@author: ecoramy
"""
import pandas as pd
import numpy as np
import common
from sklearn.model_selection import KFold
from pathlib import Path
import os
import re
from get_factordis import factordis
import common
from adj_hermite import adj_hermite
import omega
import math
from numpy import linalg as LA



def duplicate_last_value(lst):
    """
    Takes a list and duplicates the last value once.
    
    Args:
        lst (list): The input list.
    
    Returns:
        list: The modified list with the last value duplicated.
    """
    if not lst:  # Ensure the list is not empty
        return lst
    
    lst.append(lst[-1])  # Duplicate the last value
    return lst


def get_XtX_inv(data, order=4, pen_lbd=np.arange(0.1, 10.50, 0.5), factor_thresholds ="factor_thresholds"):
    """This function does foos

      Returns:
          Tuple: 
              XtX_list : list[Array[ order+1, order+1] ],
              R_list : list[Array[ order+1, order+1] ],
              R_inv_list : list[Array[ order+1, order+1] ],
              XtX_list : list[Array[ order+1, order+1] ],
              x : Array[ len(data), order+1 ]
              
      Raises:
           numpy.linalg.LinAlgError - when Matrix is not positive definite
      """       
        
        # for i in range(data.shape[1]):
        # if len(data.iloc[:,i])  -len(data.iloc[:,i].drop_duplicates()) +1 <=math.ceil( len(data.iloc[:,i])  *0.7):
        #     x = data.iloc[:,i]                              
        #     continue 
    
    
    
    
    X = adj_hermite(data, order, factor_thresholds)

    # XtX_list = list(map(lambda xxx: X.T @ X + xxx *
    #                 (10 ** (-5)) * O, pen_lbd))
    
    XtX_list = list(map(lambda xxx: X.T @ X + xxx *
                    omega.get_omega(order=order) , pen_lbd))
    
    
    R_list = list(map(lambda xxx: np.linalg.cholesky(xxx).transpose(), XtX_list))
    
    
    
    R_inv_list = list(map(lambda xxx: np.linalg.inv(xxx), R_list))
    XtX_inv_list = list(map(lambda xxx: xxx @ xxx.T, R_inv_list))

    return (XtX_list, R_list, R_inv_list, XtX_inv_list, X)

    # In[0]
def main(portfolio_date, options):
    order = 4
    print(Path(__file__).stem, portfolio_date)

    date_regex = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
    match = date_regex.match(portfolio_date)

    assert match, "invalid portfolio_date"

    n_splits = 5
    kf = KFold(n_splits=n_splits)

    #portfolio_date = "2021-03-31"
    base_directory = Path(__file__).parent.resolve() / "portfolios" / portfolio_date

    # equity_return_file = base_directory / "2_dataProcessed" / "Assets" / "Asset_q_returns.csv"
    ret_equity_file = base_directory / "2_dataProcessed" / "Assets" / "Asset_q_returns.csv"
    # equity_ret = factor_ret = pd.read_csv(equity_return_file, index_col=0)
    ret_equity =pd.read_csv(ret_equity_file, index_col=0)

    # # use to calculate and test new factors
    # equity_ret_generate_factors = equity_ret.iloc[:-2,:] # 63
    # equity_ret_generate_factors_forecast = equity_ret.iloc[-2:-1,:] # second to last datapoint

    # used to evaluate chosen factors
    # equity_ret_test = equity_ret.iloc[:-1,:] # 64
    # equity_ret_test_forecast = ret_equity.iloc[-1:, :]  # last datapoint
    ret_equity_test_forecast = ret_equity.iloc[-1:, :]  # last datapoint

    ret_benchmark_file = base_directory / "2_dataProcessed" / "Benchmarks" / common.FUND_TICKER / (common.FUND_TICKER + "_q_returns.csv")
    ret_benchmark = pd.read_csv(ret_benchmark_file, index_col=0)

    ret_benchmark_test_forecast = ret_benchmark.iloc[-1:, :]  # last datapoint

    # benchmark_ret_test_forecast.index = equity_ret_test_forecast.index

    test_forecast = pd.concat([ret_benchmark_test_forecast, ret_equity_test_forecast], axis=1)

    test_forecast_file = base_directory / "5_result_cv_Q" / "AsOf" / common.FUND_TICKER / "Stock" / "test_forecast.csv"
    test_forecast.to_csv(test_forecast_file, index_label="date")

    F_folder = base_directory / "2_dataProcessed" / "Factors"

    # stock_average_y_fit = pd.DataFrame(float("nan"),columns=equity_ret.columns, index = [portfolio_date])
    # stock_factors = dict()
    stock_selection_criteria = pd.DataFrame(float("nan"), columns=[
                                            'num_factors', 'hit_ratio', 'forecast', 'actual', 'error', 'R2'], index=ret_equity.columns)
    # global_sum_hit_ratio = pd.DataFrame(float("nan"),columns=[portfolio_date] , index = equity_ret.columns)

    flag = False
    for i, stock_ticker in enumerate(common.stock_tickers):
    # for stock_ticker in os.listdir(F_folder):

        # if stock_ticker != "AMZN":
        #     continue
    
        # if stock_ticker != "NSC":
        #     continue
        
        # if stock_ticker == "NSC":
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
        ret_factor0 = pd.read_csv( F_folder_stock / 'FDM_ECOFIN_q_returns.csv', index_col=0 )
        ret_mrkt_risk =pd.read_csv(os.path.join( base_directory, R"2_dataProcessed\Factors\ETF_FACTORS_qoq_returns.csv"), header=0, index_col=0)
        ret_factor =pd.concat([ret_factor0, ret_mrkt_risk], axis=1, join='inner')
        factor_ThD = pd.read_csv( os.path.join( F_folder_stock , 'FDM_ECOFIN_ETF_T_ThD_P.csv') , header=0, index_col=0)    
                
        
        # get selection.csv stats_all.csv for portfolio_date
        stats_folder = base_directory / "5_result_cv_Q" / "AsOf" / common.FUND_TICKER / "Stock" / stock_ticker          
        stats_file = stats_folder / 'stats_all.csv'

        if not stats_file.exists():
            print("(skipping)", end="")
            continue

        stats_all = pd.read_csv(stats_file, index_col=0)
        selection = pd.read_csv(stats_folder / 'selection.csv', index_col=0)

        #factors = stats_all.loc[["fold","lbd"],selection.iloc[0,0].split("/") ].astype("int32")
        if stats_all.empty:
            print(" DataFrame is empty. Go back a pick a new stock...")
            continue
        # else:
        
        factors = stats_all[selection.iloc[0, 0].split("/")]

        # In[0] set up        
        
        # # remove factors with negative R2
        # if options.get('remove_factors_negative_r2', True):
        #     factors = factors.loc[:, (factors.loc['R_2'] > 0).values]

        errors_lag =[]
        hit_ratio_lag =[]
        stock_actual_lag =[]
        stock_forecast_lag =[]
        lbd_lag =[]
        lbd_lag_ndx =[]
        
        errors_nolag =[]
        hit_ratio_nolag =[]
        stock_actual_nolag =[]
        stock_forecast_nolag =[]
        lbd_nolag =[]
        lbd_nolag_ndx =[]
        x =[]

        pen_lbd = np.arange(0.1, 10.50, 0.5)

        for factor_ticker in factors.columns:
            try:
                fold = factors.loc["fold", factor_ticker].astype("int32")
    
                # loose last datapoint from both by
                # calulating splits on len(data) - 1
                train_index, test_index = list(
                    kf.split(ret_stock.iloc[0:-1]))[fold]
    
                # stock_ret_kfold = stock_ret.iloc[train_index]
                # ret_stock_lag_kfold = ret_stock.iloc[train_index]
                ret_stock_kfold = ret_stock.iloc[train_index]
                ret_stock_lag = ret_stock_kfold.iloc[1:-1]
                ret_stock_nolag =ret_stock_kfold.iloc[0:-1] 
                ret_stock_lag_unused_point =ret_stock_kfold.iloc[-1]
                ret_stock_nolag_unused_point =ret_stock.iloc[-1]                
                
                train_index = [i for i in train_index if 0 <= i < len(ret_factor)]
                train_index = list(map(int, train_index))  # Convert to a list of integers
                
                ret_factor_kfold = ret_factor.iloc[train_index, ret_factor.columns.get_loc(factor_ticker)]                
                ret_factor_lag = ret_factor_kfold.iloc[0:-2]
                ret_factor_nolag = ret_factor_kfold.iloc[0:-1]
                ret_factor_lag_unused_point = ret_factor_kfold.iloc[-2]
                # ret_factor_nolag_unused_point =ret_factor_kfold.iloc[-1]
    
                # unused_factor_datapoint_transformed = np.polynomial.hermite_e.hermevander(unused_factor_datapoint, order)
                
                # factor_name = factors.columns[0]
                factor_name = factor_ticker
                cols_ThD =[factor_name + '_T' + str(k) for k in [1, 2]]
                factor_thresholds =factor_ThD.loc[fold,cols_ThD]                 
    
                # In[0] lag estimation
                ret_factor_lag_unused_point_hermite =adj_hermite(pd.Series(ret_factor_lag_unused_point), order, factor_thresholds)
                (XtX_lag_list, R_lag_list, R_inv_lag_list, XtX_inv_lag_list, X_lag) =get_XtX_inv(ret_factor_lag, order, pen_lbd=pen_lbd, factor_thresholds =factor_thresholds )
                
                ret_stock_lagminus1 = ret_stock_lag
                if len(ret_stock_lag) -len(X_lag) ==1:
                    ret_stock_lagminus1 = ret_stock_lag[:len(ret_stock_lag)-1]
                elif len(ret_stock_lag) -len(X_lag) ==2:
                    ret_stock_lagminus1 = ret_stock_lag[:len(ret_stock_lag)-2]
                                
                c_lag_list = list(map(lambda xxx: xxx @ X_lag.T @ ret_stock_lagminus1, XtX_inv_lag_list))
                Y_fit_lag_list = list(map(lambda xxx: ret_factor_lag_unused_point_hermite @ xxx, c_lag_list))
                    
                error_lag_list =np.sqrt( (( ret_stock_lag_unused_point  -Y_fit_lag_list ).T[0])**2 )
                error_lag_lowest_ndx = np.argmin(error_lag_list, axis=0)
                error_lag = error_lag_list[error_lag_lowest_ndx]
                errors_lag.append(error_lag)
                
                stock_actual_lag.append(ret_stock_lag_unused_point)                
                stock_forecast_lag.append(Y_fit_lag_list[error_lag_lowest_ndx][0])                
                hit_ratio_lag.append(int( np.array(Y_fit_lag_list[error_lag_lowest_ndx]).T[0] * ret_stock_lag_unused_point > 0) )                                
                
                # In[0] nolag estimation
                # ret_factor_nolag_unused_point_hermite =adj_hermite(pd.Series(ret_factor_nolag_unused_point ), order, factor_thresholds)                               
                (XtX_nolag_list, R_nolag_list, R_inv_nolag_list, XtX_inv_nolag_list, X_nolag) = get_XtX_inv(ret_factor_nolag, order, pen_lbd=pen_lbd, factor_thresholds =factor_thresholds)
                
                if len(ret_stock_nolag) -len(X_nolag) ==1:
                    ret_stock_nolag = ret_stock_nolag[:len(ret_stock_nolag)-1]
                elif len(ret_stock_nolag) -len(X_nolag) ==2:
                    ret_stock_nolag = ret_stock_nolag[:len(ret_stock_nolag)-2]
                
                c_nolag_list = list(map(lambda xxx: xxx @ X_nolag.T @ ret_stock_nolag, XtX_inv_nolag_list))                          
                Y_fit_nolag_list = list(map(lambda xxx: ret_factor_lag_unused_point_hermite @ xxx, c_nolag_list))
                
                error_nolag_list =np.sqrt( (( ret_stock_lag_unused_point  -Y_fit_nolag_list ).T[0])**2 )
                error_nolag_lowest_ndx = np.argmin(error_nolag_list, axis=0)
                error_nolag = error_nolag_list[error_nolag_lowest_ndx]
                errors_nolag.append(error_nolag)
                
                stock_actual_nolag.append(ret_stock_lag_unused_point)
                stock_forecast_nolag.append(Y_fit_nolag_list[error_nolag_lowest_ndx][0])
                hit_ratio_nolag.append(int(np.array(Y_fit_nolag_list[error_nolag_lowest_ndx]) * ret_stock_lag_unused_point > 0))               
                               
                # In[0] append
                erros_lagnolag_diff =( pd.Series(errors_lag) -pd.Series(errors_nolag) ).tolist()               
                erros_lagnolag_boolean =[0 if num < 0 else 1 for num in erros_lagnolag_diff]  # 0 ==lag and 1 ==nolag                                          
                
                lbd_lag_ndx.append(error_lag_lowest_ndx)
                lbd_lag.append(pen_lbd[error_lag_lowest_ndx])
                lbd_nolag_ndx.append(error_nolag_lowest_ndx)
                lbd_nolag.append(pen_lbd[error_nolag_lowest_ndx])
            except LA.LinAlgError as e:
                print(factor_ticker, str(e))
                errors_lag.append(float("nan"))
                lbd_lag_ndx.append(float("nan"))
                lbd_lag.append(float("nan"))
                errors_nolag.append(float("nan"))
                lbd_nolag_ndx.append(float("nan"))
                lbd_nolag.append(float("nan"))                                
    

        if abs(len(lbd_lag) -len(factors.columns)) ==1:
            factors =factors.drop(factors.columns[[-1]], axis=1)
             
        
        elif abs(len(lbd_lag) -len(factors.columns)) ==2:
            factors =factors.drop(factors.columns[[-2]], axis=1)                             
                                                   
        else:           
            # add best_lambdas to factors
            new_row = pd.DataFrame([lbd_lag], columns=factors.columns, index=["lbd_lag"])        
            factors =pd.concat([factors, new_row], axis=0)
                    
            # add best_lambdas_idx to factors
            new_row = pd.DataFrame([lbd_lag_ndx], columns=factors.columns, index=["lbdidx_lag"])        
            factors =pd.concat([factors, new_row], axis=0)
            
            # add residuals to factors
            new_row = pd.DataFrame([errors_lag], columns=factors.columns, index=["error_lag"])        
            factors= pd.concat([factors, new_row], axis=0)
            
            new_row = pd.DataFrame([errors_nolag], columns=factors.columns, index=["error_nolag"])        
            factors= pd.concat([factors, new_row], axis=0)
            
            new_row = pd.DataFrame([erros_lagnolag_boolean], columns=factors.columns, index=["erros_lagnolag_boolean"])        
            factors= pd.concat([factors, new_row], axis=0)    
            
        # sort factors by residuals
        # factors = factors[factors.columns[np.argsort(factors.loc['error_lag',:])]]

        # In[0] forecast estimation                           
                
        # factors = factors.iloc[:, :5] # select best 5 factors 

        #stock_factors[stock_ticker] = factors
        factors_file = base_directory / "5_result_cv_Q" / "AsOf" / common.FUND_TICKER / "Stock" / stock_ticker / "stock_factors.csv"
        factors.to_csv(factors_file)        

        hit_ratio = []
        stock_actual = []
        stock_forecast = []
        
        hit_ratio_lagfr =[]
        stock_forecast_lagfr =[]
        hit_ratio_nolagfr =[]
        stock_forecast_nolagfr =[]       

        for factor_ticker in factors.columns:

            fold = factors.loc["fold", factor_ticker].astype("int32")

            # loose last datapoint from both by
            # calulating splits on len(data) - 1
            train_index, test_index = list(
                kf.split(ret_stock.iloc[0:-1]))[fold]
            
            ret_stock_fr_kfold = ret_stock.iloc[train_index]
            ret_stock_lagfr =ret_stock_fr_kfold.iloc[1:]           
            ret_stock_unused_point =ret_stock.iloc[-1]
            
            train_index = [i for i in train_index if 0 <= i < len(ret_factor)]
            train_index = list(map(int, train_index))  # Convert to a list of integers
            
            ret_factor_kfold = ret_factor.iloc[train_index, ret_factor.columns.get_loc(factor_ticker)]
            ret_factor_lagfr =ret_factor_kfold.iloc[0:-1] 
            ret_factor_lagfr_unused_point =ret_factor_kfold.iloc[-1]
            
            if (factors.loc["erros_lagnolag_boolean", factor_ticker].astype("int32")).any() ==0:
                
                # In[0] lagged forecast vs. no lagged forecast returns                
                factor_name = factor_ticker
                cols_ThD =[factor_name + '_T' + str(k) for k in [1, 2]]
                factor_thresholds =factor_ThD.loc[fold,cols_ThD] 
                
                ret_factor_lagfr_unused_point_hermite =adj_hermite(pd.Series(ret_factor_lagfr_unused_point ), order, factor_thresholds)                 
                (XtX_lagfr_list, R_lagfr_list, R_inv_lagfr_list, XtX_inv_lagfr_list, X_lagfr) = get_XtX_inv(ret_factor_lagfr, order, pen_lbd=[factors.loc['lbd_lag', factor_ticker]], factor_thresholds =factor_thresholds)                
                
                if len(ret_stock_lagfr) -len(X_lagfr) ==1:
                    ret_stock_lagfr = ret_stock_lagfr[:len(ret_stock_lagfr)-1]
                elif len(ret_stock_lagfr) -len(X_lagfr) ==2:
                    ret_stock_lagfr = ret_stock_lagfr[:len(ret_stock_lagfr)-2]
                
                c_lagfr_list = list(map(lambda xxx: xxx @ X_lagfr.T @ ret_stock_lagfr, XtX_inv_lagfr_list))                            
                Y_fit_lagfr_list = list(map(lambda xxx: ret_factor_lagfr_unused_point_hermite @ xxx, c_lagfr_list))
                                    
                stock_actual.append(ret_stock_unused_point)
                stock_forecast_lagfr.append(Y_fit_lagfr_list[0][0])
                hit_ratio_lagfr.append(int(np.array(Y_fit_lagfr_list[0][0]) * ret_stock_unused_point > 0))                                             
                    
            else:
                 
                ret_factor_nolagfr_unused_point_hermite =adj_hermite(pd.Series(ret_factor_lagfr_unused_point ), order, factor_thresholds)                
                (XtX_nolagfr_list, R_nolagfr_list, R_inv_nolagfr_list, XtX_inv_nolagfr_list, X_nolagfr) = get_XtX_inv(ret_factor_kfold, order, pen_lbd=[factors.loc['lbd_lag', factor_ticker]], factor_thresholds =factor_thresholds)
                
                if len(ret_stock_fr_kfold) -len(X_nolagfr) ==1:
                    ret_stock_fr_kfold = ret_stock_fr_kfold[:len(ret_stock_fr_kfold)-1]
                elif len(ret_stock_fr_kfold) -len(X_nolagfr) ==2:
                    ret_stock_fr_kfold = ret_stock_fr_kfold[:len(ret_stock_fr_kfold)-2]
                
                c_nolagfr_list = list(map(lambda xxx: xxx @ X_nolagfr.T @ ret_stock_fr_kfold, XtX_inv_nolagfr_list))                              
                Y_fit_nolagfr_list = list(map(lambda xxx: ret_factor_nolagfr_unused_point_hermite @ xxx, c_nolagfr_list))                    
                
                stock_actual.append(ret_stock_unused_point)
                # stock_forecast_nolagfr.append(Y_fit_nolagfr_list[0][0])
                stock_forecast_lagfr.append(Y_fit_nolagfr_list[0][0])
                # hit_ratio_nolagfr.append(int(np.array(Y_fit_nolagfr_list[0][0]) * ret_stock_unused_point > 0))
                hit_ratio_lagfr.append(int(np.array(Y_fit_nolagfr_list[0][0]) * ret_stock_unused_point > 0))
                           
            # In[0] lag estimation
                
            # add y forecast to factors
        new_row = pd.DataFrame([stock_actual], columns=factors.columns, index=["stock_actual"])
        factors = pd.concat([factors, new_row], axis=0)

        # add y fit to factors
        new_row = pd.DataFrame([stock_forecast_lagfr], columns=factors.columns, index=["stock_forecast"])
        factors = pd.concat([factors, new_row], axis=0)

        # add hit_ratio to factors
        new_row = pd.DataFrame([hit_ratio_lagfr], columns=factors.columns, index=["hit_ratio"])
        # factors = factors.append(new_row)
        factors = pd.concat([factors, new_row], axis=0)

        # if len(factors.columns)==0:
        #     average_Y_forecast = float("nan")

        stock_selection_criteria.loc[stock_ticker, 'hit_ratio'] = sum(factors.loc['hit_ratio', :]) /len(factors.loc['hit_ratio', :])
        stock_selection_criteria.loc[stock_ticker, 'forecast'] = factors.loc['stock_forecast', :].mean()
        stock_selection_criteria.loc[stock_ticker, 'actual'] = factors.loc['stock_actual', :].mean()
        stock_selection_criteria.loc[stock_ticker, 'error'] = abs(stock_selection_criteria.loc[stock_ticker, 'forecast']\
                                                                - stock_selection_criteria.loc[stock_ticker, 'actual'])
        stock_selection_criteria.loc[stock_ticker, 'R2'] = factors.loc["R_2", :].mean()
        # stock_selection_criteria.loc[stock_ticker, 'R2'] = factors.loc["R_2", factor_ticker].mean()
        
        stock_selection_criteria.loc[stock_ticker, 'num_factors'] = len(factors.columns)

    # stock_selection_criteria = stock_selection_criteria.sort_values(by='hit_ratio', ascending=False)
    stock_selection_criteria_file = base_directory / "5_result_cv_Q" / "AsOf" / common.FUND_TICKER / "Stock" / "stock_selection_criteria.csv"
    stock_selection_criteria.to_csv( stock_selection_criteria_file, index_label="stock")

    print("")


if __name__ == "__main__":
    options = dict()
    options["remove_factors_negative_r2"] = True
             
    main("2021-03-31", options) 
