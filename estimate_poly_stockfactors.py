"""
Created on Thu Jun 27 14:20:21 2024

@author: ecoramy
"""
import pandas as pd
import numpy as np
# import PM_Functions_adj_M_3Y_new as pm
from pathlib import Path
import common
import os
import re
from sklearn.model_selection import KFold  
from adj_hermite import adj_hermite, adj_hermite_oos
from copy import deepcopy
import compressed_pickle
import scipy.stats as statsf
from adj_hermite import adj_hermite_oos
from adj_hermite import adj_hermite
from sklearn import preprocessing

def main(portfolio_date, options):
    print(Path(__file__).stem,portfolio_date)
    
    date_regex = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
    match = date_regex.match(portfolio_date)
    
    assert match, "invalid portfolio_date" 
    
    base_directory = Path(__file__).parent.resolve() / "portfolios" / portfolio_date 

    market = common.FUND_TICKER
    level = 'Stock'
    as_of_date = "AsOf" 
    order = 4
    pen_lbd_cv=np.arange(0.1, 10.50, 0.5)
    
    result_folder = os.path.join( base_directory, R"5_result_cv_Q\{0}\{1}\{2}".format(as_of_date,market,level))
    Path(result_folder).mkdir(parents=True, exist_ok=True)
    
    # mrkt_risks_levels = pd.read_csv(os.path.join( base_directory, R"2_dataProcessed\Factors\MRKT_RISK_qoq_prices.csv"), header=0, index_col=0)
    ret_mrkt_risk =pd.read_csv(os.path.join( base_directory, R"2_dataProcessed\Factors\ETF_FACTORS_qoq_returns.csv"), header=0, index_col=0)
    
    asset_ret = pd.read_csv(os.path.join( base_directory, R"2_dataProcessed\Assets\Asset_q_returns.csv"), header=0, index_col=0)

    SVaR_summary= pd.DataFrame([])
    SMean_summary= pd.DataFrame([])
    ASVaR_summary= pd.DataFrame([])
    ASMean_summary= pd.DataFrame([])
    
    common_factors = False
    M_dist = None
    M_dist_oos = None
    factor_ret = None
    H_adj_kfolds =None
    H_adj_kfolds_oos =None
    
    factors_folder = Path(base_directory) / "2_dataProcessed" / "Factors"    
    M_dist_file = factors_folder / "XtX_inv_order4_kfolds.npy"
    M_dist_oos_file = factors_folder / "XtX_inv_order4_kfolds_oos.npy"    
    factor_ret_file = factors_folder / "FDM_ECOFIN_q_returns.csv"
    factor_var_file = factors_folder / "FDM_ECOFIN_ETF_Q_Qtl_P.csv"
    factor_ThD_file = factors_folder / "FDM_ECOFIN_ETF_T_ThD_P.csv"
    H_adj_kfolds_file =factors_folder / "X_adjHer_order4_kfolds.npy"
    H_adj_kfolds_file_oos =factors_folder / "X_adjHer_order4_kfolds_oos.npy"
    
        
    
    if options.get('compressed_pickle', False):
        if M_dist_file.exists():
            M_dist=compressed_pickle.decompress_pickle(M_dist_file) # M_dist = np.load(M_dist_file, allow_pickle=True)[()]            
                    
        if M_dist_oos_file.exists():
            M_dist_oos=compressed_pickle.decompress_pickle(M_dist_oos_file)            
            
        if H_adj_kfolds_file.exists():
            H_adj_kfolds =compressed_pickle.decompress_pickle(H_adj_kfolds_file)
            
        if H_adj_kfolds_file_oos.exists():
            H_adj_kfolds_oos =compressed_pickle.decompress_pickle(H_adj_kfolds_file_oos)
             
    else:       
        if M_dist_file.exists():
            M_dist = np.load(M_dist_file, allow_pickle=True)[()]
                    
        if M_dist_oos_file.exists():
            M_dist_oos = np.load(M_dist_oos_file, allow_pickle=True)[()]
            
        if H_adj_kfolds_file.exists():
           H_adj_kfolds = np.load(H_adj_kfolds_file, allow_pickle=True)[()]
           
        if H_adj_kfolds_file_oos.exists():
           H_adj_kfolds_oos = np.load(H_adj_kfolds_file_oos, allow_pickle=True)[()]
             
    if factor_ret_file.exists():
        factor_ret = pd.read_csv(factor_ret_file, header=0, index_col=0)
        
    if factor_var_file.exists():
        factor_var = pd.read_csv(factor_var_file, header=0, index_col=0)
        
    if factor_ThD_file.exists():
        factor_ThD = pd.read_csv(factor_ThD_file, header=0, index_col=0)
    
    if M_dist != None and M_dist_oos != None and H_adj_kfolds != None and H_adj_kfolds_oos != None and factor_ret != None and factor_var != None and factor_ThD != None:
        common_factors = True
        print("Using common factors")
    
        assert M_dist != None, "M_dist_file {0} not found".format(M_dist_file)
        assert M_dist_oos != None, "M_dist_oos_file {0} not found".format(M_dist_oos_file)
        assert factor_ret != None,  "factor_ret_file {0} not found".format(factor_ret_file)
        assert factor_var != None,  "factor_ret_file {0} not found".format(factor_var_file)
        assert factor_ThD != None,  "factor_ret_file {0} not found".format(factor_ThD_file)
        assert H_adj_kfolds != None,  "factor_ret_file {0} not found".format(H_adj_kfolds_file)
        assert H_adj_kfolds_oos != None,  "factor_ret_file {0} not found".format(H_adj_kfolds_file_oos)
        
    
    # for each stock
    for i, asset_ticker in enumerate(common.stock_tickers):
    # for i in range(asset_ret.shape[1]): #range(asset_ret.shape[1])
    # for i in [asset_ret.columns.get_loc('AAPL')]:
        # asset_ticker = asset_ret.columns[i]
        
        # if asset_ticker != "ADBE":
        #     continue
        
        if i: print( "," , end="") 
        print( asset_ticker, end="")
    
        
        if not common_factors:       
            asset_folder = Path(base_directory) / "2_dataProcessed" / "Factors" / asset_ticker 
            
            M_dist_file = asset_folder / "XtX_inv_order4_kfolds.npy"
            M_dist_oos_file = asset_folder / "XtX_inv_order4_kfolds_oos.npy" 
            factor_ret_file =  asset_folder / "FDM_ECOFIN_q_returns.csv"
            factor_var_file =  asset_folder / "FDM_ECOFIN_ETF_Q_Qtl_P.csv"
            factor_ThD_file =  asset_folder / "FDM_ECOFIN_ETF_T_ThD_P.csv"
            H_adj_kfolds_file = asset_folder / "X_adjHer_order4_kfolds.npy"
            H_adj_kfolds_file_oos = asset_folder / "X_adjHer_order4_kfolds_oos.npy"
            
            if not factor_ret_file.exists():                            
                print(f"\n Warning: File not found - {factor_ret_file}. Skipping to next ticker.")
                continue  # Skip to the next iteration
            
            
            M_dist = np.load(M_dist_file, allow_pickle=True)[()]
            M_dist_oos = np.load(M_dist_oos_file, allow_pickle=True)[()]
            ret_factor = pd.read_csv(factor_ret_file, header=0, index_col=0)
            factor_var = pd.read_csv(factor_var_file, header=0, index_col=0)
            factor_ThD = pd.read_csv(factor_ThD_file, header=0, index_col=0)
            H_adj_kfolds =np.load(H_adj_kfolds_file, allow_pickle=True)[()]
            H_adj_kfolds_oos =np.load(H_adj_kfolds_file_oos, allow_pickle=True)[()]
            
            
        all_FandA = []
        coef_acc = []
        stats_acc = []
        
        ret_asset_series = asset_ret[asset_ticker].dropna()
        
        asset_folder = os.path.join( result_folder , asset_ticker )
        
        Path(asset_folder).mkdir(parents=True, exist_ok=True)
        
        best_sigga_polyreg_file = Path(base_directory) / "get_joined_reordered_wexpand_df.csv"
        best_sigga_polyreg_df =pd.read_csv(best_sigga_polyreg_file, header=0, index_col=None)
        
    
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
        factor_sigga = best_sigga_polyreg_df[best_sigga_polyreg_df['stock_ticker'] == asset_ticker]
        # Extract the gene column names as a list
        gene_columns =factor_sigga[['gene1', 'gene2', 'gene3', 'gene4', 'gene5']].values.flatten().tolist()
        gene_columns_unique = list(set(gene_columns))
        # Ensure the column names exist in the temp dataframe
        # gene_columns = [col for col in gene_columns if col in temp.columns]
        gene_columns_unique = [col for col in gene_columns_unique if col in temp.columns]
        # factor_ret =temp[gene_columns]
        factor_ret =temp[gene_columns_unique]
        
        # for each factor
        for j in range(factor_ret.shape[1]): 
        # for j in [factor_ret.columns.get_loc('pb')]: 

            factor_ret_series = factor_ret.iloc[:, j].dropna()
            factor_ret_series.index = pd.to_datetime(factor_ret_series.index)
            ret_asset_series.index = pd.to_datetime(ret_asset_series.index)
            tempp = pd.concat([factor_ret_series, ret_asset_series], axis=1, join='inner')
            all_FandA.append(tempp)
            
            
        
        for FandA in all_FandA: # factor and asset
            gcv = poly_regress_cv( FandA, M_dist, M_dist_oos, H_adj_kfolds, H_adj_kfolds_oos, order,pen_lbd_cv)
            if gcv == None:
                print("(skipping\u2011R)", end="") # "\u2011" is a non breaking hyphen
                continue
            coef, stats = gcv
            coef_acc.append(coef)
            stats_acc.append(stats)
                        
        if len(coef_acc)==0:
            print("(skipping\u2011F)", end="") # "\u2011" is a non breaking hyphen
            continue
                              

        # results obtained after regularization
        coef_all = pd.concat(coef_acc, axis=1, join='outer', sort=True)      
        stats_all = pd.concat(stats_acc, axis=1, join='outer', sort=True)
        stats_all.sort_values(by=['StDError'], ascending=True, axis=1, inplace=True)
        stats_all_sel =stats_all.loc[:,stats_all.loc["R_2"] >-20.0 ] # second screening; first one (cv_min)  
                          
            
        number_of_factors = len(stats_all.columns)
        number_of_factors_sel = len(stats_all_sel.columns)
        # number_of_ThD =len(factor_ThD.columns)
        factor_candidate = pd.Series(stats_all_sel.columns[:number_of_factors_sel], np.arange(number_of_factors_sel))
        # factor_candidate_ThD = pd.Series(factor_ThD.columns[:number_of_ThD], np.arange(number_of_ThD))

        window_dates=list(reversed(ret_asset_series.index[-1:]))      
        
        SVaR_name= pd.DataFrame([])
        DF= pd.DataFrame(0, index=window_dates, columns=['DominantFactor'], dtype='object')
        SVaR= pd.DataFrame(0, index=window_dates, columns=[asset_ticker], dtype='float64')
        SMean= pd.DataFrame(0, window_dates, columns=[asset_ticker], dtype='float64')
        ASVaR= pd.DataFrame(0, window_dates, columns=[asset_ticker], dtype='float64')
        ASMean= pd.DataFrame(0, window_dates, columns=[asset_ticker], dtype='float64')
                      

        fq_name = list(map(lambda xxx: [xxx + '_Q' + str(h) for h in [1, 16, 50, 84, 99]], factor_candidate))
        fq_name_ThD = list(map(lambda xxx: [xxx + '_T' + str(h) for h in [1, 2]], factor_candidate))
        fq_list =stats_all_sel.loc["fold"].tolist()
                
        # SVaR_list = list(map(lambda xxx, yyy : adj_hermite_oos(factor_var.loc[0, xxx], order) @ np.array(coef_all[yyy]).T, fq_name,factor_candidate ))
        SVaR_list = list(map(lambda xxx, yyy, zzz, www : adj_hermite(factor_var.loc[www, xxx], order, factor_ThD.loc[www, zzz]) \
                              @ np.array(coef_all[yyy]).T, fq_name,factor_candidate,fq_name_ThD, fq_list ))

        selection = pd.DataFrame(0, index=window_dates, columns=["Pre_Selection"])
        selection.index.name='Time'
        selection = selection.astype(str)  # Convert the entire DataFrame to string
        selection.iloc[0, :] = '/'.join(factor_candidate)
        date_name = window_dates[0]

        
        SVaR_min_name = list(map(lambda xxx, yyy: yyy[np.argmin(xxx)], SVaR_list, fq_name))
        SVaR_min = list(map(lambda xxx: np.min(xxx), SVaR_list))
        SVaR_min_df = pd.DataFrame(SVaR_min, columns=[date_name], index=SVaR_min_name)
        
        SMean_list = list(map(lambda xxx: np.mean(xxx), SVaR_list))
        SMean_list_df = pd.DataFrame(SMean_list, columns=[date_name], index=factor_candidate)

        SVaR_line= pd.DataFrame('/'.join(SVaR_min_name), columns=['SVaR_name'], index=pd.Index([date_name],  name='Time'))
        SVaR_name= pd.concat([SVaR_name, SVaR_line], axis=0); SVaR_name.index.name='Time'

        SVaR.loc[date_name, :] =SVaR_min_df.min().iloc[0]; SVaR.index.name='Time'
        
        DF.loc[date_name, :] =SVaR_min_df.idxmin().iloc[0]; DF.index.name='Time'
        
        SMean.loc[date_name, :] = SMean_list_df.loc[SVaR_min_df.idxmin().iloc[0].split('_')[0], date_name]; SMean.index.name='Time'                    
        
        ASVaR.loc[date_name, :] =SVaR_min_df.mean().iloc[0]; ASVaR.index.name='Time'
        
        ASMean.loc[date_name, :] = SMean_list_df.mean().iloc[0]; ASMean.index.name='Time'
               
        # Saving results
        coef_all.to_csv(  os.path.join( asset_folder ,'coef_all.csv'))
        stats_all.to_csv(os.path.join( asset_folder , 'stats_all.csv'))
        selection.to_csv(os.path.join( asset_folder , 'selection.csv'))
        
        SVaR_name.to_csv(os.path.join( asset_folder , 'Sname.csv'))
        DF.to_csv(os.path.join( asset_folder ,'DominantFactor.csv'))
        SVaR.to_csv(os.path.join( asset_folder , 'SVaR.csv'))        
        SMean.to_csv(os.path.join( asset_folder ,'SMean.csv'))
        ASVaR.to_csv(os.path.join( asset_folder ,'ASVaR.csv'))
        ASMean.to_csv(os.path.join( asset_folder ,'ASMean.csv'))

        SVaR_summary= pd.concat([SVaR_summary, SVaR], axis=1, join='outer', sort=True)
        SMean_summary= pd.concat([SMean_summary, SMean], axis=1, join='outer', sort=True)
        ASVaR_summary= pd.concat([ASVaR_summary, ASVaR], axis=1, join='outer', sort=True)
        ASMean_summary= pd.concat([ASMean_summary, ASMean], axis=1, join='outer', sort=True)    

     
    # In[0] final results
    SVaR_summary.to_csv(os.path.join( result_folder ,'SVaR_summary.csv'))
    SMean_summary.to_csv(os.path.join( result_folder , 'SMean_summary.csv'))
    ASVaR_summary.to_csv(os.path.join( result_folder , 'ASVaR_summary.csv'))
    ASMean_summary.to_csv(os.path.join( result_folder , 'ASMean_summary.csv'))

    print("")

    # In[0] 
def poly_regress_cv(data, M_dist, M_dist_oos, H_adj_kfolds, H_adj_kfolds_oos, order=3, pen_lbd=np.arange(0.1, 10.50, 0.5)):


    # remove 1 most recent data point
    data = data.iloc[:-1, :]
    
    n_splits=5
    kf = KFold(n_splits=n_splits)   
  

    c = np.empty([n_splits, order + 1])
    lbd_cv = np.empty([n_splits, 1])
    lbd_idx_cv = np.empty([n_splits, 1])
    e = np.empty([int(len(data)/n_splits)  + 1, n_splits])
    e[:] = np.nan
    # r = np.empty([n_splits, 1])
    ftest = np.empty([n_splits, 1])
    rsquaredadj =np.empty([n_splits, 1])
    rsquared =np.empty([n_splits, 1])
    p_value = np.empty([n_splits, 1])
    bic = np.empty([n_splits, 1])
    hat = {}
    hat_oos = {}
    GCV_min = np.empty([n_splits, 1])
    std_error =np.empty([n_splits, 1])
    fold_cv =np.empty([n_splits, 1])

    X_All = np.array(data.iloc[:, 0])    
    Y_All = np.array(data.iloc[:, 1])
    
    
    factor_name = data.columns[0]
    
    for f, (train_index, test_index) in enumerate(kf.split(X_All)):
                
        M_key = "{factor_name}_{fold}".format(factor_name = factor_name, fold=f)
        
        XtX_inv_list = M_dist.get(M_key)
        if XtX_inv_list == None:
            print("factor not found:",M_key)
            return
        
        XtX_inv_list_oos = M_dist_oos.get(M_key)
        if XtX_inv_list_oos == None:
            print("oos factor not found:",M_key)
            return
        
        
        H_adj_kfolds_list = H_adj_kfolds.get(M_key)
        if (all(H_adj_kfolds_list.tolist()) == False): 
            print("hermite adj kfold(s) matrix not found:",M_key)
            return
        
        H_adj_kfolds_list_oos = H_adj_kfolds_oos.get(M_key)
        if (all(H_adj_kfolds_list_oos.tolist()) == False): 
            print("hermite adj kfold(s) matrix not found:",M_key)
            return
                        
        X = H_adj_kfolds_list
        Y = Y_All[train_index]  
        
        # atemp =data.iloc[test_index, 0]
        # X_out_of_sample =adj_hermite_oos(atemp, order)
        test_size_oos =len( test_index )  
        X_oos = H_adj_kfolds_list_oos
        X_out_of_sample =X_oos
        Y_out_of_sample=Y_All[test_index]               
                

        # print (len(X_All_her))
        c_list = map(lambda xxx: xxx @ X.T @ Y, XtX_inv_list)
        c_list = [item for item in c_list]

        FiM_list = map(lambda xxx: X @ xxx @ X.T, XtX_inv_list) # fisher information matrix
        FiM_list = [item for item in FiM_list]

        FiM_list_oos = map(lambda xxx: X_out_of_sample @ xxx @ X_out_of_sample.T, XtX_inv_list_oos)
        FiM_list_oos = [item for item in FiM_list_oos]
        
        FiM_trace_list = map(lambda xxx: xxx.diagonal().sum(), FiM_list)
        FiM_trace_list = [item for item in FiM_trace_list]

        FiM_trace_list_oos = map(lambda xxx: xxx.diagonal().sum(), FiM_list_oos)
        FiM_trace_list_oos = [item for item in FiM_trace_list_oos]

        Y_fit_list = map(lambda xxx: X_out_of_sample @ xxx, c_list)
        Y_fit_list = [item for item in Y_fit_list]
                
        GCV_list = map(lambda xxx, yyy: np.square((Y_out_of_sample - xxx) / (1 - yyy / test_size_oos )).mean(), Y_fit_list, FiM_trace_list_oos)
        GCV_list = [item for item in GCV_list]
        
        error_list = map(lambda xxx: (Y_out_of_sample - xxx), Y_fit_list)
        error_list = [item for item in error_list]
        
        sse_list = map(lambda xxx: (Y_out_of_sample - xxx).T @ (Y_out_of_sample - xxx), Y_fit_list)
        sse_list = [item for item in sse_list]
        
        ssr_list = map(lambda xxx: xxx.T @ xxx, Y_fit_list)
        ssr_list = [item for item in ssr_list]
        
        muu            =Y_out_of_sample.mean()
        sse_df1        =test_size_oos -order  -1
        sse_df2        =test_size_oos -order  
        sst_df         =test_size_oos -1
        ssr_df         =order -1 
        
        sst =( Y_out_of_sample ).T @ ( Y_out_of_sample ) 
        sst_muu =( Y_out_of_sample -muu).T @ ( Y_out_of_sample -muu) 
        
        rsquared_list = map(lambda xxx: 1 -( (Y_out_of_sample - xxx).T @ (Y_out_of_sample - xxx) ) /( (Y_out_of_sample - muu).T @ (Y_out_of_sample - muu) ), Y_fit_list)
        rsquared_list = [item for item in rsquared_list]
        
        rsquaredadj_list = map(lambda xxx: 1 -( ((Y_out_of_sample - xxx).T @ (Y_out_of_sample - xxx))/sse_df1 ) /( ((Y_out_of_sample -muu).T @ (Y_out_of_sample -muu))/sst_df ), Y_fit_list)
        rsquaredadj_list = [item for item in rsquaredadj_list]
        
        ftest_list = map(lambda xxx: ( (xxx.T @ xxx)/order ) /( ((Y_out_of_sample -xxx).T @ (Y_out_of_sample -xxx))/sse_df1 ), Y_fit_list)
        ftest_list = [item for item in ftest_list]
        
        pvalue_list = map(lambda xxx: 1 -statsf.f.cdf(xxx, order, sse_df1), ftest_list)
        pvalue_list = [item for item in pvalue_list]
        
        bic_list = map(lambda xxx: test_size_oos + test_size_oos  * np.log(2 * np.pi) + test_size_oos * np.log(np.square(xxx).mean()) + np.log(test_size_oos) * (order + 1), error_list)
        bic_list = [item for item in bic_list]
        
        lbd_idx = GCV_list.index(min(GCV_list))

        coef = c_list[lbd_idx]
        Y_fit = Y_fit_list[lbd_idx]
        cv_min = GCV_list[lbd_idx]

        c[f] = coef
        lbd_cv[f] = pen_lbd[lbd_idx]
        lbd_idx_cv[f] = lbd_idx
        e[0:len(Y_out_of_sample), f] = Y_out_of_sample - Y_fit  #e =e[~np.isnan(e)]
        my_e =e[:,f]
        my_e =my_e[ ~np.isnan(my_e) ]
        sum_squares_err =sum( ( my_e )**2 ) #np.sqrt(my_e.T @ my_e/sse_df2)           # SSE
        sum_squares_reg = Y_fit.T @ Y_fit                                    # SSR      
        sum_squares_tot =( Y_out_of_sample -muu ).T @ ( Y_out_of_sample -muu)# TSS 
        rsquared[f] = 1 - ((Y_out_of_sample - Y_fit).T @ (Y_out_of_sample - Y_fit)) / ((Y_out_of_sample - Y_out_of_sample.mean()).T @ (Y_out_of_sample - Y_out_of_sample.mean()))
        rsquaredadj[f] =1 -( sum_squares_err/sse_df1 )/( sum_squares_tot/sst_df )
        ftest[f]       =( sum_squares_reg/order ) /( sum_squares_err/sse_df1)
        p_value[f] = 1 -statsf.f.cdf(ftest[f], order, test_size_oos -order -1) # vs. excel =1 -stats.f.cdf(163.207176122972, 2, 7)
        bic[f] = test_size_oos + test_size_oos  * np.log(2 * np.pi) + test_size_oos * np.log(np.square(my_e).mean()) + np.log(test_size_oos) * (order + 1)
        std_error[f] =np.sqrt(sum_squares_err /sse_df2)
        fold_cv[f] =f
        
        
        hat_name = data.columns[1] + '_' + data.columns[0] + str(f) # data.index[window[-1]]
        hat[hat_name] = XtX_inv_list[lbd_idx]
        hat_oos[hat_name] = XtX_inv_list_oos[lbd_idx]
        GCV_min[f] = cv_min
       
    GCV_min2 = deepcopy( GCV_min)
    # GCV_min2[rsquared<0] = np.inf
    best = np.array(GCV_min2).argmin()
    
    # if rsquared.all() < 0: # this is selecting the results with negative r2
    #     return
    
    coefficient = pd.DataFrame(c, columns=[data.columns[0] + '_b' + str(i) for i in range(order + 1)],index=np.arange(n_splits))
    lbd = pd.DataFrame(lbd_cv, columns=[data.columns[0] + '_' + 'lbd'],index=np.arange(n_splits))
    lbd_index = pd.DataFrame(lbd_idx_cv, columns=[data.columns[0] + '_' + 'lbdidx'], index=np.arange(n_splits))
    R_2 = pd.DataFrame(rsquared, columns=[data.columns[0] + '_' + 'R_2'], index=np.arange(n_splits))
    R_2_adj = pd.DataFrame(rsquaredadj, columns=[data.columns[0] + '_' + 'R_2_adj'], index=np.arange(n_splits))
    pvalue = pd.DataFrame(p_value, columns=[data.columns[0] + '_' + 'pvalue'], index=np.arange(n_splits))
    BIC = pd.DataFrame(bic, columns=[data.columns[0] + '_' + 'BIC'], index=np.arange(n_splits))
    GCV = pd.DataFrame(GCV_min, columns=[data.columns[0] + '_' + 'GCV'], index=np.arange(n_splits))
    standard_error = pd.DataFrame(std_error, columns=[data.columns[0] + '_' + 'StDError'], index=np.arange(n_splits))
    best_fold = pd.DataFrame(fold_cv, columns=[data.columns[0] + '_' + 'fold'], index=np.arange(n_splits))
    
    
    best_coefficient = coefficient.iloc[best,:]
    best_coefficient.index = ['b' + str(i) for i in range(order + 1)]
    best_coefficient.name = data.columns[0]
    
    # best_StD_error = np.sqrt( (e[:,best].T@e[:,best])/sse_df2 )
   
    # best_results = pd.Series([ lbd.iloc[best][0], lbd_index.iloc[best][0], R_2.iloc[best][0], pvalue.iloc[best][0], BIC.iloc[best][0], GCV.iloc[best][0], best_StD_error, best ], 
    #                             index=['lbd','lbdidx', 'R_2', 'P_Value', 'BIC', 'GCV','error','fold'], name = data.columns[0])
    
    best_results = pd.Series([ lbd.iloc[best].iloc[0], lbd_index.iloc[best].iloc[0], R_2.iloc[best].iloc[0], pvalue.iloc[best].iloc[0], 
                              BIC.iloc[best].iloc[0], GCV.iloc[best].iloc[0], standard_error.iloc[best].iloc[0], best_fold.iloc[best].iloc[0] ], 
                                index=['lbd','lbdidx', 'R_2', 'P_Value', 'BIC', 'GCV','StDError','fold'], name = data.columns[0])
    

    return [best_coefficient, best_results]





if __name__ == '__main__':
    main("2021-03-31" )

    # my_s =adj_hermite(factor_var.loc[0, ["accoci_Q1","accoci_Q16","accoci_Q50","accoci_Q84","accoci_Q99"]], 4, factor_ThD.loc[0, ["accoci_T1","accoci_T2"]])
    # my_g =adj_hermite(factor_var.loc[0, ['USAWINV_Q1', 'USAWINV_Q16', 'USAWINV_Q50', 'USAWINV_Q84', 'USAWINV_Q99']], 4, factor_ThD.loc[0, ["USAWINV_T1","USAWINV_T2"]])
