# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:39:59 2023

@author: ecoramy
"""
# import get_sigga_wslide_v11 as get_SiGGA_wslide
# import get_sigga_wdyadic_v0
# import get_polynomial_sig_v0 as get_polynomial_sig
import get_data
import get_data_etf
import get_sigga_wslide
import get_sigga_wexpand
import get_sigga_wdyadic
import get_polynomial_sig
import get_polynomial_sig_agg
import get_polynomial_sig_heatmap
import get_factor_distribution_all
import get_inverse_factors_all
import get_inverse_factors_oos_all
import estimate_poly_stockfactors
import estimate_poly_stockfactors_b4sigga
import compare_forecast

# import warnings
# from pandas.errors import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# https://stackabuse.com/how-to-print-colored-text-in-python/

def partial_run(portfolio_date, options):    
    # get_data.main(portfolio_date, options)    
    # get_data_etf.main(portfolio_date, options)    
    # get_sigga_wslide.main(portfolio_date, options)             
    # get_sigga_wdyadic.main(portfolio_date, options)    
    get_sigga_wexpand.main(portfolio_date, options)        
    # get_polynomial_sig.main(portfolio_date, options)
    # get_polynomial_sig_agg.main(portfolio_date, options)    
    # get_polynomial_sig_heatmap.main(portfolio_date, options)        
    # get_factor_distribution_all.main(portfolio_date)    
    # get_inverse_factors_all.main(portfolio_date, options)    
    # get_inverse_factors_oos_all.main(portfolio_date, options)    
    # estimate_poly_stockfactors.main(portfolio_date, options)
    # estimate_poly_stockfactors_b4sigga.main(portfolio_date, options)  
    # compare_forecast.main(portfolio_date, options)
  
# def run(portfolio_date, options):
#     get_data.main(portfolio_date, options)
#     get_factor_distribution.main(portfolio_date)
#     get_inverse_factors.main(portfolio_date, options)
#     get_inverse_factors_oos.main(portfolio_date, options)
#     estimate_poly_stockfactors.main(portfolio_date, options)  
#     compare_forecast.main(portfolio_date, options)
  
        
    
def main():
    
    options = dict()
    # options["relative_equity"] = True
    # options["absolute_factor"] = True
    # options["relative_factor"] = True
    
    
    # options["ignore_usa_factors"] = False
    # options["nmf_groups"] = "nmf_groups.csv"
    # options["nmf_group_size"] =  "nmf_group_size.csv"
    
    options["remove_factors_negative_r2"] = True
    # options["compressed_pickle"] = False
    
    options["relative_equity"] = False
    options["absolute_factor"] = True
    options["relative_factor"] = False
    
    options["ignore_usa_factors"] = False
    #options["nmf_groups"] = "nmf_groups.csv"
    #options["nmf_group_size"] =  "nmf_group_size.csv"
    #options["nmf_corel"] =  80 
    
    partial_run("2017-06-30", options )
    partial_run("2017-09-30", options )
    partial_run("2017-12-31", options )
    # partial_run("2018-03-31", options )
    # partial_run("2018-06-30", options )
    # partial_run("2018-09-30", options )
    # partial_run("2018-12-31", options )
    # partial_run("2019-03-31", options )
    # partial_run("2019-06-30", options )   # from here i am using the correct dates on the ETF file
    # partial_run("2019-09-30", options )
    # partial_run("2019-12-31", options )
    # partial_run("2020-03-31", options )
    # partial_run("2020-06-30", options )
    # partial_run("2020-09-30", options )
    # partial_run("2020-12-31", options )
    # partial_run("2021-03-31", options )

    
    # partial_run("2021-03-31", options )
    # partial_run("2020-12-31", options )
    # partial_run("2020-09-30", options )
    # partial_run("2020-06-30", options )
    # partial_run("2020-03-31", options )
    # partial_run("2019-12-31", options )
    # partial_run("2019-09-30", options )
    # partial_run("2019-06-30", options )
    # partial_run("2019-03-31", options )
    # partial_run("2018-12-31", options )
    # partial_run("2018-09-30", options )
    # partial_run("2018-06-30", options )
    # partial_run("2018-03-31", options )
    # partial_run("2017-12-31", options )
    # partial_run("2017-09-30", options )
    # partial_run("2017-06-30", options )


if __name__ == "__main__":
    main()      

