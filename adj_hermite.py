# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 06:47:41 2024

@author: ecoramy
"""

import numpy as np
import pandas as pd
from get_factordis import factordis

def adj_hermite(ori_x, n, factor_thresholds):
    """ Input Type 'Series': [ori_x],  [factor_thresholds]. Input Type 'int' for [n] 

      Returns:
          Array/Matrix: 
                            
      Uses:
           np.polynomial.hermite_e.hermevander
      """
    up_tail_threshold = factor_thresholds.iloc[0]
    down_tail_threshold = factor_thresholds.iloc[1]
        
    
    mx = np.empty([ori_x.shape[0], n + 1])
    flag1 = (ori_x < -down_tail_threshold)
    flag2 = (ori_x > up_tail_threshold)
    flag3 = (ori_x >= -down_tail_threshold) & (ori_x <= up_tail_threshold)
    if n == 4:
        mx[flag3, :] = np.polynomial.hermite_e.hermevander(ori_x[flag3], n)
        slope1 = np.array([0, 1, 2*(-down_tail_threshold), 3*(-down_tail_threshold)**2-3, 4*(-down_tail_threshold)**3-12*(-down_tail_threshold)])
        inc1 = np.polynomial.hermite_e.hermevander((-down_tail_threshold), n)[0,:] - slope1*(-down_tail_threshold)
        
        slope2 = np.array([0, 1, 2*up_tail_threshold, 3*up_tail_threshold**2-3, 4*up_tail_threshold**3-12*up_tail_threshold])
        inc2 = np.polynomial.hermite_e.hermevander(up_tail_threshold, n)[0,:] - slope2*up_tail_threshold
        
        mx[flag1, 0] = inc1[0] + slope1[0] * ori_x[flag1]
        mx[flag1, 1] = inc1[1] + slope1[1] * ori_x[flag1]
        mx[flag1, 2] = inc1[2] + slope1[2] * ori_x[flag1]
        mx[flag1, 3] = inc1[3] + slope1[3] * ori_x[flag1]
        mx[flag1, 4] = inc1[4] + slope1[4] * ori_x[flag1]
        
        mx[flag2, 0] = inc2[0] + slope2[0] * ori_x[flag2]
        mx[flag2, 1] = inc2[1] + slope2[1] * ori_x[flag2]
        mx[flag2, 2] = inc2[2] + slope2[2] * ori_x[flag2]
        mx[flag2, 3] = inc2[3] + slope2[3] * ori_x[flag2]
        mx[flag2, 4] = inc2[4] + slope2[4] * ori_x[flag2]
        
    elif n == 3:
        mx[flag3, :] = np.polynomial.hermite_e.hermevander(ori_x[flag3], n)
        slope1 = np.array([0, 1, 2*(-down_tail_threshold), 3*(-down_tail_threshold)**2-3])
        inc1 = np.polynomial.hermite_e.hermevander((-down_tail_threshold), n)[0,:] - slope1*(-down_tail_threshold)
        
        slope2 = np.array([0, 1, 2*up_tail_threshold, 3*up_tail_threshold**2-3])
        inc2 = np.polynomial.hermite_e.hermevander(up_tail_threshold, n)[0,:] - slope2*up_tail_threshold
        
        mx[flag1, 0] = inc1[0] + slope1[0] * ori_x[flag1]
        mx[flag1, 1] = inc1[1] + slope1[1] * ori_x[flag1]
        mx[flag1, 2] = inc1[2] + slope1[2] * ori_x[flag1]
        mx[flag1, 3] = inc1[3] + slope1[3] * ori_x[flag1]
        
        mx[flag2, 0] = inc2[0] + slope2[0] * ori_x[flag2]
        mx[flag2, 1] = inc2[1] + slope2[1] * ori_x[flag2]
        mx[flag2, 2] = inc2[2] + slope2[2] * ori_x[flag2]
        mx[flag2, 3] = inc2[3] + slope2[3] * ori_x[flag2]


    else:
        
        print("Please check the polynomial order!")

    return mx


def adj_hermite_oos(ori_x, n):
    average = ori_x.mean()
    stdv = ori_x.std()
    my_factordis ={ 'up_tail_threshold': average +1.65*stdv, 'down_tail_threshold': average +1.65*stdv }
    up_tail_threshold =abs( my_factordis["up_tail_threshold"] )
    down_tail_threshold =abs( my_factordis["down_tail_threshold"] )
         
    
    mx = np.empty([ori_x.shape[0], n + 1])
    flag1 = (ori_x < -down_tail_threshold)
    flag2 = (ori_x > up_tail_threshold)
    flag3 = (ori_x >= -down_tail_threshold) & (ori_x <= up_tail_threshold)
    if n == 4:
        mx[flag3, :] = np.polynomial.hermite_e.hermevander(ori_x[flag3], n)
        slope1 = np.array([0, 1, 2*(-down_tail_threshold), 3*(-down_tail_threshold)**2-3, 4*(-down_tail_threshold)**3-12*(-down_tail_threshold)])
        inc1 = np.polynomial.hermite_e.hermevander((-down_tail_threshold), n)[0,:] - slope1*(-down_tail_threshold)
        
        slope2 = np.array([0, 1, 2*up_tail_threshold, 3*up_tail_threshold**2-3, 4*up_tail_threshold**3-12*up_tail_threshold])
        inc2 = np.polynomial.hermite_e.hermevander(up_tail_threshold, n)[0,:] - slope2*up_tail_threshold
        
        mx[flag1, 0] = inc1[0] + slope1[0] * ori_x[flag1]
        mx[flag1, 1] = inc1[1] + slope1[1] * ori_x[flag1]
        mx[flag1, 2] = inc1[2] + slope1[2] * ori_x[flag1]
        mx[flag1, 3] = inc1[3] + slope1[3] * ori_x[flag1]
        mx[flag1, 4] = inc1[4] + slope1[4] * ori_x[flag1]
        
        mx[flag2, 0] = inc2[0] + slope2[0] * ori_x[flag2]
        mx[flag2, 1] = inc2[1] + slope2[1] * ori_x[flag2]
        mx[flag2, 2] = inc2[2] + slope2[2] * ori_x[flag2]
        mx[flag2, 3] = inc2[3] + slope2[3] * ori_x[flag2]
        mx[flag2, 4] = inc2[4] + slope2[4] * ori_x[flag2]
        
    elif n == 3:
        mx[flag3, :] = np.polynomial.hermite_e.hermevander(ori_x[flag3], n)
        slope1 = np.array([0, 1, 2*(-down_tail_threshold), 3*(-down_tail_threshold)**2-3])
        inc1 = np.polynomial.hermite_e.hermevander((-down_tail_threshold), n)[0,:] - slope1*(-down_tail_threshold)
        
        slope2 = np.array([0, 1, 2*up_tail_threshold, 3*up_tail_threshold**2-3])
        inc2 = np.polynomial.hermite_e.hermevander(up_tail_threshold, n)[0,:] - slope2*up_tail_threshold
        
        mx[flag1, 0] = inc1[0] + slope1[0] * ori_x[flag1]
        mx[flag1, 1] = inc1[1] + slope1[1] * ori_x[flag1]
        mx[flag1, 2] = inc1[2] + slope1[2] * ori_x[flag1]
        mx[flag1, 3] = inc1[3] + slope1[3] * ori_x[flag1]
        
        mx[flag2, 0] = inc2[0] + slope2[0] * ori_x[flag2]
        mx[flag2, 1] = inc2[1] + slope2[1] * ori_x[flag2]
        mx[flag2, 2] = inc2[2] + slope2[2] * ori_x[flag2]
        mx[flag2, 3] = inc2[3] + slope2[3] * ori_x[flag2]


    else:
        
        print("Please check the polynomial order!")

    return mx