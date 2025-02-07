# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 23:29:22 2024

@author: ecoramy
"""

import numpy as np

def factordis(factordata, percent=20):
    #print ('len of factordata', len(factordata))
    average = factordata.mean()
    stdv = factordata.std()
    Return = np.array(factordata)
    NReturn = -np.array(factordata)

    # In[1] :negative tail
    TailSample = NReturn[NReturn > max(0, np.percentile(NReturn, 100 - percent))]
    LenSample = len(NReturn)
    LenTail = len(TailSample)

    TailProb = 1 - (np.arange((LenSample - LenTail + 1), LenSample + 1) - 0.5) / LenSample
    OrderTail = np.sort(TailSample)
    OrderLogTail = np.log(OrderTail)
    # OrderLogTail =OrderTail
    OrderLogTailProb = np.log(TailProb)
    A = np.vstack([OrderLogTail, np.ones(len(OrderLogTail))]).T
    slope, c = np.linalg.lstsq(A, OrderLogTailProb, rcond=-1)[0]
    nConstant = np.exp(c)
    nPowerLaw = -slope

    # In[2] :positive tail
    UpTailSample = Return[Return > max(0, np.percentile(Return, 100 - percent))]
    UpLenSample = len(Return)
    LenUpTail = len(UpTailSample)

    UpTailProb = 1 - (np.arange((UpLenSample - LenUpTail + 1), UpLenSample + 1) - 0.5) / UpLenSample
    OrderUpTail = np.sort(UpTailSample)
    OrderLogUpTail = np.log(OrderUpTail)
    # OrderLogUpTail = OrderUpTail
    OrderLogUpTailProb = np.log(UpTailProb)
    UpA = np.vstack([OrderLogUpTail, np.ones(len(OrderLogUpTail))]).T
    Upslope, Upc = np.linalg.lstsq(UpA, OrderLogUpTailProb, rcond=-1)[0]
    upConstant = np.exp(Upc)
    upPowerLaw = -Upslope

    # In[3] : VaR pareto for large CL and hist for all other CL, 
    VaR99ptl_pareto = -(nConstant / (1 - 0.99)) ** (1 / nPowerLaw)
    VaR99ptl = (VaR99ptl_pareto)
    
    VaR84ptl = np.percentile(factordata, 16)
    VaR50ptl = np.percentile(factordata, 50)
    VaR16ptl = np.percentile(factordata, 84)
    
    VaR1ptl_pareto = (upConstant / (1 - 0.99)) ** (1 / upPowerLaw)
    VaR1ptl = (VaR1ptl_pareto)
    
    VaR = np.array([VaR99ptl, VaR84ptl, VaR50ptl, VaR16ptl, VaR1ptl])
    
    up_tail_threshold = upConstant + upPowerLaw*np.mean(UpTailSample)
    down_tail_threshold = nConstant + nPowerLaw*np.mean(TailSample)
    
    ThD =np.array([up_tail_threshold, down_tail_threshold])
        
    #return {'VaR': VaR, 'mean': average, 'std': stdv, 'up_tail_threshold':up_tail_threshold,'down_tail_threshold':down_tail_threshold }
    return {'VaR': VaR, 'mean': average, 'std': stdv, 'ThD':ThD }

