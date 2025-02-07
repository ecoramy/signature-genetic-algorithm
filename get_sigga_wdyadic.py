# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:43:32 2024

@author: ecoramy
"""

import pandas as pd
import numpy as np
import math
import random
import common
from pathlib import Path
import scipy.stats as statsf
import os
import re
from sklearn import preprocessing
from iisignature import *
from sklearn import linear_model
import numbers
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
import csv
np.random.seed(42)  # Set NumPy seed
pygad.random.seed(42)  # Set pyGAD's internal random seed


# Solve Non-Deterministic Problems
# https://pygad.readthedocs.io/en/latest/pygad_more.html#solve-non-deterministic-problems
# https://blog.derlin.ch/genetic-algorithms-with-pygad
# Hyperparameter Tuning the Random Forest in Python
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74


def main(portfolio_date, options):
    print(Path(__file__).stem,portfolio_date)
   
    date_regex = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
    match = date_regex.match(portfolio_date)
   
    assert match, "invalid portfolio_date"
   
    base_directory = Path(__file__).parent.resolve() / "portfolios" / portfolio_date
   
    # market = common.FUND_TICKER
    market = "IVV"
    level = 'Stock'
    as_of_date = "AsOf"
       
    result_SiGGA = os.path.join( base_directory, R"3_results_sigga\{0}\{1}\{2}".format(as_of_date,market,level))
    Path(result_SiGGA).mkdir(parents=True, exist_ok=True)

    # mrkt_risks_levels = pd.read_csv(os.path.join( base_directory, R"2_dataProcessed\Factors\MRKT_RISK_qoq_prices.csv"), header=0, index_col=0)
    ret_mrkt_risk =pd.read_csv(os.path.join( base_directory, R"2_dataProcessed\Factors\MRKT_RISK_qoq_returns.csv"), header=0, index_col=0)
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
    F_ret_file = F_folder / "FUN_ECOFIN_q_returns.csv"
           
             
    # if factor_ret_file.exists():
    #     factor_ret = pd.read_csv(factor_ret_file, header=0, index_col=0)
               
    # if factor_ret != None:
    #     common_factors = True
    #     print("Using common factors")    
       
    #     assert factor_ret != None,  "factor_ret_file {0} not found".format(factor_ret_file)
        
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
            
            F_ret_file =  equity_folder / "FUN_ECOFIN_q_returns.csv"                      
            ret_factor = pd.read_csv(F_ret_file, header=0, index_col=0)
                       
           
        all_FandA = []
        coef_acc = []
               
        # asset_ret_series = asset_ret.iloc[:, i+1].dropna()
        ret_asset_series = ret_equity.iloc[:, i].dropna()
        asset_folder = os.path.join( result_SiGGA , asset_ticker )
       
        Path(asset_folder).mkdir(parents=True, exist_ok=True)   
        
        
        best_ga_file = Path(asset_folder) / "selection.csv"
        

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
        
        genome = Genome(standardized_returns, lead_lag=1, signature_level=3,best_ga_file=best_ga_file)    
        population_size = 100
        num_generations = 100              
                        
        ga_instance = pygad.GA(sol_per_pop=population_size,
                               num_generations=num_generations,                        
                               num_parents_mating=int(60),  #population_size*0.8
                               parent_selection_type="tournament",
                               fitness_func=genome.fitness_func,           
                               gene_space = range(genome.genome_length-1),
                               num_genes=genome.chromosone_length,
                               mutation_num_genes=1,
                               gene_type=int,
                               on_generation=genome.on_generation,
                               on_start=genome.on_start,
                               on_stop=genome.on_stop,
                               allow_duplicate_genes=False,
                               keep_elitism=2)
        
        # RUN GA        
        ga_instance.run()
        ga_instance.on_stop = True
        ga_instance.plot_fitness()
        
                
    # solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))        
    # print(','.join(list(genome.genes[solution.astype(bool)])))

def print_func(text, base_file_path=None):
    # Determine the directory from the base file path
    if base_file_path:
        solution_generations_file = Path(base_file_path).parent / "solution_generations.txt"
    else:
        solution_generations_file = "solution_generations.txt"
    
    # Write the output to the file
    with open(solution_generations_file, "a") as file:
        file.write(f"{text}")


def get_base_scale(c):
    return 100+(c/ pow(10, math.floor(math.log10(max(abs(c)))))/1)

def get_returns(c):
    return np.diff(np.log(c))    

def sliding_window(elements, window_size):
    # https://dock2learn.com/tech/implement-a-sliding-window-using-python/
    if len(elements) <= window_size:
        return elements
    for i in range(len(elements) - window_size + 1):
        yield elements[i:i+window_size]

def accuracy(predictions, y, nb_parameters):
    muu =np.mean(y)    
    # R2 =1 -( ((y -predictions).T@(y -predictions))) /( ((y -muu).T@(y -muu)))
    # mae =mean_absolute_error(y, predictions)
    # mse =mean_squared_error(y, predictions)
    # rmse =np.sqrt(mse)
    r2 =r2_score(y, predictions)     
    sse_df      =len(y) -nb_parameters -1
    sst_df =len(y) -1    
    # r2_adj =1 -((y - predictions).T @ (y - predictions))/sse_df  /((y -muu).T @ (y -muu)/sst_df )
    # r2_adj =1 - (1 - r2) * (len(y) - 1) / (len(y) - nb_parameters - 1)
    r2_adj =1 - (1 - r2) * sst_df / sse_df
    
    # return r2
    return r2_adj


class SigLearn:

    def __init__(self, order=2, alpha=0.1,chromosone_length=5):
        if not isinstance(order, numbers.Integral) or order<1:
            raise NameError('The order must be a positive integer.')
        if not isinstance(alpha, numbers.Real) or alpha<=0.0:
            raise NameError('Alpha must be a positive real.')
           
        self.order=int(order)
        self.reg=None
        self.alpha=alpha
        self.chromosone_length =chromosone_length
        # self.nb_parameters =nb_parameters
       
    def train(self, x, y):
        '''
        Trains the model using signatures.
       
        x: list of inputs, where each element of
            the list is a list of tuples.
        y: list of outputs.
        '''

        # We check that x and y have appropriate types
        if x is None or y is None:
            return
        if not (type(x) is list or type(x) is tuple) or not (type(y) is list or type(y) is tuple):
            raise NameError('Input and output must be lists or tuples.')
        if len(x)!=len(y):
            raise NameError('The number of inputs and the number of outputs must coincide.')
        ###
       
        X=[list(sig(np.array(stream), self.order)) for stream in x]
        # X =get_sig_functions.path_to_sig(np.array(x), self.order)         
        # https://git.maths.ox.ac.uk/perez/Digit-recognition-using-signatures/-/blob/master/sigLearn.py
        # self.reg = RandomForestRegressor(n_estimators=10, oob_score=False)
        # nb_parameters = math.floor( ([len(element) for element in X][0]) /5 )
        # nb_parameters = 10
        self.reg = PLSRegression(n_components =self.chromosone_length *2)
        # self.reg = PLSRegression(n_components =10) 
        # Signature level =1, we have n_components =5  +1 =6
        # self.reg = PLSRegression(n_components =6) 
        # self.reg = linear_model.Lasso(alpha = self.alpha)
        self.reg.fit(X, y)
        
    def train_dyadic(self, x, y, sig_dimension):
        '''
        Trains the model using signatures.
       
        x: list of inputs, where each element of
            the list is a list of tuples.
        y: list of outputs.
        '''

        # We check that x and y have appropriate types
        if x is None or y is None:
            return
        if not (type(x) is list or type(x) is tuple) or not (type(y) is list or type(y) is tuple):
            raise NameError('Input and output must be lists or tuples.')
        if len(x)!=len(y):
            raise NameError('The number of inputs and the number of outputs must coincide.')
        ###
        X=[list(get_sig_functions.path_to_dyadic_sig(np.array(stream), self.order,1, sig_dimension)) for stream in x]        
        # https://git.maths.ox.ac.uk/perez/Digit-recognition-using-signatures/-/blob/master/sigLearn.py
        # self.reg = RandomForestRegressor(n_estimators=10, oob_score=False)
        # nb_parameters = math.floor( ([len(element) for element in X][0]) /5 )
        nb_parameters = 10
        self.reg = PLSRegression(n_components =nb_parameters)
        # self.reg = linear_model.Lasso(alpha = self.alpha)
        self.reg.fit(X, y)    
        
       
       
    def train2(self, x, y):
        self.reg = RandomForestRegressor(n_estimators=10, oob_score=False)
        # self.reg = linear_model.Lasso(alpha = self.alpha)
        self.reg.fit(x, y)
       
    def predict2(self, x):
        return self.reg.predict(x)

       
    def predict(self, x):
        '''
        Predicts the outputs of the inputs x using the the
        pre-trained model.
       
        x: list of inputs, where each element of
            the list is a list of tuples.

        Returns:

        list of predicted outputs.
        '''
        if self.reg is None:
            raise NameError('The model is not trained.')

        X=[list(sig(np.array(stream), self.order))
                for stream in x]
 
        return self.reg.predict(X)
    
   
    def predict_dyadic(self, x, sig_dimension):
        '''
        Predicts the outputs of the inputs x using the the
        pre-trained model.
       
        x: list of inputs, where each element of
            the list is a list of tuples.

        Returns:

        list of predicted outputs.
        '''
        if self.reg is None:
            raise NameError('The model is not trained.')

        # X=[list(sig(np.array(stream), self.order)) for stream in x]                 
        X=[list(get_sig_functions.path_to_dyadic_sig(np.array(stream), self.order,1, sig_dimension)) for stream in x] 
        return self.reg.predict(X)
   

# set parameters

class Genome:
   
   
    def __init__(self,standardized_returns, lead_lag=1, signature_level=3, chromosone_length=5, best_ga_file=None  ):
        self.standardized_returns = standardized_returns
        self.standardized_factor_returns = standardized_returns.iloc[:,1:]
        self.genes = standardized_returns.columns[1:]
        self.gen_time = self.total_time = 0        
        self.signature_level = signature_level
        self.lead_lag = lead_lag
        self.chromosone_length=chromosone_length
        self.genome_length=len(self.standardized_returns.columns)-1
        self.best_ga_file=best_ga_file
        self.best_solution_fitness = None
               

    def fitness_func(self, ga_instance, solution, solution_idx):
        solution = solution.copy()
       
        df = self.standardized_factor_returns.iloc[:,solution]
       
        # r = get_sig_functions.time_embedding(df) # time agumentation
        # r_r = get_sig_functions.basepoint_time_embedding(df) # time + basepoint agumentation
        r_r_r =get_sig_functions.leadlag_embedding(df, ll =2) # time + lead lag agumentation
        
        window_size = 6
        sw_gen = sliding_window(r_r_r,window_size)
        X = [w for w in sw_gen]
       
        #
        y = self.standardized_returns.iloc[:,0][len(self.standardized_returns)-len(X):].tolist()
               
        indexes = list(range(0,len(X)))        
       
        train_size= int(len(y)*.7) # use 70% data for training and 30% for testing
        training_indexes=indexes[0:train_size]
        testing_indexes=indexes[train_size:]
       
        X_train = [ X[ndx] for ndx in training_indexes ]
        X_test =  [ X[ndx] for ndx in testing_indexes]
                      
        y_train = [ y[ndx] for ndx in training_indexes ]
        y_test =  [ y[ndx] for ndx in testing_indexes ]
        
        # X_train_dyadic = r_r[training_indexes]
        # X_test_dyadic =  r_r[testing_indexes]
        
        # y_train_dyadic =r_r[training_indexes]
        # y_test_dyadic =r_r[testing_indexes]
        
               
        # The model is trained.
        # sig_dimension1  =int( (  len(r[0]) **(self.signature_level +1)  -1  )  /(len(r[0])  -  1)  -1 )
        sig_dimension  =int( (  len(r_r_r[0]) **(self.signature_level +1)  -1  )  /(len(r_r_r[0])  -  1)  -1 )
        # sig_dimension =isig.siglength(len(r[0]), (self.signature_level))
        model=SigLearn(3)      
        # import time      
        # st = time.time()
        # model.train(X_train, y_train)
        model.train_dyadic(X_train, y_train,sig_dimension)
        # predictions=model.predict(X_test) 
        predictions=model.predict_dyadic( X_test,sig_dimension ) 
        theAccuracy = accuracy( predictions,y_test, self.chromosone_length *2 )
             
        return ( theAccuracy )

   
    def on_start(self, ga_instance):
        self.gen_start = self.ga_start= time.perf_counter_ns()
        # print("on_start")
               
  
    def on_generation(self, ga_instance):
        now = time.perf_counter_ns()
        self.gen_time = (now-self.gen_start)/1000000
        self.total_time = (now-self.ga_start)/1000000
       
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        
        print("Gen: {0}, Best={1}, gen_time={2}, total_time={3}, chromosone={4}\n".format(
            ga_instance.generations_completed,
            solution_fitness,
            timedelta(milliseconds =self.gen_time),
            timedelta(milliseconds =self.total_time),
            ','.join(list(self.genes[solution]))))                                  
        
        
        # # Print the population of the current generation
        # print("Population at Generation {0}:\n{1}\n".format(
        # ga_instance.generations_completed,
        # ga_instance.population))
        
        
        # # Print the population and their fitness values
        # print("Population and Fitnesses at Generation {0}:\n".format(ga_instance.generations_completed))
        # for individual, fitness in zip(ga_instance.population, ga_instance.last_generation_fitness):
        #     print("Individual: {0}, Fitness: {1}".format(individual, fitness))

        
        # {0}: is the Gen number. {1}: start time of Gen. {2}: total time on Gen. {4}: Chromosone selected
        # Gen: {0}, Best={1}, gen_time={2}, total_time={3}, chromosome={4}
        formatted_output = "{0}, {1}, {2}, {3}, {4}\n".format(
        ga_instance.generations_completed,
        solution_fitness,
        timedelta(milliseconds=self.gen_time),
        timedelta(milliseconds=self.total_time),
        ','.join(list(self.genes[solution])))
                          
        # Print to console
        # print(formatted_output)
        
        # Save to the solution_generations.txt in the same directory as best_ga_file
        print_func(formatted_output, base_file_path=self.best_ga_file)        
        self.gen_start=now
       

    def on_stop(self, ga_instance, last_population_fitness):
        solution, solution_fitness, solution_idx = ga_instance.best_solution(last_population_fitness)
        print(f"Fitness value of the best solution = {solution_fitness}")
        best_solution = ','.join(list(self.genes[solution]))
        print(best_solution)
        # print(ga_instance.best_solutions_fitness)
        
        if self.best_ga_file is not None:
            string_to_write = f"{solution_fitness},{best_solution}\n"
                                                          
            with open(self.best_ga_file, 'w') as file:
                file.write(string_to_write)



if __name__ == "__main__":  
    options=dict()  
    main("2021-03-31", options)