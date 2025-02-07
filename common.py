# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 19:21:03 2023

@author: ecoramy
"""

from collections import Counter

FUND_TICKER = 'IVV'
BASE_DIR = "C:/Users/ecoramy/my_wdc/Polymodel-v3/"

OUR_STOCKS = {'AAPL', 'ADBE', 'MSFT', 'NFLX', 'SBUX', 'TJX', 'VZ', 'UPS', 'WST', 'XRAY', 'BA', 'BIIB', 'BLK', 'COST', 'DE', 'DIS', 'EBAY', 'DTE', 'EQIX', 'GE', 'FRT', 
               'INTC', 'IBM', 'HAS', 'INTU', 'J', 'MAR', 'NKE', 'PH', 'WELL', 'AMZN', 'JPM', 'JNJ', 'NVDA', 'PG', 'HD', 'T', 'CMCSA', 'NEE', 'UNP', 'ADP', 'COP', 'PLD', 
               'BXP', 'EQIX', 'NSC', 'ECL', 'ORLY', 'MGM', 'PNR', 'TSCO', }
# OUR_STOCKS = {'AAPL',}


stock_tickers = [
    "AAPL", "MSFT", "AMZN", "AMGN", "GOOGL", "BMY", "JPM", "JNJ", "GE", "UNH",
    "DIS", "PG", "NVDA", "HD", "IBM", "BAC", "SBUX", "INTC", "CMCSA", "VZ",
    "XOM", "NFLX", "ADBE", "CSCO", "T", "ABT", "KO", "PFE", "CVX", "PEP",
    "MRK", "CRM", "UPS", "INTU", "WMT", "TMO", "ACN", "TXN", "NKE", "MCD",
    "WFC", "MDT", "COST", "C", "HON", "QCOM", "LLY", "NEE", "UNP", "RTX",
    "DE", "RMD", "WELL", "BLL", "LEN", "KR", "TSCO", "MTD", "DLTR", "J",
    "ECL", "SIVB", "FITB", "MMM", "WY", "VIAC", "CBRE", "BBY", "TGT", "ZBRA",
    "DTE", "VFC", "AVB", "ED", "PNR", "AIZ", "CVS", "KIM", "BEN", "PVH", "DVA",
    "FLIR", "PBCT", "COG", "ROL", "BXP", "PH", "VNO", "SEE", "FRT", "BLK",
    "APA", "LEG", "GPS", "RL", "UNM", "HFC", "PRGO", "NOV", "ORLY", "AMAT"
]

# for partial_run("2020-09-30", options )
# stock_tickers = [
#     "AAPL", "MSFT", "AMZN", "AMGN", "GOOGL", "BMY", "JPM", "JNJ", "GE", "UNH",
#     "DIS", "PG", "NVDA", "HD", "IBM", "BAC", "SBUX", "INTC", "CMCSA", "VZ",
#     "XOM", "NFLX", "ADBE", "CSCO", "T", "ABT", "KO", "PFE", "CVX", "PEP",
#     "MRK", "CRM", "UPS", "INTU", "WMT", "TMO", "ACN", "TXN", "NKE", "MCD",
#     "WFC", "MDT", "COST", "C", "HON", "QCOM", "LLY", "NEE", "UNP", "RTX",
#     "DE", "RMD", "WELL", "BLL", "LEN", "KR", "TSCO", "MTD", "DLTR", "J",
#     "ECL", "SIVB", "FITB", "MMM", "WY", "VIAC", "CBRE", "BBY", "TGT", "ZBRA",
#     "DTE", "VFC", "AVB", "ED", "PNR", "AIZ", "CVS", "KIM", "BEN", "PVH", "DVA",
#     "FLIR", "PBCT", "COG", "ROL", "BXP", "PH", "VNO", "SEE", "FRT", "BLK",
#     "APA", "LEG", "GPS", "RL", "UNM", "HFC", "PRGO", "ORLY", "AMAT"
# ]

# for partial_run("2020-06-30", options )
# stock_tickers = [
#     "AAPL", "MSFT", "AMZN", "AMGN", "GOOGL", "BMY", "JPM", "JNJ", "GE", "UNH",
#     "DIS", "PG", "NVDA", "HD", "IBM", "BAC", "SBUX", "INTC", "CMCSA", "VZ",
#     "XOM", "NFLX", "ADBE", "CSCO", "T", "ABT", "KO", "PFE", "CVX", "PEP",
#     "MRK", "CRM", "UPS", "INTU", "WMT", "TMO", "ACN", "TXN", "NKE", "MCD",
#     "WFC", "MDT", "COST", "C", "HON", "QCOM", "LLY", "NEE", "UNP", "RTX",
#     "DE", "RMD", "WELL", "BLL", "LEN", "KR", "TSCO", "MTD", "DLTR", "J",
#     "ECL", "SIVB", "FITB", "MMM", "WY", "VIAC", "CBRE", "BBY", "TGT", "ZBRA",
#     "DTE", "VFC", "AVB", "ED", "PNR", "AIZ", "CVS", "KIM", "BEN", "PVH", "DVA",
#     "FLIR", "PBCT", "COG", "ROL", "BXP", "PH", "VNO", "SEE", "FRT", "BLK",
#     "LEG", "GPS", "RL", "UNM", "HFC", "PRGO", "ORLY", "AMAT"
# ]


stock_tickers_backup = [
    "AMGN", "DHR", "BMY", "BA", "LOW", "PM", "ORCL", "SBUX", "CAT", "AMAT",
    "UPS", "IBM", "RTX", "DE", "MS", "GE", "GS", "MMM", "BLK", "AMT", "INTU",
    "TGT", "CVS", "MU", "NOW"
]


# # get unique elements, 
# x =dict.fromkeys(stock_tickers)
# no_dupes = [x for n, x in enumerate(stock_tickers) if x not in stock_tickers[:n]]
# print(no_dupes) # [[1], [2], [3], [5]]

# counts = Counter(stock_tickers)
# duplicates = [item for item, count in counts.items() if count > 1]
# print(duplicates)