# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:56:17 2025

@author: ecoramy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[] sigga_3aug3deg_barchart
# def create_barStacked_wnd(comparaison_table):
#     # data = {
#     #     'Aug \ Window': [
#     #         'bp_time_sig_D1', 'bp_time_sig_D2', 'bp_time_sig_D3',
#     #         'LL1_time_sig_D1', 'LL1_time_sig_D2', 'LL1_time_sig_D3',
#     #         'LL2_time_sig_D1', 'LL2_time_sig_D2', 'LL2_time_sig_D3'
#     #     ],
#     #     'expand > slide': [69.306931, 56.435644, 66.336634, 64.356436, 46.534653, 52.475248, 64.356436, 41.584158, 60.396040],
#     #     'expand > dyadic': [71.287129, 52.475248, 46.534653, 57.425743, 48.514851, 70.297030, 81.188119, 44.554455, 67.326733],
#     #     'slide > dyadic': [63.366337, 46.534653, 39.603960, 56.435644, 57.425743, 64.356436, 73.267327, 53.465347, 58.415842]
#     # }
    
#     # # Convert to DataFrame
#     df = pd.DataFrame(comparaison_table)
    
#     # Plot configuration
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Define bar positions and bar width
#     x = range(len(df))
#     bar_width = 0.8
    
#     # Plot stacked bars
#     p1 = ax.bar(x, df['expand > slide'], width=bar_width, label='Expand > Slide')
#     p2 = ax.bar(x, df['expand > dyadic'], width=bar_width, bottom=df['expand > slide'], label='Expand > Dyadic')
#     p3 = ax.bar(x, df['slide > dyadic'], width=bar_width, bottom=df['expand > slide'] + df['expand > dyadic'], label='Slide > Dyadic')
    
#     # Add values on bars
#     def add_values_on_bars(bars, y_offset=0):
#         for bar in bars:
#             height = bar.get_height()
#             if height > 0:
#                 ax.text(
#                     bar.get_x() + bar.get_width() / 2,  # Center of the bar
#                     bar.get_y() + height + y_offset,  # Top of the bar
#                     f'{height:.2f}',
#                     ha='center', va='bottom', fontsize=8
#                 )
    
#     add_values_on_bars(p1)
#     add_values_on_bars(p2)
#     add_values_on_bars(p3)
    
#     # Set labels and title
#     # ax.set_xlabel('Aug per Window Method')
#     ax.set_ylabel('Percentage')
#     ax.set_title('Comparison of Window Methods')
#     ax.set_xticks(x)
#     ax.set_xticklabels(df['Aug per Window Method'], rotation=45, ha='right')
#     ax.legend()
    
#     # Show the plot
#     plt.tight_layout()
#     plt.show()

#     return

# In[] same as above w/o added values on the bars, and manual
# # Create the DataFrame
# data = {
#     "Aug \\ Window": [
#         "bp_time_sig_D1",
#         "bp_time_sig_D2",
#         "bp_time_sig_D3",
#         "LL1_time_sig_D1",
#         "LL1_time_sig_D2",
#         "LL1_time_sig_D3",
#         "LL2_time_sig_D1",
#         "LL2_time_sig_D2",
#         "LL2_time_sig_D3",
#     ],
#     "expand > slide": [
#         69.306931,
#         56.435644,
#         66.336634,
#         64.356436,
#         46.534653,
#         52.475248,
#         64.356436,
#         41.584158,
#         60.396040,
#     ],
#     "expand > dyadic": [
#         71.287129,
#         52.475248,
#         46.534653,
#         57.425743,
#         48.514851,
#         70.297030,
#         81.188119,
#         44.554455,
#         67.326733,
#     ],
#     "slide > dyadic": [
#         63.366337,
#         46.534653,
#         39.603960,
#         56.435644,
#         57.425743,
#         64.356436,
#         73.267327,
#         53.465347,
#         58.415842,
#     ],
# }

# df = pd.DataFrame(data)

# # Set up the bar chart
# bar_width = 0.6
# indices = np.arange(len(df))

# fig, ax = plt.subplots(figsize=(10, 6))

# # Stacked bar chart
# ax.bar(
#     indices, df["expand > slide"], bar_width, label="Expand > Slide"
# )
# ax.bar(
#     indices, df["expand > dyadic"], bar_width, bottom=df["expand > slide"], label="Expand > Dyadic"
# )
# ax.bar(
#     indices,
#     df["slide > dyadic"],
#     bar_width,
#     bottom=df["expand > slide"] + df["expand > dyadic"],
#     label="Slide > Dyadic",
# )

# # Customizing the chart
# ax.set_xlabel("Category", fontsize=12)
# ax.set_ylabel("Percentage (%)", fontsize=12)
# ax.set_title("Comparison of Windows by Category", fontsize=14)
# ax.set_xticks(indices)
# ax.set_xticklabels(df["Aug \\ Window"], rotation=45, ha="right")
# ax.legend()

# plt.tight_layout()
# plt.show()

# In[] same as sigga_3aug3deg_barchart, manual
# data = {
#     'Aug_Window': [
#         'bp_time_sig_D1', 'bp_time_sig_D2', 'bp_time_sig_D3',
#         'LL1_time_sig_D1', 'LL1_time_sig_D2', 'LL1_time_sig_D3',
#         'LL2_time_sig_D1', 'LL2_time_sig_D2', 'LL2_time_sig_D3'
#     ],
#     'expand > slide': [69.306931, 56.435644, 66.336634, 64.356436, 46.534653, 52.475248, 64.356436, 41.584158, 60.396040],
#     'expand > dyadic': [71.287129, 52.475248, 46.534653, 57.425743, 48.514851, 70.297030, 81.188119, 44.554455, 67.326733],
#     'slide > dyadic': [63.366337, 46.534653, 39.603960, 56.435644, 57.425743, 64.356436, 73.267327, 53.465347, 58.415842]
# }

# # # these results are taken from excel file called 3_results_sigga_windows, workesheet { across| 3 windows }
# # # data = {
# # #     'Aug_Window': [
# # #         'bp time', 
# # #         'LL1 time', 
# # #         'LL2 time'
# # #     ],
# # #     'expand > slide': [64.0264026, 54.4554455, 55.4455446],
# # #     'expand > dyadic': [56.765676, 58.7458746, 64.3564356],
# # #     'slide > dyadic': [49.8349835, 59.4059406, 61.7161716]
# # # }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Plot configuration
# fig, ax = plt.subplots(figsize=(10, 6))

# # Define bar positions and bar width
# x = range(len(df))
# bar_width = 0.8

# # Plot stacked bars
# p1 = ax.bar(x, df['expand > slide'], width=bar_width, label='Expand > Slide')
# p2 = ax.bar(x, df['expand > dyadic'], width=bar_width, bottom=df['expand > slide'], label='Expand > Dyadic')
# p3 = ax.bar(x, df['slide > dyadic'], width=bar_width, bottom=df['expand > slide'] + df['expand > dyadic'], label='Slide > Dyadic')

# # Add values on bars
# def add_values_on_bars(bars, y_offset=0):
#     for bar in bars:
#         height = bar.get_height()
#         if height > 0:
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,  # Center of the bar
#                 bar.get_y() + height + y_offset,  # Top of the bar
#                 f'{height:.2f}',
#                 ha='center', va='bottom', fontsize=8
#             )

# add_values_on_bars(p1)
# add_values_on_bars(p2)
# add_values_on_bars(p3)

# # Set labels and title
# # ax.set_xlabel('Aug \ Window')
# ax.set_ylabel('Percentage')
# ax.set_title('Comparison of Window Methods')
# ax.set_xticks(x)
# ax.set_xticklabels(df['Aug_Window'], rotation=45, ha='right')
# ax.legend()

# # Show the plot
# plt.tight_layout()
# plt.show()



# In[] sigga_win_barchart, manual
# these results are taken from excel file called 3_results_sigga_windows, workesheet { across| 3 windows }
# data = {
#     'Aug_Method': [
#         'bp time', 
#         'LL1 time', 
#         'LL2 time'
#     ],
#     'expand > slide': [64.0264026, 54.4554455, 55.4455446],
#     'expand > dyadic': [56.765676, 58.7458746, 64.3564356],
#     'slide > dyadic': [49.8349835, 59.4059406, 61.7161716]
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Plot configuration
# fig, ax = plt.subplots(figsize=(4, 4))

# # Define bar positions and bar width
# x = range(len(df))
# bar_width = 0.4

# # Plot stacked bars
# p1 = ax.bar(x, df['expand > slide'], width=bar_width, label='Expand > Slide')
# p2 = ax.bar(x, df['expand > dyadic'], width=bar_width, bottom=df['expand > slide'], label='Expand > Dyadic')
# p3 = ax.bar(x, df['slide > dyadic'], width=bar_width, bottom=df['expand > slide'] + df['expand > dyadic'], label='Slide > Dyadic')

# # Add values on bars
# def add_values_on_bars(bars, y_offset=0):
#     for bar in bars:
#         height = bar.get_height()
#         if height > 0:
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,  # Center of the bar
#                 bar.get_y() + height + y_offset,  # Top of the bar
#                 f'{height:.2f}',
#                 ha='center', va='bottom', fontsize=8
#             )

# add_values_on_bars(p1)
# add_values_on_bars(p2)
# add_values_on_bars(p3)

# # Set labels and title
# # ax.set_xlabel('Aug \ Window')
# ax.set_ylabel('Percentage')
# ax.set_title('Comparison of Window Methods')
# ax.set_xticks(x)
# ax.set_xticklabels(df['Aug_Method'], rotation=45, ha='right')
# ax.legend()

# # Show the plot
# plt.tight_layout()
# plt.show()

# In[] sigga_aug_barchart, manual
# these results are taken from excel file called 3_results_sigga_windows, workesheet { across| aug }
# data = {
#     'Wnd_Method': [
#         'slide', 
#         'expand', 
#         'dyadic'
#     ],
#     'LL2 time > bp time': [80.528053, 71.617162, 68.976898],
#     'LL2 time > LL1 time': [53.465347, 56.435644, 50.165017],
#     'LL1 time > bp time': [81.188119, 68.976898, 70.957096]
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Plot configuration
# fig, ax = plt.subplots(figsize=(4, 4))

# # Define bar positions and bar width
# x = range(len(df))
# bar_width = 0.4

# # Plot stacked bars
# p1 = ax.bar(x, df['LL2 time > bp time'], width=bar_width, label='LL2 time > bp time')
# p2 = ax.bar(x, df['LL2 time > LL1 time'], width=bar_width, bottom=df['LL2 time > bp time'], label='LL2 time > LL1 time')
# p3 = ax.bar(x, df['LL1 time > bp time'], width=bar_width, bottom=df['LL2 time > bp time'] + df['LL2 time > LL1 time'], label='LL1 time > bp time')

# # Add values on bars
# def add_values_on_bars(bars, y_offset=0):
#     for bar in bars:
#         height = bar.get_height()
#         if height > 0:
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,  # Center of the bar
#                 bar.get_y() + height + y_offset,  # Top of the bar
#                 f'{height:.2f}',
#                 ha='center', va='bottom', fontsize=8
#             )

# add_values_on_bars(p1)
# add_values_on_bars(p2)
# add_values_on_bars(p3)

# # Set labels and title
# # ax.set_xlabel('Aug \ Window')
# ax.set_ylabel('Percentage')
# ax.set_title('Comparison of Aug Methods')
# ax.set_xticks(x)
# ax.set_xticklabels(df['Wnd_Method'], rotation=45, ha='right')
# ax.legend()

# # Show the plot
# plt.tight_layout()
# plt.show()


# In[] sigga_deg_barchart, manual
# these results are taken from excel file called 3_results_sigga_windows, workesheet { across| deg }
# data = {
#     'Wnd_Method': [
#         'slide', 
#         'expand', 
#         'dyadic'
#     ],
#     'Sig D3 > Sig D1': [53.79538, 48.18482, 61.38614],
#     'Sig D3 > Sig D2': [44.88449, 57.09571, 42.90429],
#     'Sig D2 > Sig D1': [52.14521, 41.91419, 65.01650]
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Plot configuration
# fig, ax = plt.subplots(figsize=(4, 4))

# # Define bar positions and bar width
# x = range(len(df))
# bar_width = 0.4

# # Plot stacked bars
# p1 = ax.bar(x, df['Sig D3 > Sig D1'], width=bar_width, label='sigD3 > sigD1')
# p2 = ax.bar(x, df['Sig D3 > Sig D2'], width=bar_width, bottom=df['Sig D3 > Sig D1'], label='Sig D3 > Sig D2')
# p3 = ax.bar(x, df['Sig D2 > Sig D1'], width=bar_width, bottom=df['Sig D3 > Sig D1'] + df['Sig D3 > Sig D2'], label='Sig D2 > Sig D1')

# # Add values on bars
# def add_values_on_bars(bars, y_offset=0):
#     for bar in bars:
#         height = bar.get_height()
#         if height > 0:
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,  # Center of the bar
#                 bar.get_y() + height + y_offset,  # Top of the bar
#                 f'{height:.2f}',
#                 ha='center', va='bottom', fontsize=8
#             )

# add_values_on_bars(p1)
# add_values_on_bars(p2)
# add_values_on_bars(p3)

# # Set labels and title
# # ax.set_xlabel('Aug \ Window')
# ax.set_ylabel('Percentage')
# ax.set_title('Comparison of Sig Levels (Deg)')
# ax.set_xticks(x)
# ax.set_xticklabels(df['Wnd_Method'], rotation=45, ha='right')
# ax.legend()

# # Show the plot
# plt.tight_layout()
# plt.show()



# In[]    
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Sample data as a dictionary
# data = {
#     "adj-r2": [0.0968498482999541, -0.0047938083504783, -0.7138012368739128, 0.2239501369536991, -0.5653080425226553],
#     "gene1": ["USAMKT", "LQD", "USAMKT", "USAGASSC", "deferredrev"],
#     "gene2": ["USANLTTF", "USAGSP", "USAFER", "de", "receivables"],
#     "gene3": ["USAEXPX", "DVY", "accoci", "IJH", "gp"],
#     "gene4": ["USARSY", "USACARS", "capex", "USAPHS", "retearn"],
#     "gene5": ["assetsnc", "rnd", "USAPFED", "USAMORTG", "rnd"],
#     "stock_ticker": ["AAPL", "MSFT", "AMZN", "AMGN", "GOOGL"]
# }

# # Load data into DataFrame
# df = pd.DataFrame(data)

# # Define categories and tagging
# categories = {
#     "ecofin": ["deferredrev", "USAMKT", "USAGASSC", "de"],
#     "etf": ["LQD", "IJH", "DVY"],
#     "fdm": ["assetsnc", "receivables", "capex", "rnd", "gp"]
# }

# # Function to tag variables
# def tag_variable(value):
#     for category, values in categories.items():
#         if value in values:
#             return f"{category}_{value}"
#     return value

# # Apply tagging to gene1 through gene5
# for col in ["gene1", "gene2", "gene3", "gene4", "gene5"]:
#     df[col] = df[col].apply(tag_variable)

# # Reshape data for heatmap
# df_melted = df.melt(id_vars=["adj-r2", "stock_ticker"], value_vars=["gene1", "gene2", "gene3", "gene4", "gene5"],
#                     var_name="gene", value_name="variable")

# # Pivot table to prepare for heatmap
# heatmap_data = df_melted.pivot_table(index="variable", columns="stock_ticker", values="adj-r2", aggfunc="mean")

# # Plot heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f", linewidths=.5)
# plt.title("Heatmap of adj-r2 by Variables and Stock Tickers")
# plt.show()    

# In[]
# Sample data as a dictionary

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# data = {
#     "adj-r2": [0.0968498482999541, -0.0047938083504783, -0.7138012368739128, 0.2239501369536991, -0.5653080425226553],
#     "gene1": ["USAMKT", "LQD", "USAMKT", "USAGASSC", "deferredrev"],
#     "gene2": ["USANLTTF", "USAGSP", "USAFER", "de", "receivables"],
#     "gene3": ["USAEXPX", "DVY", "accoci", "IJH", "gp"],
#     "gene4": ["USARSY", "USACARS", "capex", "USAPHS", "retearn"],
#     "gene5": ["assetsnc", "rnd", "USAPFED", "USAMORTG", "rnd"],
#     "stock_ticker": ["AAPL", "MSFT", "AMZN", "AMGN", "GOOGL"]
# }

# # Load data into DataFrame
# df = pd.DataFrame(data)

# # Define categories and tagging
# categories = {
#     "ecofin": ["deferredrev", "USAMKT", "USAGASSC", "de"],
#     "etf": ["LQD", "IJH", "DVY"],
#     "fdm": ["assetsnc", "receivables", "capex", "rnd", "gp"]
# }

# # Function to tag variables
# def tag_variable(value):
#     for category, values in categories.items():
#         if value in values:
#             return f"{category}_{value}"
#     return value

# # Apply tagging to gene1 through gene5
# for col in ["gene1", "gene2", "gene3", "gene4", "gene5"]:
#     df[col] = df[col].apply(tag_variable)

# # Reshape data for heatmap
# df_melted = df.melt(id_vars=["adj-r2"], value_vars=["gene1", "gene2", "gene3", "gene4", "gene5"],
#                     var_name="gene", value_name="variable")

# # Aggregate adj-r2 by variable
# heatmap_data = df_melted.groupby("variable")["adj-r2"].mean().reset_index()

# # Prepare heatmap data
# heatmap_data = heatmap_data.pivot_table(index="variable", values="adj-r2")

# # Plot heatmap
# plt.figure(figsize=(8, 12))
# sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f", linewidths=.5)
# plt.title("Heatmap of adj-r2 by Variables")
# plt.xlabel("adj-r2")
# plt.ylabel("Tagged Variables")
# plt.show()


# In[]

# import matplotlib.pyplot as plt
# import numpy as np

# # Data from the table
# categories = ['ECOFIN', 'ETF', 'FDM']
# x_labels = ['Top 100', 'Center 100', 'Bottom 100', 'Bottom 140']
# data = np.array([
#     [50, 45, 44, 40],  # ECOFIN
#     [33, 37, 40, 37],  # ETF
#     [17, 18, 16, 23]   # FDM
# ])

# # Create the plot
# fig, ax = plt.subplots(figsize=(10, 6))

# # Set the width of each bar and the positions of the bars
# width = 0.25
# x = np.arange(len(x_labels))

# # Plot bars for each category
# for i, category in enumerate(categories):
#     ax.bar(x + i*width, data[i], width, label=category)

# # Customize the plot
# ax.set_ylabel('Percentage')
# ax.set_title('% count by category across top, center, bottom adj-R², expand window')
# ax.set_xticks(x + width)
# ax.set_xticklabels(x_labels)
# ax.legend()

# # Add value labels on top of each bar
# for i in range(len(categories)):
#     for j in range(len(x_labels)):
#         ax.text(j + i*width, data[i][j], str(data[i][j]), 
#                 ha='center', va='bottom')

# # Adjust layout and display the plot
# plt.tight_layout()
# plt.show()


# In[]
# import matplotlib.pyplot as plt
# import numpy as np

# # Data from the table
# categories = ['ECOFIN', 'ETF', 'FDM']
# x_labels = ['Top 100', 'Center 100', 'Bottom 100', 'Bottom 140']
# # data = np.array([
# #     [50, 45, 44, 40],  # ECOFIN
# #     [33, 37, 40, 37],  # ETF
# #     [17, 18, 16, 23]   # FDM
# # ])

# # data = np.array([
# #     [52, 43, 47, 47],  # ECOFIN
# #     [33, 36, 34, 34],  # ETF
# #     [15, 20, 19, 19]   # FDM
# # ])

# data = np.array([
#     [48, 39, 42, 43],  # ECOFIN
#     [32, 33, 34, 34],  # ETF
#     [20, 28, 24, 23]   # FDM
# ])

# # Create the plot
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot lines for each category
# for i, category in enumerate(categories):
#     ax.plot(x_labels, data[i], marker='o', label=category)

# # Customize the plot
# ax.set_ylabel('Percentage')
# ax.set_xlabel('Adjusted R² Groups')
# # ax.set_title('% count by category across top, center, bottom adj-R², expand window')
# # ax.set_title('% count by category across top, center, bottom adj-R², slide window')
# ax.set_title('% count by category across top, center, bottom adj-R², dyadic window')
# ax.legend()

# # Add value labels for each point
# for i, category in enumerate(categories):
#     for j, value in enumerate(data[i]):
#         ax.annotate(str(value), (j, value), textcoords="offset points", 
#                     xytext=(0,10), ha='center')

# # Set y-axis to start from 0
# ax.set_ylim(bottom=0)

# # Add grid for better readability
# ax.grid(True, linestyle='--', alpha=0.7)

# # Adjust layout and display the plot
# plt.tight_layout()
# plt.show()
