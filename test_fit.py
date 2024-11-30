import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Currently the code is set to visualise 5 parameter estimation. Follow comments to customise
#for 2 and 3 parameter estimations.

#pass your output file here
data = pd.read_csv('k_results_estimate_5p.csv') 

#use this format to visualise a file from inside the folders:
#data = pd.read_csv('k_results_estimate_ac_bg_bn/0.1_0.1_ac_bg_bn.csv') 

# extract and format data from output file
PE=data['PE']
PE_formatted=[]
for value in PE:
    cleaned_value = value.replace('[', '').replace(']', '').strip()
    array = cleaned_value.split()
    formatted_array = [round(float(num), 3) for num in array]
    PE_formatted.append(formatted_array)

GT=np.array(data['GT'])
GT_formatted=[]
for value in GT:
    tuple_value = ast.literal_eval(value)
    formatted_array = list(tuple_value)
    GT_formatted.append(formatted_array)

# plot (edit based on which parameters recovered, currently set to all 5)
para = ['ag', 'an', 'ac', 'bg', 'bn']
bounds = [(0, 1), (0, 1), (0, 0.5), (0, 5), (0, 5)]

for i in range(5):  #CHANGE 5 TO NUMBER OF PARAMETERS ESTIMATED
    y_array = [row[i] for row in GT_formatted]
    x_array = [row[i] for row in PE_formatted]
    
    lower, upper = bounds[i]
    ticks = np.arange(lower, upper*1.1, upper/10)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_array, y_array, color='red')
    
    plt.xlabel('Recovered')
    plt.ylabel('True')
    plt.title(f'Recovery of {para[i]}')
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.plot([lower, upper], [lower, upper], color='grey', linestyle='--')

    tick_size = (upper - lower) / 10
    plt.plot([lower, upper], [lower + 2 * tick_size, upper + 2 * tick_size], color='black', linestyle='-.', label='+2 Ticks')
    plt.plot([lower, upper], [lower - 2 * tick_size, upper - 2 * tick_size], color='black', linestyle='-.', label='-2 Ticks')

    plt.tight_layout()
    plt.savefig(f'Recovery_of_{para[i]}.png')
    plt.close()  
    
# obtain outliers
tick_size = [(upper - lower) / 10 for lower, upper in bounds]
outliers = set()

for i in range(5):  #CHANGE 5 TO NUMBER OF PARAMETERS ESTIMATED
    y_array = [row[i] for row in GT_formatted]
    x_array = [row[i] for row in PE_formatted]
    
    threshold = 2*tick_size[i]

    for x, y in zip(x_array, y_array):
        distance = abs(y - x)
        if distance > threshold:
            outliers.add((tuple(GT_formatted[x_array.index(x)]), tuple(PE_formatted[x_array.index(x)])))

do = list(outliers)
print(len(do))
print(do)  #print outliers