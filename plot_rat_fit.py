import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

data = pd.read_csv('P_rat_2_0.01.csv')

#format output file
PE = data['PE']
PE_formatted = []
for value in PE:
    cleaned_value = value.replace('[', '').replace(']', '').strip()
    array = cleaned_value.split()
    formatted_array = [round(float(num), 5) for num in array]
    PE_formatted.append(formatted_array)

ag = [item[0] for item in PE_formatted]
an = [item[1] for item in PE_formatted]
data['ag'] = ag
data['an'] = an
data.to_csv('Updated_output_file.csv', index=False)
print(data.head())

#The following portion is set to plot *ag* for CIE vs AIR groups for *male* rats. 
#Change these parameters as desired.

#SET 1: ag, male, CIE
data = pd.read_csv('Updated_output_file.csv')
filtered_data = data[(data['Group'] == 1) & (data['Sex'] == 1)] #Sex =1 => male; =2 => female
#make dictionary output to store results. Named to reflect grouping. 
#Change with groups to avoid confusion. ag=> parameter, m=> male, g1=> CIE
pe_dict_ag_m_g1_ss = {
    (row['ID'], row['Session']): row['ag']
    for index, row in filtered_data.iterrows()}

#SET 2: ag, male, AIR
data = pd.read_csv('Updated_output_file.csv')
filtered_data = data[(data['Group'] == 2) & (data['Sex'] == 1)] #Sex =1 => male; =2 => female
#naming convention: ag=> parameter, m=> male, g2=> AIR
pe_dict_ag_m_g2_ss = {
    (row['ID'], row['Session']): row['ag']
    for index, row in filtered_data.iterrows()}


# plot + t_test
cie_values = list(pe_dict_ag_m_g1_ss.values())
air_values = list(pe_dict_ag_m_g2_ss.values())

cie_mean = np.mean(cie_values)
air_mean = np.mean(air_values)

t_stat, p_value = ttest_ind(cie_values, air_values)

plt.figure(figsize=(8, 6))

plt.scatter(['CIE'] * len(cie_values), cie_values, color='blue', label='CIE', alpha=0.6)
plt.scatter(['AIR'] * len(air_values), air_values, color='red', label='AIR', alpha=0.6)

plt.scatter('CIE', cie_mean, color='blue', marker='D', s=100, label='CIE Mean')
plt.scatter('AIR', air_mean, color='red', marker='D', s=100, label='AIR Mean')

plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1))

plt.xticks(['CIE', 'AIR'])

plt.xlabel('Groups')
plt.ylabel('ag_recovered')
plt.title('ag_male')
plt.legend()

plt.text(0.5, 0.1, f't={t_stat:.2f}, p={p_value:.3f}', transform=plt.gca().transAxes, ha='center', fontsize=10)

plt.grid()
plt.tight_layout()
plt.savefig('ag_male.png')
