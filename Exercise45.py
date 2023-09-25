import pandas as pd
import scipy.stats as stats

columns = ['Surgery', 'Age', 'Hospital number', 'Rectal temperature', 'Pulse', 'Respiratory rate', 'Temperature of extremities', 'Peripheral pulse', 'Mucous membranes', 'Capillary refill time', 'Pain', 'Peristalsis', 'Abdominal distension', 'Nasogastric tube', 'Nasogastric reflux', 'Nasogastric reflux PH', 'Rectal examination', 'Abdomen', 'Packed cell volume', 'Total protein', 'Abdominocentesis appearance', 'Abdomcentesis total protein', 'Outcome', 'Surgical lesion', 'Type of lesion1', 'Type of lesion2', 'Type of lesion3', 'cp_data']
df = pd.read_csv('../Statistical Data Analysis 4/horse-colic.data.csv', names=columns, na_values='?', delimiter=r"\s+")

#create 2 subdataframes with first having the instances with surgery variable = 1 and the second having surgery variable = 2
had_surgery = df[df['Surgery'] == 1]
no_surgery = df[df['Surgery'] == 2]

#check pulse and rectal temperature variables for normality 
print('Had surgery:', stats.shapiro(had_surgery['Rectal temperature'].dropna()))
print('No surgery:', stats.shapiro(no_surgery['Rectal temperature'].dropna()))

print("Pulse", stats.shapiro(had_surgery['Pulse'].dropna()))
print("Pulse", stats.shapiro(no_surgery['Pulse'].dropna()))

#RECTAL TEMPERATURE
had_surgery_temp = had_surgery['Rectal temperature'].dropna()
no_surgery_temp = no_surgery['Rectal temperature'].dropna()
#perform mann whitney u test 
print("Mann Whitney U test temperature:", stats.mannwhitneyu(had_surgery_temp, no_surgery_temp))

#PULSE
had_surgery_pulse = had_surgery['Pulse'].dropna()
no_surgery_pulse = no_surgery['Pulse'].dropna()
#perform mann whitney u test
print("Mann Whitney U test pulse:", stats.mannwhitneyu(had_surgery_pulse, no_surgery_pulse))
#As the pulse p-value is less than 0.05 whe can reject the null hypothesis and conclude that there is a significant difference between the medians 

#AGE
had_surgery_age = had_surgery['Age'].dropna()
no_surgery_age = no_surgery['Age'].dropna()
ages = pd.DataFrame({'Had surgery': had_surgery_age, 'No surgery': no_surgery_age})
ages = ages.apply(pd.value_counts)
#perform chi squared test
print("Chi squared test:", stats.chi2_contingency(ages)[1])