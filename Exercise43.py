import pandas as pd
import scipy.stats as stats
import statsmodels
import statsmodels.api 
import statsmodels.formula.api as smf

df = pd.read_csv('../Statistical Data Analysis 4/simulated_data_2_5.csv')

#Test all columns for normality with shapiro
print(stats.shapiro(df['F']))
print(stats.shapiro(df['G']))
print(stats.shapiro(df['H']))
print(stats.shapiro(df['I']))

# I is not normal 

#create a subset dataframe from columns F, G, H
data = df[['F','G','H']]

#modify data for anova
modified_data = pd.melt(data)
modified_data.columns = ['column', 'value']

#creaete model
model = smf.ols('value ~ C(column)', data=modified_data).fit()

#ANOVA
anova = statsmodels.api.stats.anova_lm(model, typ=1)
print(anova)

#subset dataframe from columns F, G, I
data = df[['F','G','I']]

#kruskal wallis test on the data 
print("P-Value:" ,stats.kruskal(data['F'], data['G'], data['I'])[1])